"""June 1 top-detector profile preparation; original inputs are read-only."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = ["event", "time_s", "coincident", "adc", "sipm_mv",
        "deadtime_s", "temperature_c", "pressure_pa"]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, config, input_dir=None):
    cfg = json.loads(config.read_text(encoding="utf-8"))
    def source(key):
        configured = Path(cfg[key])
        return input_dir / configured.name if input_dir else root / configured
    detector_path, logger_path = source("detector_file"), source("logger_file")
    d = pd.read_csv(detector_path, sep=r"\s+", comment="#", header=None)
    if d.shape[1] != 8:
        raise ValueError("Detector input must have exactly eight columns.")
    d.columns = COLS
    d = d.apply(pd.to_numeric, errors="raise")
    if len(d) < 2 or not np.isfinite(d.to_numpy()).all():
        raise ValueError("Detector records must be finite numeric values.")
    t = d.time_s.to_numpy()
    dead = d.deadtime_s.to_numpy()
    if (np.diff(t) <= 0).any() or not d.event.diff().iloc[1:].eq(1).all():
        raise ValueError("Review detector timestamp/event discontinuities first.")
    if not d.coincident.isin([0, 1]).all():
        raise ValueError("Unexpected coincidence flags.")
    if (dead < 0).any() or (np.diff(dead) < 0).any():
        raise ValueError("Review recorded deadtime before calculating candidates.")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "results" / cfg["flight_id"] / run
    clean = root / "clean" / cfg["flight_id"] / run
    out.mkdir(parents=True, exist_ok=False)
    clean.mkdir(parents=True, exist_ok=False)

    # Flag pressure measurements without removing their radiation events.
    local = pd.Series(d.pressure_pa.to_numpy(),
                      index=pd.to_timedelta(t, unit="s")).rolling(
                          f'{cfg["pressure_spike_window_seconds"]}s',
                          center=True, min_periods=1).median().to_numpy()
    threshold = np.maximum(cfg["pressure_spike_absolute_pa"],
                           cfg["pressure_spike_fraction"] * local)
    bad = ((d.pressure_pa.to_numpy() <= 0)
           | (np.abs(d.pressure_pa.to_numpy() - local) > threshold))
    d["pressure_flag"] = bad
    flagged = d.loc[bad].copy()
    flagged.insert(0, "source_record", flagged.index + 1)
    flagged.to_csv(out / "pressure_flags.csv", index=False)

    width = float(cfg["bin_seconds"])
    start, stop = np.ceil(t[0] / width) * width, np.floor(t[-1] / width) * width
    if stop <= start:
        raise ValueError("Recording has no complete interior bins.")
    edges = np.arange(start, stop + width / 2, width)
    left = np.searchsorted(t, edges[:-1], side="left")
    right = np.searchsorted(t, edges[1:], side="left")
    counts = right - left
    prefix = np.r_[0, np.cumsum(d.coincident.to_numpy())]
    coinc = prefix[right] - prefix[left]
    # This interpolation is only a candidate cumulative-deadtime calculation.
    # It does not establish acquisition continuity or synchronize clocks.
    dead_at_edges = np.interp(edges, t, dead)
    loss = np.diff(dead_at_edges)
    exposure = width - loss
    if (exposure <= 0).any() or (loss < 0).any():
        raise ValueError("Nonphysical candidate live-time exposure.")
    boundaries = np.asarray(
        cfg["phase_boundaries_detector_seconds_provisional"], dtype=float)
    labels = np.array(["prelaunch", "ascent", "descent", "postlanding"])
    rows = []
    for k, (a, b) in enumerate(zip(left, right)):
        block = d.iloc[a:b]
        pressure = block.loc[~block.pressure_flag, "pressure_pa"].median()
        mid = (edges[k] + edges[k + 1]) / 2
        rows.append({
            "flight_id": cfg["flight_id"], "detector_id": cfg["detector_id"],
            "bin_start_s": edges[k], "bin_end_s": edges[k + 1],
            "phase_provisional": labels[np.searchsorted(boundaries, mid)],
            "crosses_phase_boundary": bool(
                ((boundaries > edges[k]) & (boundaries < edges[k + 1])).any()),
            "counts_all": int(counts[k]), "counts_coincident": int(coinc[k]),
            "elapsed_s": width, "deadtime_s_candidate": loss[k],
            "live_time_s_candidate": exposure[k],
            "cps_elapsed_all": counts[k] / width,
            "cps_live_candidate_all": counts[k] / exposure[k],
            "cps_live_candidate_coincident": coinc[k] / exposure[k],
            "pressure_pa": pressure, "depth_g_cm2": pressure / 98.0665,
            "pressure_flags": int(block.pressure_flag.sum()),
            "temperature_c": block.temperature_c.median(),
            "training_ready": False,
        })
    bins = pd.DataFrame(rows)
    bins.to_csv(clean / "top_bins_10s.csv", index=False)
    before, after = int((t < start).sum()), int((t >= stop).sum())
    assert int(counts.sum()) + before + after == len(d)
    assert int(coinc.sum()) + int(d.loc[t < start, "coincident"].sum()) + int(
        d.loc[t >= stop, "coincident"].sum()) == int(d.coincident.sum())

    l = pd.read_csv(logger_path)
    lt = pd.to_numeric(l["Timer[S]"], errors="raise").to_numpy()
    if not np.isfinite(lt).all():
        raise ValueError("Nonfinite logger timer.")
    wraps = np.r_[False, np.diff(lt) < 0]
    period = cfg["logger_rollover_seconds_candidate"]
    for i in np.flatnonzero(wraps):
        if not (period - 2 <= lt[i - 1] < period and 0 <= lt[i] <= 2):
            raise ValueError("Logger reversal does not match the reviewed rollover pattern.")
        for column in ["Events CW1", "Events CW2"]:
            if l[column].iloc[i] < l[column].iloc[i - 1]:
                raise ValueError("Counter reversal at logger timer jump; review as a reset.")
    candidate = lt + np.cumsum(wraps) * cfg["logger_rollover_seconds_candidate"]
    if (np.diff(candidate) <= 0).any():
        raise ValueError("Candidate logger clock is not strictly increasing.")
    l["timer_s_candidate"] = candidate
    l["candidate_continuous_clock"] = np.all(np.diff(candidate) > 0)
    l["gps_basic_valid"] = (
        l.Sats.ge(4) & l.Latitude.between(-90, 90)
        & l.Longitude.between(-180, 180)
        & ~(l.Latitude.eq(0) & l.Longitude.eq(0))
        & l["Altitude[m]"].between(-500, 60000))
    l.to_csv(clean / "logger_time_review.csv", index=False)
    jumps = []
    for i in np.flatnonzero(wraps):
        jumps.append({"source_record": int(i + 1),
                      "before_timer_s": float(lt[i - 1]),
                      "after_timer_s": float(lt[i]),
                      "candidate_gap_s": float(candidate[i] - candidate[i - 1])})
    report = {
        "status": "provisional; no clock synchronization or model fit performed",
        "config": cfg, "source_sha256": {
            "detector": fingerprint(detector_path), "logger": fingerprint(logger_path)},
        "source_events": len(d), "binned_events": int(counts.sum()),
        "edge_events_before": before, "edge_events_after": after,
        "coincident_events": int(d.coincident.sum()),
        "pressure_flagged_event_rows": int(bad.sum()), "bins": len(bins),
        "maximum_detector_interevent_gap_s": float(np.diff(t).max()),
        "logger_rollover_candidates": jumps,
        "logger_candidate_clock_increasing": bool(np.all(np.diff(candidate) > 0)),
        "minimum_candidate_live_time_s": float(exposure.min()),
        "clean_directory": str(clean),
        "limitations": [
            "Ground/flight phase boundaries are provisional pressure-based choices.",
            "Pressure spike rule is diagnostic; sensor accuracy/range is unverified.",
            "Live time assumes cumulative seconds and interpolates bin boundaries.",
            "Quiet intervals do not prove continuous acquisition; pauses need records.",
            "Coincident counts are a subset of all counts, not independent samples.",
            "Logger candidate clock is not aligned to detector or UTC.",
            "All bins remain training_ready=False until scientific review."
        ]
    }
    (out / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    shown = d.iloc[::max(1, len(d) // 5000)]
    axes[0, 0].scatter(shown.time_s / 60, shown.pressure_pa, s=2)
    axes[0, 0].set(title="Top: own elapsed clock", xlabel="Minutes",
                   ylabel="Pressure [Pa]")
    shown_l = l.iloc[::max(1, len(l) // 5000)]
    axes[0, 1].scatter(shown_l.timer_s_candidate / 60,
                       shown_l["Pressure[Pa]"], s=2)
    axes[0, 1].set(title="Logger: candidate unwrapped clock (not aligned)",
                   xlabel="Minutes", ylabel="Pressure [Pa]")
    for phase in ["ascent", "descent"]:
        sub = bins[(bins.phase_provisional == phase)
                   & ~bins.crosses_phase_boundary & bins.pressure_pa.notna()]
        axes[1, 0].scatter(sub.depth_g_cm2, sub.cps_live_candidate_all,
                           s=4, label=phase)
        axes[1, 1].scatter(sub.depth_g_cm2, sub.cps_live_candidate_coincident,
                           s=4, label=phase)
    for ax, title in zip(axes[1], ["All recorded events", "Coincidence-flagged subset"]):
        ax.set(title=title, xlabel="Pressure-derived depth [g/cm²]",
               ylabel="Counts / candidate live second")
        ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("PRELIMINARY: phase, pressure and exposure assumptions unverified")
    fig.tight_layout()
    fig.savefig(out / "top_profile_review.png", dpi=150)
    plt.close(fig)
    print(f"Clean tables: {clean}")
    print(f"Review outputs: {out}")
    print(f"Events accounted for: {int(counts.sum())} + {before} + {after} = {len(d)}")
    return report, out, clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/2025-06-01_top.json")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path,
                        help="Optional folder containing the two original input files.")
    args = parser.parse_args()
    root = args.root.resolve()
    config = Path(args.config)
    prepare(root, config if config.is_absolute() else root / config, args.input_dir)
