"""Prepare a configured detector profile; original inputs are read-only."""
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


def read_detector(path):
    """Read a recording without sorting, dropping or altering source events."""
    d = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if d.shape[1] != 8:
        raise ValueError("Detector input must have exactly eight columns.")
    d.columns = COLS
    d = d.apply(pd.to_numeric, errors="raise")
    if len(d) < 2 or not np.isfinite(d.to_numpy()).all():
        raise ValueError("Detector records must be finite numeric values.")
    if (np.diff(d.time_s) <= 0).any() or not d.event.diff().iloc[1:].eq(1).all():
        raise ValueError("Review detector timestamp/event discontinuities first.")
    if not d.coincident.isin([0, 1]).all():
        raise ValueError("Unexpected coincidence flags.")
    if (d.deadtime_s < 0).any() or (np.diff(d.deadtime_s) < 0).any():
        raise ValueError("Review recorded deadtime before calculating candidates.")
    return d


def review_logger_clock(frame, period, max_gap=30.0, counter_columns=None):
    """Return a CANDIDATE unwrapped timer, not detector/UTC synchronization.

    A reversal is accepted only near the assumed timer boundaries, with a
    positive candidate gap no larger than max_gap and continuous counters.
    The 30 s gap limit is a diagnostic choice, not a firmware specification.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError("A finite positive candidate rollover period is required.")
    if not np.isfinite(max_gap) or not 0 < max_gap < period / 2:
        raise ValueError("Candidate rollover gap limit must be positive and < period/2.")
    l = frame.copy()
    if counter_columns is None:
        pairs = [("Events CW1&2", "Events CW1&2&3"), ("Events CW1", "Events CW2")]
        counter_columns = next((list(p) for p in pairs if set(p).issubset(l.columns)), None)
    if (not counter_columns or len(set(counter_columns)) != len(counter_columns)
            or not set(counter_columns).issubset(l.columns)):
        raise ValueError("Set logger_counter_columns to existing cumulative counter columns.")
    lt = pd.to_numeric(l["Timer[S]"], errors="raise").to_numpy(dtype=float)
    if len(lt) < 2 or not np.isfinite(lt).all() or (lt < 0).any() or (lt >= period).any():
        raise ValueError("Logger timer must be finite, nonnegative and below the candidate period.")
    for column in counter_columns:
        v = pd.to_numeric(l[column], errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(v).all() or (v < 0).any()
                or (v != np.floor(v)).any() or (np.diff(v) < 0).any()):
            raise ValueError(f"Counter {column!r} is not continuous cumulative integer data; review resets.")
    wraps = np.r_[False, np.diff(lt) < 0]
    jumps = []
    for i in np.flatnonzero(wraps):
        gap = period + lt[i] - lt[i - 1]
        if not (period - max_gap <= lt[i - 1] < period
                and 0 <= lt[i] <= max_gap and 0 < gap <= max_gap):
            raise ValueError("Logger reversal does not match the candidate rollover rule; review it.")
        jumps.append({"source_record": int(i + 1), "source_csv_row": int(i + 2),
                      "before_timer_s": float(lt[i - 1]), "after_timer_s": float(lt[i]),
                      "candidate_gap_s": float(gap)})
    candidate = lt + np.cumsum(wraps) * period
    if (np.diff(candidate) <= 0).any():
        raise ValueError("Candidate logger clock is not strictly increasing; review duplicate/reset records.")
    l["timer_s_candidate"] = candidate
    l["candidate_continuous_clock"] = True
    l["logger_clock_synchronized"] = False
    return l, jumps, list(counter_columns)


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
    d = read_detector(detector_path)
    t = d.time_s.to_numpy()
    dead = d.deadtime_s.to_numpy()
    if (np.diff(t) <= 0).any() or not d.event.diff().iloc[1:].eq(1).all():
        raise ValueError("Review detector timestamp/event discontinuities first.")
    if not d.coincident.isin([0, 1]).all():
        raise ValueError("Unexpected coincidence flags.")
    if (dead < 0).any() or (np.diff(dead) < 0).any():
        raise ValueError("Review recorded deadtime before calculating candidates.")

    width = float(cfg["bin_seconds"])
    if not np.isfinite(width) or width <= 0:
        raise ValueError("bin_seconds must be finite and positive.")
    phase_marks = cfg.get("phase_boundaries_detector_seconds_provisional")
    boundaries = np.array([], dtype=float) if phase_marks is None else np.asarray(phase_marks, dtype=float)
    phases_available = phase_marks is not None
    if phases_available and (boundaries.shape != (3,) or not np.isfinite(boundaries).all()
                             or (np.diff(boundaries) <= 0).any()):
        raise ValueError("Supply three increasing finite phase boundaries, or null for diagnostic-only preparation.")
    l, jumps, counter_columns = review_logger_clock(
        pd.read_csv(logger_path), float(cfg["logger_rollover_seconds_candidate"]),
        float(cfg.get("logger_rollover_max_gap_seconds_candidate", 30.0)),
        cfg.get("logger_counter_columns"))

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
    labels = np.array(["prelaunch", "ascent", "descent", "postlanding"])
    rows = []
    for k, (a, b) in enumerate(zip(left, right)):
        block = d.iloc[a:b]
        pressure = block.loc[~block.pressure_flag, "pressure_pa"].median()
        mid = (edges[k] + edges[k + 1]) / 2
        rows.append({
            "flight_id": cfg["flight_id"], "detector_id": cfg["detector_id"],
            "bin_start_s": edges[k], "bin_end_s": edges[k + 1],
            "phase_provisional": labels[np.searchsorted(boundaries, mid)] if phases_available else "unclassified",
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
    bins.to_csv(clean / f"{cfg['detector_id']}_bins_{width:g}s.csv", index=False)
    before, after = int((t < start).sum()), int((t >= stop).sum())
    assert int(counts.sum()) + before + after == len(d)
    assert int(coinc.sum()) + int(d.loc[t < start, "coincident"].sum()) + int(
        d.loc[t >= stop, "coincident"].sum()) == int(d.coincident.sum())

    l["gps_basic_valid"] = (
        l.Sats.ge(4) & l.Latitude.between(-90, 90)
        & l.Longitude.between(-180, 180)
        & ~(l.Latitude.eq(0) & l.Longitude.eq(0))
        & l["Altitude[m]"].between(-500, 60000))
    l.to_csv(clean / "logger_time_review.csv", index=False)
    report = {
        "status": "provisional; no clock synchronization or model fit performed",
        "phase_labels_available": phases_available,
        "preparation_script_sha256": fingerprint(Path(__file__)),
        "config": cfg, "source_sha256": {
            "detector": fingerprint(detector_path), "logger": fingerprint(logger_path)},
        "source_events": len(d), "binned_events": int(counts.sum()),
        "edge_events_before": before, "edge_events_after": after,
        "coincident_events": int(d.coincident.sum()),
        "coincidence_flag_status": "all_zero_interpretation_unverified" if not d.coincident.any()
                                   else "nonzero_flags_present_meaning_unverified",
        "pressure_flagged_event_rows": int(bad.sum()), "bins": len(bins),
        "maximum_detector_interevent_gap_s": float(np.diff(t).max()),
        "logger_rollover_candidates": jumps,
        "logger_candidate_clock_increasing": True,
        "logger_counter_columns": counter_columns,
        "logger_rollover_max_gap_seconds_candidate": float(cfg.get("logger_rollover_max_gap_seconds_candidate", 30.0)),
        "minimum_candidate_live_time_s": float(exposure.min()),
        "clean_directory": str(clean),
        "limitations": [
            "Ground/flight phase boundaries are provisional pressure-based choices.",
            "Absent phase boundaries leave bins unclassified; ascent/descent model fitting must wait.",
            "Pressure spike rule is diagnostic; sensor accuracy/range is unverified.",
            "Live time assumes cumulative seconds and interpolates bin boundaries.",
            "Quiet intervals do not prove continuous acquisition; pauses need records.",
            "Coincident counts are a subset of all counts, not independent samples.",
            "Logger candidate clock is not aligned to detector or UTC.",
            "Rollover period/gap rule are candidate assumptions, not firmware verification.",
            "Zero coincidence flags do not establish absence of physical coincidences.",
            "All bins remain training_ready=False until scientific review."
        ]
    }
    (out / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    shown = d.iloc[::max(1, len(d) // 5000)]
    axes[0, 0].scatter(shown.time_s / 60, shown.pressure_pa, s=2)
    axes[0, 0].set(title=f"{cfg['detector_id'].capitalize()}: own elapsed clock", xlabel="Minutes",
                   ylabel="Pressure [Pa]")
    shown_l = l.iloc[::max(1, len(l) // 5000)]
    axes[0, 1].scatter(shown_l.timer_s_candidate / 60,
                       shown_l["Pressure[Pa]"], s=2)
    axes[0, 1].set(title="Logger: candidate unwrapped clock (not aligned)",
                   xlabel="Minutes", ylabel="Pressure [Pa]")
    for phase in (["ascent", "descent"] if phases_available else ["unclassified"]):
        sub = bins[(bins.phase_provisional == phase)
                   & ~bins.crosses_phase_boundary & bins.pressure_pa.notna()]
        axes[1, 0].scatter(sub.depth_g_cm2, sub.cps_elapsed_all,
                           s=4, label=phase)
        axes[1, 1].scatter(sub.depth_g_cm2, sub.counts_coincident / sub.elapsed_s,
                           s=4, label=phase)
    for ax, title in zip(axes[1], ["All recorded events", "Coincidence-flagged subset"]):
        ax.set(title=title, xlabel="Pressure-derived depth [g/cm²]",
               ylabel="Counts / elapsed second")
        ax.legend()
    if not d.coincident.any():
        axes[1, 1].text(0.5, 0.65, "All flags are zero; physical coincidence status is unknown",
                        transform=axes[1, 1].transAxes, ha="center", wrap=True, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("PRELIMINARY: phase, pressure and exposure assumptions unverified" if phases_available
                 else "DIAGNOSTIC ONLY: flight phases unclassified; no clock synchronization")
    fig.tight_layout()
    fig.savefig(out / f"{cfg['detector_id']}_profile_review.png", dpi=150)
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
