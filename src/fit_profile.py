"""Fit exploratory Poisson GAMs to a table made by prepare_profiles.py.

Install dependencies: python -m pip install numpy pandas matplotlib pygam==0.12.0
Run from the project folder:
    python src/fit_profile.py --input clean/2025-06-01/RUN_ID/top_bins_10s.csv

One flight and detector per input; ascent and descent are fitted separately.
The target is integer counts, with exposure passed to PoissonGAM explicitly:
    expected counts = exposure_seconds * exp(smooth(log(depth_g_cm2))).
Natural-log depth spreads the spline knots over the pressure range.

Default exposure is elapsed time, assuming continuous acquisition. Use
--exposure candidate-live only as a comparison until deadtime is verified.
Raw files, the input CSV, and its training_ready flags are never modified.
False training_ready flags are accepted because this script is exploratory.

Smoothing is selected by UBRE on the fitting data, not held-out validation.
Intervals describe the fitted mean under the Poisson model, with smoothing
treated as fixed. They omit serial dependence, calibration/systematic error,
and smoothing-selection uncertainty. They are NOT peak-location intervals.
An interior curve maximum is a candidate, not a verified Pfotzer maximum.

API reference:
https://pygam.readthedocs.io/en/latest/reference/_autosummary/pygam.pygam.PoissonGAM.html
"""

import argparse
import hashlib
import json
import pickle
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pygam import PoissonGAM, s
except ImportError as exc:
    raise SystemExit("Install pyGAM: python -m pip install pygam==0.12.0") from exc


PHASES = ("ascent", "descent")
PA_PER_G_CM2 = 98.0665
LAMBDA_GRID = np.logspace(-2, 4, 9)
EXPOSURES = {"elapsed": "elapsed_s", "candidate-live": "live_time_s_candidate"}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean_column(values, name):
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{name} must contain True/False or 1/0, with no blanks.")
    return parsed.astype(bool)


def load_bins(path, channel, exposure, include_flagged=False):
    """Validate the table and give each excluded row an explicit reason."""
    count_column = f"counts_{channel}"
    exposure_column = EXPOSURES[exposure]
    df = pd.read_csv(path, dtype={"flight_id": str, "detector_id": str})
    required = {
        "flight_id", "detector_id", "bin_start_s", "bin_end_s", "elapsed_s",
        "phase_provisional", "crosses_phase_boundary", "pressure_pa",
        "depth_g_cm2", "pressure_flags", "training_ready", count_column,
        exposure_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Input table is empty.")
    df.insert(0, "source_csv_row", np.arange(len(df)) + 2)
    for name in ("flight_id", "detector_id"):
        if df[name].isna().any() or df[name].nunique() != 1:
            raise ValueError(f"Provide exactly one nonmissing {name} per file.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", df[name].iloc[0]):
            raise ValueError(f"{name} must be a simple identifier, without spaces/slashes.")
    for name in ("crosses_phase_boundary", "training_ready"):
        df[name] = boolean_column(df[name], name)
    if df.phase_provisional.isna().any():
        raise ValueError("Every bin needs a phase_provisional label.")

    numeric = {
        "bin_start_s", "bin_end_s", "elapsed_s", "pressure_pa", "depth_g_cm2",
        "pressure_flags", count_column, exposure_column,
    }
    for name in numeric:
        df[name] = pd.to_numeric(df[name], errors="raise")
    for name in numeric - {"pressure_pa", "depth_g_cm2"}:
        if not np.isfinite(df[name]).all():
            raise ValueError(f"{name} contains missing or nonfinite values.")
    for name in (count_column, "pressure_flags"):
        if (df[name] < 0).any() or not np.equal(df[name], np.floor(df[name])).all():
            raise ValueError(f"{name} must contain nonnegative integers.")

    df = df.sort_values("bin_start_s").reset_index(drop=True)
    duration = df.bin_end_s - df.bin_start_s
    if (duration <= 0).any() or not np.allclose(
        duration, df.elapsed_s, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("elapsed_s must equal the positive bin duration.")
    if (df.bin_start_s.to_numpy()[1:] < df.bin_end_s.to_numpy()[:-1] - 1e-6).any():
        raise ValueError("Bins overlap or are duplicated; review input segmentation.")
    if (df[exposure_column] <= 0).any() or (
        df[exposure_column] > df.elapsed_s + 1e-6
    ).any():
        raise ValueError("Exposure must be positive and cannot exceed elapsed time.")

    valid_pressure = (
        np.isfinite(df.pressure_pa) & df.pressure_pa.gt(0)
        & np.isfinite(df.depth_g_cm2) & df.depth_g_cm2.gt(0)
    )
    if not np.allclose(
        df.loc[valid_pressure, "depth_g_cm2"],
        df.loc[valid_pressure, "pressure_pa"] / PA_PER_G_CM2,
        rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("Pressure/depth values disagree; check units and preparation.")
    # Reasons have precedence so their counts reconcile exactly to input rows.
    reasons = np.select(
        [~df.phase_provisional.isin(PHASES), df.crosses_phase_boundary,
         ~valid_pressure, df.pressure_flags.gt(0) & (not include_flagged)],
        ["outside_ascent_descent", "crosses_phase_boundary", "invalid_pressure_or_depth",
         "flagged_pressure_in_bin"], default="included",
    )
    df["fit_selection"] = reasons
    df["counts_used"] = df[count_column].astype(np.int64)
    df["exposure_s_used"] = df[exposure_column]
    return df


def fit_phase(data, n_splines, fixed_lambda=None):
    """Fit one phase; return model, per-bin predictions, curve and diagnostics."""
    if len(data) < max(30, n_splines + 5) or data.depth_g_cm2.nunique() < n_splines:
        raise ValueError("Too few bins or distinct depths for the requested spline count.")
    depth = data.depth_g_cm2.to_numpy(dtype=float)
    x = np.log(depth)[:, None]
    y = data.counts_used.to_numpy(dtype=float)
    exposure = data.exposure_s_used.to_numpy(dtype=float)
    if y.sum() == 0:
        raise ValueError("Selected channel has no events; cannot estimate its profile.")
    # Explicit fits keep exposure a keyword argument for every candidate.
    # Do not replace this loop with PoissonGAM.gridsearch: in pyGAM 0.12.0,
    # its inherited positional fit call mishandles non-unit exposure.
    penalties = LAMBDA_GRID if fixed_lambda is None else [fixed_lambda]
    gam, best_score, smoothing_scores = None, np.inf, []
    for penalty in penalties:
        candidate = PoissonGAM(s(0, n_splines=n_splines), lam=float(penalty),
                               max_iter=500, tol=1e-6)
        candidate.fit(x, y, exposure=exposure)
        if not candidate.logs_.get("diffs") or candidate.logs_["diffs"][-1] >= candidate.tol:
            raise ValueError(f"GAM did not converge at lambda={penalty}; review settings.")
        score = float(candidate.statistics_["UBRE"])
        if not np.isfinite(score):
            raise ValueError("Smoothing score is nonfinite; review the input data.")
        smoothing_scores.append({"lambda": float(penalty), "UBRE": score})
        if score < best_score:
            gam, best_score = candidate, score

    # With exposure omitted, predictions are rates per one second.
    rate = gam.predict(x)
    expected = gam.predict(x, exposure=exposure)
    grid_depth = np.geomspace(depth.min(), depth.max(), 1000)
    grid_x = np.log(grid_depth)[:, None]
    grid_rate = gam.predict(grid_x)
    intervals = gam.confidence_intervals(grid_x, width=0.95)
    if not all(np.isfinite(a).all() for a in (rate, expected, grid_rate, intervals)):
        raise ValueError("Nonfinite model output; no successful fit will be reported.")
    if (expected <= 0).any() or (grid_rate <= 0).any():
        raise ValueError("Nonpositive predictions; numerical fit failure.")
    # An unpenalized intercept should reconcile fitted and observed total counts.
    if not np.isclose(expected.sum(), y.sum(), rtol=1e-4, atol=1e-3):
        raise ValueError("Fitted count total does not reconcile; check exposure or convergence.")

    terms = np.zeros_like(y)
    positive = y > 0
    terms[positive] = y[positive] * np.log(y[positive] / expected[positive])
    deviance = 2 * np.maximum(terms - y + expected, 0)
    pearson = (y - expected) / np.sqrt(expected)
    fitted = data.copy()
    fitted["observed_rate_cps"] = y / exposure
    fitted["fitted_rate_cps"] = rate
    fitted["fitted_counts"] = expected
    fitted["pearson_residual"] = pearson
    fitted["deviance_residual"] = np.sign(y - expected) * np.sqrt(deviance)
    curve = pd.DataFrame({
        "phase_provisional": data.phase_provisional.iloc[0],
        "depth_g_cm2": grid_depth, "pressure_pa": grid_depth * PA_PER_G_CM2,
        "fitted_rate_cps": grid_rate,
        "mean_rate_lower95_conditional": intervals[:, 0],
        "mean_rate_upper95_conditional": intervals[:, 1],
    })

    peak_index = int(np.argmax(grid_rate))
    peak_depth = float(grid_depth[peak_index])
    left = int(np.sum(depth < peak_depth))
    right = int(np.sum(depth > peak_depth))
    boundary = peak_index in (0, len(grid_depth) - 1)
    peak_status = "boundary_maximum" if boundary else "interior_fitted_maximum"
    if not boundary and min(left, right) < 5:
        peak_status = "insufficient_measurements_on_both_sides"
    edof = float(gam.statistics_["edof"])
    stats = {
        "bins": len(data), "events": int(y.sum()),
        "exposure_seconds": float(exposure.sum()),
        "depth_range_g_cm2": [float(depth.min()), float(depth.max())],
        "selected_lambda": float(np.asarray(gam.lam).ravel()[0]),
        "smoothing_scores": smoothing_scores,
        "effective_degrees_of_freedom": edof,
        "fitted_total_counts": float(expected.sum()),
        "in_sample_mean_poisson_deviance": float(deviance.mean()),
        "in_sample_rate_mae_cps": float(np.mean(np.abs(y / exposure - rate))),
        "pearson_chi2_per_residual_df": float(np.sum(pearson**2) / (len(y) - edof)),
        "fitted_maximum": {
            "status": peak_status, "depth_g_cm2": peak_depth,
            "pressure_pa": peak_depth * PA_PER_G_CM2,
            "rate_cps": float(grid_rate[peak_index]),
            "bins_below_depth": left, "bins_above_depth": right,
            "peak_location_interval": None,
        },
    }
    return gam, fitted, curve, stats


def save_plot(fitted, curves, report, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, phase in enumerate(PHASES):
        data = fitted[fitted.phase_provisional.eq(phase)]
        curve = curves[curves.phase_provisional.eq(phase)]
        ax = axes[0, column]
        ax.scatter(data.depth_g_cm2, data.observed_rate_cps, s=10, alpha=0.5,
                   color="#52697e", label="Observed bins")
        ax.fill_between(curve.depth_g_cm2, curve.mean_rate_lower95_conditional,
                        curve.mean_rate_upper95_conditional, color="#007f86",
                        alpha=0.18, label="95% conditional mean interval")
        ax.plot(curve.depth_g_cm2, curve.fitted_rate_cps, color="#007f86",
                linewidth=2, label="Poisson GAM")
        peak = report["phases"][phase]["fitted_maximum"]
        if peak["status"] == "interior_fitted_maximum":
            ax.axvline(peak["depth_g_cm2"], color="#bd7525", linestyle="--",
                       label="Candidate fitted maximum")
        ax.set(xscale="log", xlabel="Atmospheric depth [g/cm²; log scale]",
               ylabel="Recorded events / exposure second",
               title=f"{phase.capitalize()}: {len(data)} bins")
        ax.legend(fontsize=8)
        axes[1, column].scatter(
            (data.bin_start_s + data.bin_end_s) / 120, data.pearson_residual,
            s=10, color="#52697e", alpha=0.7,
        )
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set(xlabel="Detector elapsed time [minutes]",
                            ylabel="Pearson residual", title="Residuals on fitting data")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"EXPLORATORY | {report['flight_id']} | {report['detector_id']} | "
        f"{report['channel']} events | {report['exposure']} exposure"
    )
    fig.text(0.5, 0.015,
             "Provisional phases and exposure. Bands omit calibration/systematic error "
             "and are not peak-location intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Prepared bins CSV.")
    parser.add_argument("--channel", choices=("all", "coincident"), default="all")
    parser.add_argument("--exposure", choices=EXPOSURES, default="elapsed")
    parser.add_argument("--n-splines", type=int, default=12)
    parser.add_argument("--lam", type=float, default=None,
                        help="Fixed smoothing penalty; otherwise select from a UBRE grid.")
    parser.add_argument("--include-flagged-pressure-bins", action="store_true",
                        help="Include bins with pressure flags for a sensitivity check.")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args(argv)
    if args.n_splines < 4:
        parser.error("--n-splines must be at least 4 for cubic splines.")
    if args.lam is not None and (not np.isfinite(args.lam) or args.lam <= 0):
        parser.error("--lam must be a finite positive number.")

    source = args.input.resolve()
    df = load_bins(source, args.channel, args.exposure, args.include_flagged_pressure_bins)
    selected = df[df.fit_selection.eq("included")]
    models, fits, curves, phase_stats = {}, [], [], {}
    for phase in PHASES:
        data = selected[selected.phase_provisional.eq(phase)].copy()
        print(f"Fitting {phase}: {len(data)} bins", flush=True)
        model, fitted, curve, stats = fit_phase(data, args.n_splines, args.lam)
        models[phase] = model
        fits.append(fitted)
        curves.append(curve)
        phase_stats[phase] = stats
    fitted, curve = pd.concat(fits, ignore_index=True), pd.concat(curves, ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_root.resolve() / df.flight_id.iloc[0] / f"{stamp}_gam"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "exploratory fit; not scientifically validated",
        "flight_id": df.flight_id.iloc[0], "detector_id": df.detector_id.iloc[0],
        "channel": args.channel, "exposure": args.exposure,
        "model": "Poisson GAM: log(expected counts) = log(exposure) + spline(log(depth))",
        "n_splines": args.n_splines, "lambda_grid": LAMBDA_GRID.tolist()
        if args.lam is None else [args.lam],
        "smoothing_selection": "in-sample UBRE" if args.lam is None else "user-specified",
        "include_flagged_pressure_bins": args.include_flagged_pressure_bins,
        "input_csv": str(source), "input_sha256": fingerprint(source),
        "script_sha256": fingerprint(__file__),
        "input_rows": len(df), "input_training_ready_false": int((~df.training_ready).sum()),
        "selection_counts": {str(k): int(v) for k, v in df.fit_selection.value_counts().items()},
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("pygam", "numpy", "pandas", "scipy", "matplotlib")}},
        "phases": phase_stats,
        "limitations": [
            "Exposure assumes continuous acquisition; event gaps cannot verify it.",
            "Candidate live time additionally assumes cumulative deadtime in seconds.",
            "Flight phases, pressure calibration and detector response remain provisional.",
            "No logger/APRS synchronization, altitude conversion or amplitude calibration.",
            "Metrics are in-sample diagnostics, not accuracy on independent flights.",
            "95% pointwise mean intervals condition on smoothing and Poisson independence.",
            "Intervals exclude calibration/systematic error and peak-location uncertainty.",
            "Large Pearson chi2 per residual df suggests extra variation or mean-model error.",
            "Curve maxima are restricted to observed depth ranges; boundary maxima are unsupported peaks.",
        ],
    }
    fitted.to_csv(out / "fitted_bins.csv", index=False)
    curve.to_csv(out / "profile_curves.csv", index=False)
    df.loc[~df.fit_selection.eq("included")].to_csv(out / "excluded_bins.csv", index=False)
    with (out / "gam_models.pkl").open("wb") as handle:
        pickle.dump({"models": models, "metadata": report}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    save_plot(fitted, curve, report, out / "gam_profile.png")
    (out / "fit_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                          encoding="utf-8")
    print(f"Results: {out}")
    print("Saved models, plot, predictions, excluded bins and fit_summary.json.")
    print("Exploratory only. Input data and training_ready flags were not changed.")
    return out


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Cannot fit profile: {exc}") from exc
