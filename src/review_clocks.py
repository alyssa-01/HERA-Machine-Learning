"""Read-only clock diagnostics for flight or simultaneous ground recordings.

No clock mapping is fitted, no events are paired, and no source is rewritten.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare_profiles import read_detector, review_logger_clock, fingerprint


def binned_rate(times, weights, width):
    start = np.ceil(times[0] / width) * width
    stop = np.floor(times[-1] / width) * width
    if stop <= start:
        raise ValueError("Recording too short for complete bins.")
    edges = np.arange(start, stop + width / 2, width)
    keep = (times >= start) & (times < stop)
    counts = np.histogram(times[keep], bins=edges, weights=weights[keep])[0]
    return pd.DataFrame({"bin_start_s": edges[:-1], "bin_end_s": edges[1:],
                         "counts": counts, "elapsed_s": width, "rate_cps": counts / width})


def review(root, detector_files, logger_file, run_id, bin_seconds=10.0,
           rollover_seconds_candidate=4294.967296,
           rollover_max_gap_seconds_candidate=30.0, logger_counter_columns=None):
    root = Path(root).resolve()
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in run_id):
        raise ValueError("run_id must be a simple identifier.")
    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive and finite.")
    if not detector_files or "logger" in detector_files:
        raise ValueError("Provide at least one detector, using identities other than 'logger'.")
    files = {str(k): Path(v).resolve() for k, v in detector_files.items()}
    files["logger"] = Path(logger_file).resolve()
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing original input: {path}")
    hashes = {name: fingerprint(path) for name, path in files.items()}
    frames = {name: read_detector(path) for name, path in files.items() if name != "logger"}
    logger_raw = pd.read_csv(files["logger"])
    logger, jumps, channels = review_logger_clock(
        logger_raw, rollover_seconds_candidate, rollover_max_gap_seconds_candidate,
        logger_counter_columns)
    summary = []
    rates = []
    for name, frame in frames.items():
        times = frame.time_s.to_numpy(dtype=float)
        summary.append({"device": name, "rows": len(frame), "first_event": int(frame.event.iloc[0]),
            "last_event": int(frame.event.iloc[-1]), "first_timer_s": float(times[0]),
            "last_timer_s": float(times[-1]), "recorded_span_hours": float((times[-1]-times[0])/3600),
            "backward_timer_steps": int((np.diff(times) < 0).sum()),
            "maximum_interevent_gap_s": float(np.diff(times).max()),
            "flagged_coincidence_events": int(frame.coincident.sum()),
            "coincidence_flag_status": "all_zero_interpretation_unverified" if not frame.coincident.any()
                else "nonzero_flags_present_meaning_unverified",
            "initial_pressure_median_pa": float(frame.loc[frame.time_s < times[0]+300, "pressure_pa"].median())})
        rate = binned_rate(times, np.ones(len(frame)), bin_seconds)
        rate["device"] = name
        rate["channel"] = "all_recorded_events"
        rates.append(rate)
    times = logger.timer_s_candidate.to_numpy(dtype=float)
    summary.append({"device": "logger", "rows": len(logger),
        "first_timer_s": float(times[0]), "last_timer_s": float(times[-1]),
        "recorded_span_hours": float((times[-1]-times[0])/3600),
        "backward_timer_steps": int((np.diff(logger_raw["Timer[S]"]) < 0).sum())})
    for channel in channels:
        # The first cumulative counter is a snapshot without a known event time.
        delta = np.diff(logger[channel].to_numpy(dtype=float))
        rate = binned_rate(times[1:], delta, bin_seconds)
        rate["device"] = "logger"
        rate["channel"] = channel
        rates.append(rate)
    rtc_summary = {"available": False}
    rtc_issues = pd.DataFrame(columns=["source_csv_row", "issue"])
    if {"Date[MM/DD/YYYY]", "RTC[H:M:S EST]"}.issubset(logger.columns):
        rtc = pd.to_datetime(logger["Date[MM/DD/YYYY]"].astype(str) + " "
                             + logger["RTC[H:M:S EST]"].astype(str),
                             format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
        delta = rtc.diff().dt.total_seconds()
        midnight = delta < -12*3600
        backwards = delta < 0
        invalid = rtc.isna()
        bad_indices = np.flatnonzero((backwards | invalid).to_numpy())
        issue_rows = []
        for i in bad_indices:
            issue_rows.append({"source_csv_row": int(i+2),
                "issue": "unparsed_datetime" if invalid.iloc[i] else
                         "large_backward_jump_possible_stale_date" if midnight.iloc[i] else "backward_datetime_step",
                "date_recorded": str(logger["Date[MM/DD/YYYY]"].iloc[i]),
                "rtc_recorded": str(logger["RTC[H:M:S EST]"].iloc[i]),
                "step_seconds": None if invalid.iloc[i] else float(delta.iloc[i])})
        rtc_issues = pd.DataFrame(issue_rows, columns=["source_csv_row", "issue", "date_recorded",
                                                      "rtc_recorded", "step_seconds"])
        rtc_summary = {"available": True, "unparsed_datetimes": int(invalid.sum()),
            "backward_datetime_steps": int(backwards.sum()),
            "large_backward_jumps_possible_stale_date": int(midnight.sum()),
            "distinct_recorded_dates": int(logger["Date[MM/DD/YYYY]"].nunique()),
            "timezone_verified": False}
    summary = pd.DataFrame(summary)
    binned_rates = pd.concat(rates, ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "results" / run_id / f"{stamp}_clock_review"
    out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(out / "clock_summary.csv", index=False)
    binned_rates.to_csv(out / "native_clock_rates.csv", index=False)
    pd.DataFrame(jumps, columns=["source_record", "source_csv_row", "before_timer_s",
                                "after_timer_s", "candidate_gap_s"]).to_csv(out / "logger_rollovers.csv", index=False)
    rtc_issues.to_csv(out / "logger_datetime_issues.csv", index=False)
    logger.to_csv(out / "logger_time_review.csv", index=False)
    input_inventory = [{"device": name, "path": str(path), "sha256": hashes[name],
                        "size_bytes": path.stat().st_size} for name, path in files.items()]
    pd.DataFrame(input_inventory).to_csv(out / "input_inventory.csv", index=False)

    fig, axes = plt.subplots(len(files), 3, figsize=(14, 2.7*len(files)), squeeze=False)
    for row, name in enumerate(files):
        if name == "logger":
            frame, time_col, pressure_col, temperature_col = logger, "timer_s_candidate", "Pressure[Pa]", "Temperature[C]"
        else:
            frame, time_col, pressure_col, temperature_col = frames[name], "time_s", "pressure_pa", "temperature_c"
        shown = frame.iloc[::max(1, len(frame)//5000)]
        for col, field in [(0, pressure_col), (2, temperature_col)]:
            if field in frame:
                axes[row,col].scatter(shown[time_col]/3600, shown[field], s=2)
            else:
                axes[row,col].text(.5,.5,"Not recorded",ha="center",transform=axes[row,col].transAxes)
        for channel, part in binned_rates.loc[binned_rates.device.eq(name)].groupby("channel",sort=False):
            axes[row,1].plot((part.bin_start_s+part.bin_end_s)/7200,part.rate_cps,lw=.7,label=channel)
        axes[row,1].legend(fontsize=7)
        for col, label in enumerate(["Pressure [Pa]", "Recorded events / elapsed second", "Temperature [C]"]):
            axes[row,col].set(xlabel="Own clock [hours]; not synchronized",ylabel=label,title=f"{name}: {label}")
            axes[row,col].grid(alpha=.2)
    fig.suptitle("Recording overview: native detector clocks; logger timer unwrapping is provisional")
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out/"recording_overview.png",dpi=130)
    plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11,6))
    axes[0].plot(np.arange(len(logger_raw))+2,logger_raw["Timer[S]"],lw=.7)
    axes[0].set(xlabel="Original CSV row",ylabel="Raw logger timer [s]",title="Raw logger timer: backward jumps preserved")
    axes[1].plot(np.arange(len(logger))+2,logger.timer_s_candidate/3600,lw=.7)
    axes[1].set(xlabel="Original CSV row",ylabel="Candidate timer [hours]",title="Candidate unwrapped timer: NOT detector or UTC synchronization")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"logger_timer_review.png",dpi=140)
    plt.close(fig)
    if any(fingerprint(path)!=hashes[name] for name,path in files.items()):
        raise RuntimeError("An input changed during review; outputs must not be treated as complete.")
    report={"status":"complete_diagnostic_review_no_synchronization_performed","run_id":run_id,
        "input_files":input_inventory,"bin_seconds":bin_seconds,
        "logger_counter_columns":channels,"rollover_seconds_candidate":rollover_seconds_candidate,
        "rollover_max_gap_seconds_candidate":rollover_max_gap_seconds_candidate,
        "logger_rollover_candidates":jumps,"datetime_review":rtc_summary,
        "clock_corrections_applied":False,"utc_verified":False,"source_hashes_unchanged":True,
        "limitations":[
            "No clock drift model, event matching or calibration is performed.",
            "Timer period and rollover gap limit are candidate assumptions, not verified firmware properties.",
            "A continuous candidate timer does not prove the logger kept accurate time.",
            "Constant pressure/rate on the ground is not sufficient to identify clock drift.",
            "Rates assume continuous acquisition inside complete native-clock bins; logging pauses are unverified.",
            "Logger counter increments are placed at the later row; increments >1 lack individual event timestamps.",
            "Zero coincidence flags are uninformative until firmware/wiring behavior is verified.",
            "Raw dates/timezones are not repaired or interpreted as UTC.",
            "Stable event numbers do not prove absence of unrecorded physical hits."]}
    (out/"clock_review_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report, summary, out
