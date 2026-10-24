from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "2025-06-01" / "RawData"
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "results" / "2025-06-01" / run_id
OUT.mkdir(parents=True, exist_ok=False)

FILES = {
    "top": ("Bravo CW Top.txt", "detector"),
    "bottom": ("Bravo CW Bottom.txt", "detector"),
    "perpendicular": ("Bravo CW Perpendicular.txt", "detector"),
    "logger_coincident": ("Bravo CW Datalogger Coincident.csv", "logger"),
    "logger_perpendicular": ("Bravo CW Datalogger Perpendicular.csv", "logger"),
}

DETECTOR_COLUMNS = [
    "event", "time_s", "coincident", "adc",
    "sipm_mv", "deadtime_s", "temperature_c", "pressure_pa",
]

summaries = []
errors = {}
fig, axes = plt.subplots(len(FILES), 1, figsize=(12, 15))

for ax, (name, (filename, kind)) in zip(axes, FILES.items()):
    try:
        path = RAW / filename

        if kind == "detector":
            df = pd.read_csv(
                path, sep=r"\s+", comment="#", header=None
            )
            if df.shape[1] != len(DETECTOR_COLUMNS):
                raise ValueError(
                    f"Expected 8 columns; found {df.shape[1]}"
                )
            df.columns = DETECTOR_COLUMNS
            df = df.apply(pd.to_numeric, errors="raise")
            time_col, pressure_col = "time_s", "pressure_pa"
        else:
            df = pd.read_csv(path)
            numeric_columns = [
                "Timer[S]", "Pressure[Pa]", "Temperature[C]",
                "Latitude", "Longitude", "Altitude[m]", "Sats",
            ]
            for column in numeric_columns:
                df[column] = pd.to_numeric(df[column], errors="raise")
            time_col, pressure_col = "Timer[S]", "Pressure[Pa]"

        if df.empty:
            raise ValueError("No data rows")

        t = df[time_col]
        p = df[pressure_col]
        valid = np.isfinite(t) & np.isfinite(p) & (p > 0)

        summary = {
            "recording": name,
            "source_file": filename,
            "rows": len(df),
            "first_timer_s": t.iloc[0],
            "last_timer_s": t.iloc[-1],
            "timer_backward_steps": int((t.diff() < 0).sum()),
            "timer_repeats": int((t.diff() == 0).sum()),
            "invalid_time_or_pressure_rows": int((~valid).sum()),
            "minimum_positive_pressure_pa": p[
                np.isfinite(p) & (p > 0)
            ].min(),
        }

        if kind == "detector":
            summary.update({
                "coincidence_flagged_events":
                    int(df["coincident"].eq(1).sum()),
                "invalid_coincidence_flags":
                    int((~df["coincident"].isin([0, 1])).sum()),
                "event_id_nonunit_steps":
                    int(df["event"].diff().iloc[1:].ne(1).sum()),
                "deadtime_backward_steps":
                    int((df["deadtime_s"].diff() < 0).sum()),
                "last_recorded_deadtime_s":
                    df["deadtime_s"].iloc[-1],
            })
        else:
            gps_ok = (
                df["Sats"].ge(4)
                & df["Latitude"].between(-90, 90)
                & df["Longitude"].between(-180, 180)
                & ~(df["Latitude"].eq(0) & df["Longitude"].eq(0))
                & df["Altitude[m]"].between(-500, 60000)
            )
            summary["gps_rows_failing_basic_screen"] = int(
                (~gps_ok).sum()
            )
            dates = pd.to_datetime(
                df["Date[MM/DD/YYYY]"],
                format="%m/%d/%Y",
                errors="coerce",
            )
            summary["unparseable_date_rows"] = int(dates.isna().sum())
            summary["dates_before_2020"] = int(
                dates.dt.year.lt(2020).sum()
            )

        summaries.append(summary)

        # Sample for display only; checks above use every row.
        shown = df.iloc[::max(1, len(df) // 5000)]
        ax.scatter(
            shown[time_col] / 60,
            shown[pressure_col],
            s=2,
        )
        ax.set_title(name)
        ax.set_ylabel("Pressure [Pa]")
        ax.set_xlabel("Raw device timer [minutes]")
        ax.grid(alpha=0.3)

    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
        ax.set_title(f"{name}: inspection failed")

pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
(OUT / "file_errors.json").write_text(
    json.dumps(errors, indent=2), encoding="utf-8"
)
fig.tight_layout()
fig.savefig(OUT / "pressure_overview.png", dpi=150)
plt.close(fig)

print(f"Reports saved in: {OUT}")
print(pd.DataFrame(summaries).to_string(index=False))
print(f"File errors: {len(errors)}")