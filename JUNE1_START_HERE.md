June 1: first-pass top-detector profile
=====================================

Copy src/prepare_profiles.py into your project's src folder and
configs/2025-06-01_top.json into configs. Keep your current README and requirements.
The ZIP also includes the outputs produced from your uploaded recordings under
example_outputs/. Those are examples; running the script creates new local outputs.

Run from the project folder in PowerShell:

    .\.venv\Scripts\python.exe src\prepare_profiles.py

Dependencies: numpy, pandas, matplotlib (already used by inspect_flight.py).

Each run creates a unique directory under clean/2025-06-01 and results/2025-06-01.
The terminal prints both exact locations.

Outputs:

- clean/.../top_bins_10s.csv: counts, candidate exposure, pressure/depth,
  temperature, and provisional phase labels.
- clean/.../logger_time_review.csv: original logger columns plus its candidate
  elapsed clock and basic GPS flag. No UTC or detector-clock join is established.
- results/.../preparation_report.json: source hashes, configuration, timer jumps,
  count accounting, and limitations.
- results/.../pressure_flags.csv: flagged pressure rows; source_record counts
  data records, excluding comment/header lines.
- results/.../top_profile_review.png: clocks and measured profiles.

What this first pass does:

1. Requires the observed eight-column detector schema, continuous sequential event
   IDs, finite measurements, increasing time, and nondecreasing recorded deadtime.
2. Retains every event for counting. A centered 10-second pressure median flags
   differences greater than max(1000 Pa, 5% of the local median), plus nonpositive
   pressure. This diagnostic rule needs review; it is not a calibrated sensor range.
3. Uses only full interior 10-second intervals with [start, end) event membership.
   Partial recording-edge events are counted in the report instead of silently lost.
   Zero-count bins are retained, with missing pressure if no event supplies it.
4. Reports all-event counts and their coincidence-flagged subset as separate
   columns. They overlap; do not concatenate them as independent observations.
5. Calculates candidate live time by interpolating recorded cumulative deadtime
   at interval boundaries. It assumes cumulative seconds; firmware semantics and
   acquisition continuity remain unverified. No amplitude calibration is applied.
6. Tests the specific logger rollover hypothesis of 4294.967296 seconds, requiring
   each reversal to go from within two seconds of that period to within two seconds
   of zero, without a reversal in either CW counter at that boundary. Unexpected
   cases stop processing. This pattern check does not prove the firmware semantics.

The configuration uses provisional detector-clock boundaries:

- 3210 seconds (53.5 min): prelaunch/ascent.
- 9300 seconds (155 min): ascent/descent.
- 11190 seconds (186.5 min): descent/postlanding.

These choices come from the pressure curve. They are not GPS-confirmed launch,
burst or landing timestamps. Crossing intervals are marked. Review with launch
records/tracking data and adjust the configuration if necessary.

The detector pressure is kept on the detector's own elapsed clock. The logger
clock is reviewed separately; the code never treats matching timer values or
similar pressure curves as proof of synchronized devices.

All bins have training_ready=False. This is a preliminary measured profile,
not a scientifically validated training table or a fitted model.
Next: review the phase choices, pressure outliers/sensor validity, detector
settings and exposure interpretation. Then fit the planned Poisson GAM separately
by detector, phase and chosen channel, using integer counts and exposure.

Keep raw/, clean/ and results/ excluded from Git. Commit the source and config.
This script was run on the uploaded June 1 top/logger recordings. The example
outputs should not be assumed valid for other detectors, flights or firmware.
