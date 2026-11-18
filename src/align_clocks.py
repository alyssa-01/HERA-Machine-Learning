"""Exploratory event-timing alignment. Raw files and coincidence flags are never edited.

The fitted map is relative to one detector, not UTC. Local timing excesses are
statistical signals, not verified particle identities or calibration labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.stats import theilslopes, poisson


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_times(path):
    a = pd.read_csv(path, sep=r'\s+', comment='#', header=None,
                    dtype='float64').to_numpy()
    if a.ndim != 2 or a.shape[1] != 8 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError(f'Expected at least two finite eight-column events: {path}')
    if np.any(np.diff(a[:, 1]) <= 0):
        raise ValueError(f'Clock reverses or repeats; inspect segments before alignment: {path}')
    if np.any(np.diff(a[:, 0]) != 1) or np.any(a[:, 0] != np.floor(a[:, 0])):
        raise ValueError(f'Event IDs are not contiguous integers: {path}')
    if not np.isin(a[:, 2], [0, 1]).all():
        raise ValueError(f'Unexpected coincidence flag values: {path}')
    meta = {'rows': len(a), 'first_time_s': float(a[0, 1]),
            'last_time_s': float(a[-1, 1]),
            'native_span_hours': float((a[-1, 1] - a[0, 1]) / 3600),
            'coincidence_flagged_rows': int(a[:, 2].sum()),
            'filename': Path(path).name}
    return np.ascontiguousarray(a[:, 1]), meta


def _pair_count(reference, other, lag, half_width=.005):
    left = np.searchsorted(other, reference + lag - half_width, side='left')
    right = np.searchsorted(other, reference + lag + half_width, side='right')
    return int(np.sum(right - left))


def local_lag(reference, other, center_s, width_s=600., search_s=120., bin_s=.01):
    """Estimate other_time - reference_time in one fixed reference window.

    Linear 'valid' correlation uses a fixed complete reference window at every
    lag. No circular shifts or end-to-start wrapping are used for the estimate.
    """
    if not all(np.isfinite(x) and x > 0 for x in [width_s, search_s, bin_s]):
        raise ValueError('Window, search range, and correlation bin must be positive.')
    for duration in [width_s, 2 * search_s]:
        if not np.isclose(duration / bin_s, round(duration / bin_s)):
            raise ValueError('Window and twice the search range must be bin multiples.')
    lo, hi = center_s - width_s / 2, center_s + width_s / 2
    if lo < reference[0] or hi > reference[-1]:
        raise ValueError(f'Incomplete reference coverage around {center_s / 3600:g} hours.')
    if lo - search_s < other[0] or hi + search_s > other[-1]:
        raise ValueError(f'Incomplete search coverage around {center_s / 3600:g} hours.')
    r = reference[np.searchsorted(reference, lo):np.searchsorted(reference, hi)]
    q = other[np.searchsorted(other, lo-search_s):np.searchsorted(other, hi+search_s)]
    if len(r) < 50 or len(q) < 50:
        raise ValueError('Too few events for the proposed local timing search.')
    nr = int(round(width_s / bin_s))
    nq = int(round((width_s + 2 * search_s) / bin_s))
    rc = np.histogram(r, bins=np.linspace(lo, hi, nr+1))[0].astype(float)
    qc = np.histogram(q, bins=np.linspace(lo-search_s, hi+search_s, nq+1))[0].astype(float)
    cc = correlate(qc, rc, mode='valid', method='fft')
    best = int(np.argmax(cc))
    coarse_lag = -search_s + best * bin_s
    boundary = best in [0, len(cc)-1]
    # Resolve the narrow timing peak near the coarse lag, independently in each window.
    radius, fine_bin = .04, .00025
    left = np.searchsorted(q, r + coarse_lag - radius)
    right = np.searchsorted(q, r + coarse_lag + radius)
    parts = [q[i:j] - t for t, i, j in zip(r, left, right) if j > i]
    if not parts:
        raise ValueError('No local event pairs were found.')
    differences = np.concatenate(parts)
    edges = coarse_lag + np.linspace(-radius, radius, int(round(2*radius/fine_bin))+1)
    hist = np.histogram(differences, bins=edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = gaussian_filter1d(hist.astype(float), sigma=.00075/fine_bin)
    lag = float(centers[int(np.argmax(smoothed))])
    background = float(np.mean(hist[np.abs(centers-lag) > .01]))
    selected = np.abs(centers-lag) <= .005
    excess = hist[selected].astype(float) - background
    if excess.sum() > 0:
        lag = float(np.sum(centers[selected] * excess) / excess.sum())
    cluster = int(np.count_nonzero(np.abs(differences-lag) <= .005))
    expected = background * (.01 / fine_bin)
    shifts = [-.5, -.25, -.1, .1, .25, .5]
    controls = [_pair_count(r, q, lag+s) for s in shifts]
    control_mean = float(np.mean(controls))
    # Account conservatively for the coarse search and every fine histogram bin.
    # Pair counts only approximately follow an independent Poisson background:
    # this is an evidence screen, not a certified false-positive probability.
    trial_count = len(cc) * len(hist)
    adjusted_tail_screen = float(min(1., poisson.sf(cluster-1, expected)*trial_count))
    supported = bool(not boundary and cluster-expected >= 50 and
                     cluster > max(controls) + 3*np.sqrt(max(control_mean, 1)) and
                     adjusted_tail_screen <= .001)
    off_peak = cc[np.abs(np.arange(len(cc))-best) > max(1, round(.1/bin_s))]
    record = {'center_hours': center_s/3600, 'lag_other_minus_reference_s': lag,
              'coarse_lag_s': coarse_lag, 'coarse_peak_pairs': float(cc[best]),
              'coarse_off_peak_max_pairs': float(np.max(off_peak)),
              'pairs_in_10ms_band': cluster, 'estimated_accidentals_in_10ms_band': expected,
              'estimated_excess_pairs': cluster-expected,
              'shifted_control_mean_pairs': control_mean,
              'shifted_control_max_pairs': max(controls),
              'search_adjusted_poisson_tail_screen': adjusted_tail_screen,
              'search_boundary': boundary, 'timing_signal_supported': supported}
    shifted_parts = [q[i:j]-t for t,i,j in zip(
        r, np.searchsorted(q,r+coarse_lag+.25-radius),
        np.searchsorted(q,r+coarse_lag+.25+radius)) if j > i]
    shifted_differences = np.concatenate(shifted_parts) if shifted_parts else np.array([])
    detail = {'centers_s': centers, 'histogram': hist, 'lag_s': lag,
              'background_per_bin': background,
              'shifted_histogram': np.histogram(shifted_differences,bins=edges+.25)[0],
              'fine_bin_s': fine_bin}
    return record, detail


def validate_split(train_centers_s, validation_centers_s, width_s, search_s):
    train = np.asarray(train_centers_s, dtype=float)
    validation = np.asarray(validation_centers_s, dtype=float)
    if len(train) < 3 or len(validation) < 2:
        raise ValueError('Use at least three training and two validation windows.')
    if not np.isfinite(np.r_[train, validation]).all():
        raise ValueError('Window centers must be finite.')
    all_centers = np.sort(np.r_[train, validation])
    if np.any(np.diff(all_centers) <= width_s + 2 * search_s):
        raise ValueError('Windows or search margins overlap; choose separated centers.')


def fit_clock(training_rows, validation_rows):
    if not training_rows.timing_signal_supported.all():
        raise ValueError('At least one training window lacks a supported timing signal. Review first.')
    x = training_rows.center_hours.to_numpy() * 3600
    y = training_rows.lag_other_minus_reference_s.to_numpy()
    slope, intercept, _, _ = theilslopes(y, x)  # Training rows only.
    if 1 + slope <= 0:
        raise ValueError('Fitted clock map is not monotonic.')
    constant = float(np.median(y))
    xv = validation_rows.center_hours.to_numpy() * 3600
    observed = validation_rows.lag_other_minus_reference_s.to_numpy()
    residual = observed - (intercept + slope*xv)
    model = {'status': 'candidate_relative_clock_map',
             'equation': 'other_time = offset_s + (1 + drift) * reference_time',
             'inverse_equation': 'reference_time_candidate = (other_time - offset_s) / (1 + drift)',
             'offset_s': float(intercept), 'drift': float(slope),
             'relative_drift_ppm': float(slope*1e6),
             'inverse_a_s': float(-intercept/(1+slope)),
             'inverse_b': float(1/(1+slope)),
             'training_center_domain_reference_s': [float(x.min()), float(x.max())]}
    metrics = {'training_windows': len(training_rows), 'validation_windows': len(validation_rows),
               'supported_validation_windows': int(validation_rows.timing_signal_supported.sum()),
               'constant_offset_validation_mae_ms': float(np.mean(np.abs(observed-constant))*1000),
               'affine_validation_mae_ms': float(np.mean(np.abs(residual))*1000),
               'affine_validation_max_abs_error_ms': float(np.max(np.abs(residual))*1000),
               'metric_definition': 'Error in locally estimated lag at held-out window centers; not particle matching accuracy.'}
    return model, metrics


def map_to_reference(native_times, model):
    """Return candidate times and a flag for the span between training centers.

    Do not silently treat extrapolated times as validated. This function does
    not change the input array, amplitudes, deadtime values, or event labels.
    """
    t = np.asarray(native_times, dtype=float)
    if not np.isfinite(t).all():
        raise ValueError('Native event times must be finite.')
    mapped = (t-model['offset_s']) / (1+model['drift'])
    lo, hi = model['training_center_domain_reference_s']
    return mapped, (mapped >= lo) & (mapped <= hi)


def audit_logger(path, rollover_period=4294.967296):
    cols = ['Events CW1&2', 'Events CW1&2&3', 'Date[MM/DD/YYYY]',
            'RTC[H:M:S EST]', 'Timer[S]']
    d = pd.read_csv(path, usecols=cols)
    raw = d['Timer[S]'].to_numpy(dtype=float)
    counters = d[['Events CW1&2','Events CW1&2&3']].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(counters).all():
        raise ValueError('Logger timer/counters contain missing or nonfinite values.')
    if np.any(raw < 0) or np.any(raw >= rollover_period) or np.any(np.diff(counters, axis=0) < 0):
        raise ValueError('Logger period or counter continuity assumptions fail.')
    reversals = np.diff(raw) < 0
    candidate = raw + np.r_[0, np.cumsum(reversals)] * rollover_period
    if np.any(np.diff(candidate) <= 0):
        raise ValueError('Candidate logger clock is not increasing.')
    rtc = d['RTC[H:M:S EST]'].str.split(':', expand=True).astype(float)
    tod = rtc[0].to_numpy()*3600 + rtc[1].to_numpy()*60 + rtc[2].to_numpy()
    if not np.isfinite(tod).all() or np.any((tod < 0) | (tod >= 86400)):
        raise ValueError('Invalid logger RTC time of day.')
    delta = np.diff(tod)
    # Signed daily unwrapping preserves small backwards steps and handles a
    # midnight crossing followed by a jitter-induced reverse crossing.
    rtc_span_s = float(np.sum((delta+43200) % 86400 - 43200))
    return {'filename': Path(path).name, 'sha256': fingerprint(path), 'rows': len(d),
            'status': 'recording_coverage_unresolved_not_aligned',
            'rollover_period_s_candidate': rollover_period,
            'rollover_candidates': int(reversals.sum()),
            'timer_span_hours_candidate': float((candidate[-1]-candidate[0])/3600),
            'rtc_daily_unwrap_span_hours_candidate': rtc_span_s/3600,
            'recorded_dates': d['Date[MM/DD/YYYY]'].value_counts().to_dict(),
            'rtc_backward_steps': int(np.count_nonzero(delta < 0)),
            'rtc_signed_midnight_crossings': int(np.sum(delta < -43200)-np.sum(delta > 43200)),
            'utc_verified': False, 'correction_applied': False}


def plot_results(anchors, models, output_dir, reference_id, run_id, details):
    devices = list(models)
    fig, axes = plt.subplots(2, len(devices), figsize=(7*len(devices), 8), squeeze=False)
    for j, device in enumerate(devices):
        rows = anchors[anchors.device == device]
        model = models[device]
        for split, color, marker in [('train','#176b87','o'),('validation','#cc651b','x')]:
            q = rows[rows.split == split]
            axes[0,j].scatter(q.center_hours, q.lag_other_minus_reference_s,
                              color=color, marker=marker, label=split.capitalize(), s=38)
        xx = np.linspace(rows.center_hours.min(), rows.center_hours.max(), 300)
        axes[0,j].plot(xx, model['offset_s']+model['drift']*xx*3600,
                       color='#333333', label='Training-only affine fit')
        axes[0,j].set(title=f'{device} relative to {reference_id}',
                       ylabel=f'{device} time minus {reference_id} time [s]')
        axes[0,j].legend(fontsize=10)
        q = rows[rows.split == 'validation']
        residual = q.lag_other_minus_reference_s-(model['offset_s']+model['drift']*q.center_hours*3600)
        axes[1,j].scatter(q.center_hours, residual*1000, color='#cc651b', s=38)
        axes[1,j].axhline(0, color='black', lw=1)
        axes[1,j].set(xlabel=f'{reference_id} native elapsed time [hours]',
                       ylabel='Held-out local lag minus predicted lag [ms]')
        for ax in axes[:,j]:
            ax.grid(alpha=.2)
    fig.suptitle(f'{run_id}: candidate detector-to-detector clock alignment', fontsize=16)
    fig.text(.5,.015,'Relative timing only. Local lag errors are not UTC accuracy or verified particle coincidences.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.95])
    fig.savefig(output_dir/'clock_alignment.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(devices), figsize=(7*len(devices), 4), squeeze=False)
    for j, device in enumerate(devices):
        d = details[device]
        x = (d['centers_s']-d['lag_s'])*1000
        axes[0,j].step(x,d['histogram'],where='mid',label='Near candidate timing peak',color='#176b87')
        axes[0,j].step(x,d['shifted_histogram'],where='mid',label='Offset shifted by 250 ms',color='#cc651b',alpha=.8)
        axes[0,j].axhline(d['background_per_bin'],color='#777777',ls=':',label='Off-peak background')
        axes[0,j].set(xlim=(-20,20),xlabel='Time difference minus local candidate lag [ms]',
                       ylabel='Event pairs per 0.25 ms bin',title=f'{device} versus {reference_id}')
        axes[0,j].legend(fontsize=9)
        axes[0,j].grid(alpha=.2)
    fig.suptitle('Narrow shared timing signals in a training window',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.92])
    fig.savefig(output_dir/'timing_signal.png',dpi=160)
    plt.close(fig)


def run(root, detector_files, reference_id, run_id, train_centers_s,
        validation_centers_s, logger_file=None, window_s=600., search_s=120., bin_s=.01):
    root = Path(root).resolve()
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in run_id):
        raise ValueError('Use a simple run identifier.')
    if reference_id not in detector_files or len(detector_files) < 2:
        raise ValueError('Select a reference and at least one other detector.')
    paths = {k: Path(v).resolve() for k,v in detector_files.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError('Each detector must have a different source file.')
    validate_split(train_centers_s, validation_centers_s, window_s, search_s)
    before = {k:fingerprint(p) for k,p in paths.items()}
    if len(set(before.values())) != len(before):
        raise ValueError('Duplicate detector file contents cannot establish synchronization.')
    times, inventory = {}, {}
    for device,path in paths.items():
        times[device],inventory[device] = read_times(path)
        inventory[device]['sha256'] = before[device]
    reference = times[reference_id]
    rows, details = [], {}
    devices = [d for d in paths if d != reference_id]
    example_center = np.asarray(train_centers_s)[len(train_centers_s)//2]
    for split, centers in [('train',train_centers_s),('validation',validation_centers_s)]:
        for center in centers:
            for device in devices:
                row, detail = local_lag(reference,times[device],float(center),window_s,search_s,bin_s)
                row.update(device=device,split=split)
                rows.append(row)
                if split == 'train' and center == example_center:
                    details[device] = detail
    anchors = pd.DataFrame(rows)
    models,metrics = {},{}
    for device in devices:
        q = anchors[anchors.device == device]
        models[device],metrics[device] = fit_clock(q[q.split=='train'],q[q.split=='validation'])
    logger_before = fingerprint(logger_file) if logger_file is not None else None
    logger_audit = audit_logger(logger_file) if logger_file is not None else None
    if logger_file is not None and fingerprint(logger_file) != logger_before:
        raise RuntimeError('The logger source changed during analysis.')
    after = {k:fingerprint(p) for k,p in paths.items()}
    if after != before:
        raise RuntimeError('A detector source changed during analysis.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = root/'results'/run_id/f'{stamp}_clock_alignment'
    output_dir.mkdir(parents=True,exist_ok=False)
    report = {'status':'exploratory_candidate_clock_models_no_source_rewriting',
              'alignment_script_sha256':fingerprint(__file__),
              'run_id':run_id,'reference_id':reference_id,
              'settings':{'window_seconds':window_s,'search_seconds':search_s,
                          'coarse_bin_seconds':bin_s,'local_pair_band_seconds':.01,
                          'maximum_search_adjusted_poisson_tail_screen':.001,
                          'train_centers_s':list(map(float,train_centers_s)),
                          'validation_centers_s':list(map(float,validation_centers_s))},
              'inputs':inventory,'models':models,'metrics':metrics,'logger_audit':logger_audit,
              'source_hashes_unchanged':True,'utc_verified':False,
              'raw_timestamps_rewritten':False,'particle_identities_verified':False,
              'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                         'pandas':pd.__version__,'scipy':scipy.__version__},
              'limitations':[
                  'Reference clock is arbitrary; absolute duration and UTC accuracy are unverified.',
                  'Event-pair timing excesses do not prove particle identity or detector wiring.',
                  'Validation errors measure predicted local lag, not individual-event timing precision.',
                  'Sampled windows do not certify every intervening instant or extrapolated edges.',
                  'Affine residual structure may require more local corrections for tighter timing.',
                  'Clock parameters apply only to this recording; do not transfer them to another run.',
                  'Logger recording coverage is unresolved and no logger alignment is fitted.',
                  'Calibration, pressure corrections, deadtime interpretation, and flight phases are unchanged.']}
    (output_dir/'alignment_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    anchors.to_csv(output_dir/'local_lags.csv',index=False)
    pd.DataFrame([{'device':k,**v} for k,v in metrics.items()]).to_csv(output_dir/'validation_metrics.csv',index=False)
    plot_results(anchors,models,output_dir,reference_id,run_id,details)
    return report, anchors, output_dir
