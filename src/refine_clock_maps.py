"""Candidate local clock maps, using 04 fitting anchors and fresh validation windows.

No UTC claims, particle labels, logger alignment, or extrapolation are made.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import align_clocks


def build_map(training):
    q = training.sort_values('center_hours')
    if len(q) < 3 or not q['timing_signal_supported'].astype(str).str.lower().eq('true').all():
        raise ValueError('At least three supported fitting anchors are required.')
    reference = q.center_hours.to_numpy(dtype=float)*3600
    lag = q.lag_other_minus_reference_s.to_numpy(dtype=float)
    other = reference + lag
    if not np.isfinite(np.r_[reference,other]).all() or np.any(np.diff(reference)<=0) or np.any(np.diff(other)<=0):
        raise ValueError('Anchor clocks must be finite and strictly increasing.')
    return {'status':'candidate_relative_piecewise_linear_map',
            'reference_anchor_seconds':reference.tolist(),
            'device_anchor_seconds':other.tolist(),
            'lag_anchor_seconds':lag.tolist(),
            'extrapolation_allowed':False}


def map_times(native_times, model):
    """Return candidate reference-clock times and an inside-anchor-span flag.

    Outside times remain NaN. The flag is a domain check, not a precision guarantee.
    """
    t=np.asarray(native_times,dtype=float)
    if not np.isfinite(t).all():
        raise ValueError('Input event times must be finite.')
    x=np.asarray(model['device_anchor_seconds'],dtype=float)
    y=np.asarray(model['reference_anchor_seconds'],dtype=float)
    if len(x)!=len(y) or len(x)<3 or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
        raise ValueError('Invalid clock-map knots.')
    inside=(t>=x[0]) & (t<=x[-1])
    mapped=np.full(t.shape,np.nan,dtype=float)
    mapped[inside]=np.interp(t[inside],x,y)
    return mapped,inside


def predict_lag(reference_times, model):
    t=np.asarray(reference_times,dtype=float)
    x=np.asarray(model['reference_anchor_seconds'],dtype=float)
    if not np.isfinite(t).all() or np.any((t<x[0]) | (t>x[-1])):
        raise ValueError('Local lag prediction would extrapolate outside the anchors.')
    return np.interp(t,x,np.asarray(model['lag_anchor_seconds']))


def latest_alignment(root, run_id):
    choices=[p for p in (Path(root)/'results'/run_id).glob('*_clock_alignment')
             if (p/'alignment_report.json').is_file() and (p/'local_lags.csv').is_file()]
    if not choices:
        raise FileNotFoundError(f'No 04 clock-alignment output for {run_id}; run 04 first.')
    return max(choices,key=lambda p:p.name)


def _plot(anchors,fresh,models,baseline,out,run_id,reference_id):
    devices=list(models)
    fig,axes=plt.subplots(2,len(devices),figsize=(7*len(devices),8),squeeze=False)
    for j,device in enumerate(devices):
        tr=anchors[(anchors.device==device)&(anchors.split=='train')]
        q=fresh[fresh.device==device]
        axes[0,j].plot(tr.center_hours,tr.lag_other_minus_reference_s,'o-',color='#176b87',
                       label='Local map through fitting anchors',markersize=4)
        axes[0,j].scatter(q.center_hours,q.lag_other_minus_reference_s,color='#cc651b',marker='x',
                          label='Fresh validation measurements',s=35)
        b=baseline['models'][device]
        x=np.linspace(tr.center_hours.min(),tr.center_hours.max(),200)
        axes[0,j].plot(x,b['offset_s']+b['drift']*x*3600,color='#777777',ls='--',label='04 constant-drift model')
        axes[0,j].set(title=f'{device} relative to {reference_id}',ylabel='Clock lag [seconds]')
        axes[0,j].legend(fontsize=9)
        axes[1,j].plot(q.center_hours,q.affine_error_ms,'x--',color='#999999',label='04 model on fresh windows')
        axes[1,j].plot(q.center_hours,q.local_error_ms,'o-',color='#176b87',label='Local map on the same windows')
        axes[1,j].axhline(0,color='black',lw=1)
        axes[1,j].set(xlabel=f'{reference_id} native elapsed time [hours]',ylabel='Observed minus predicted local lag [ms]')
        axes[1,j].legend(fontsize=9)
        for ax in axes[:,j]:ax.grid(alpha=.2)
    fig.suptitle(f'{run_id}: candidate local clock refinement',fontsize=16)
    fig.text(.5,.015,'Fresh-window lag errors are not UTC accuracy or verified particle matching accuracy.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(out/'clock_refinement.png',dpi=160);plt.close(fig)


def run(root,run_id,detector_files,alignment_dir,fresh_centers_s):
    root=Path(root).resolve();alignment_dir=Path(alignment_dir).resolve()
    baseline=json.loads((alignment_dir/'alignment_report.json').read_text(encoding='utf8'))
    anchors=pd.read_csv(alignment_dir/'local_lags.csv')
    if baseline['run_id']!=run_id or set(baseline['inputs'])!=set(detector_files):
        raise ValueError('04 run/detector identities do not match the selected files.')
    if baseline['alignment_script_sha256']!=align_clocks.fingerprint(align_clocks.__file__):
        raise ValueError('align_clocks.py differs from the 04 version; review before mixing results.')
    reference_id=baseline['reference_id'];settings=baseline['settings']
    fresh=np.asarray(fresh_centers_s,dtype=float)
    old_validation=np.asarray(settings['validation_centers_s'],dtype=float)
    # This also excludes overlap with every 04 fitting/validation search margin.
    align_clocks.validate_split(settings['train_centers_s'],np.r_[old_validation,fresh],
                                settings['window_seconds'],settings['search_seconds'])
    paths={k:Path(v).resolve() for k,v in detector_files.items()}
    before={k:align_clocks.fingerprint(p) for k,p in paths.items()}
    if any(before[k]!=baseline['inputs'][k]['sha256'] for k in paths):
        raise ValueError('A raw source differs from the source used for the 04 anchors.')
    models={};times={}
    for device,path in paths.items():times[device],_=align_clocks.read_times(path)
    for device in baseline['models']:
        tr=anchors[(anchors.device==device)&(anchors.split=='train')]
        model=build_map(tr)
        predict_lag(fresh,model)  # Stop before evaluating if any fresh center is outside.
        models[device]=model
    rows=[]
    for device,model in models.items():
        for center in fresh:
            row,_=align_clocks.local_lag(times[reference_id],times[device],float(center),
                width_s=settings['window_seconds'],search_s=settings['search_seconds'],
                bin_s=settings['coarse_bin_seconds'])
            local=float(predict_lag(np.array([center]),model)[0])
            b=baseline['models'][device]
            affine=b['offset_s']+b['drift']*center
            row.update(device=device,affine_predicted_lag_s=affine,local_predicted_lag_s=local,
                       affine_error_ms=1000*(row['lag_other_minus_reference_s']-affine),
                       local_error_ms=1000*(row['lag_other_minus_reference_s']-local))
            rows.append(row)
    fresh_rows=pd.DataFrame(rows)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=root/'results'/run_id/f'{stamp}_clock_refinement';out.mkdir(parents=True,exist_ok=False)
    fresh_rows.to_csv(out/'fresh_window_validation.csv',index=False)
    # Preserve failed diagnostics, without publishing error scores from unreliable lag estimates.
    metrics=[]
    for device in models:
        q=fresh_rows[fresh_rows.device==device]
        usable=bool(q.timing_signal_supported.all())
        metrics.append({'device':device,'fresh_windows':len(q),
                        'supported_fresh_windows':int(q.timing_signal_supported.sum()),
                        'all_fresh_signals_supported':usable,
                        'affine_fresh_mae_ms':float(q.affine_error_ms.abs().mean()) if usable else None,
                        'local_fresh_mae_ms':float(q.local_error_ms.abs().mean()) if usable else None,
                        'local_fresh_max_abs_error_ms':float(q.local_error_ms.abs().max()) if usable else None})
    pd.DataFrame(metrics).to_csv(out/'refinement_metrics.csv',index=False)
    samples=[];coverage={}
    common_lo=max(m['reference_anchor_seconds'][0] for m in models.values())
    common_hi=min(m['reference_anchor_seconds'][-1] for m in models.values())
    for device,t in times.items():
        if device==reference_id:
            inside=(t>=common_lo)&(t<=common_hi)
            mapped=np.where(inside,t,np.nan)
        else:mapped,inside=map_times(t,models[device])
        if np.any(np.diff(mapped[inside])<=0):
            raise RuntimeError('Candidate mapped times are not strictly increasing.')
        coverage[device]={'source_events':len(t),'events_inside_anchor_span':int(inside.sum()),
                          'events_outside_anchor_span':int((~inside).sum()),
                          'outside_events_preserved_in_original':True}
        for i in np.unique(np.linspace(0,len(t)-1,300).astype(int)):
            samples.append({'device':device,'source_record_1based':int(i+1),
                            'event_time_native_s':float(t[i]),
                            'time_reference_candidate_s':float(mapped[i]) if inside[i] else None,
                            'inside_anchor_span':bool(inside[i])})
    pd.DataFrame(samples).to_csv(out/'corrected_clock_sample.csv',index=False)
    after={k:align_clocks.fingerprint(p) for k,p in paths.items()}
    if before!=after:raise RuntimeError('A raw detector file changed during analysis.')
    report={'status':'candidate_local_clock_maps_no_raw_rewriting',
            'run_id':run_id,'reference_id':reference_id,
            'refinement_script_sha256':align_clocks.fingerprint(__file__),
            'baseline_alignment_script_sha256':baseline['alignment_script_sha256'],
            'baseline_alignment_folder':str(alignment_dir),
            'baseline_report_sha256':align_clocks.fingerprint(alignment_dir/'alignment_report.json'),
            'baseline_lags_sha256':align_clocks.fingerprint(alignment_dir/'local_lags.csv'),
            'source_sha256':before,'source_hashes_unchanged':True,
            'models':models,'metrics':metrics,'mapping_coverage':coverage,
            'fresh_centers_s':fresh.tolist(),'window_seconds':settings['window_seconds'],
            'search_seconds':settings['search_seconds'],
            'models_use_04_training_anchors_only':True,
            'fresh_windows_disjoint_from_04_scheduled_windows':True,
            'logger_aligned':False,'utc_verified':False,'raw_timestamps_rewritten':False,
            'particle_identities_verified':False,'extrapolation_allowed':False,
            'limitations':[
                'Errors describe independently measured local timing-peak positions, not individual-event accuracy.',
                'Anchor-span flags do not guarantee accuracy at every instant between sparse anchors.',
                'Unmapped startup/tail events stay in original files; no extrapolated correction is supplied.',
                'Fresh windows are within the same previously inspected recordings, not independent hardware truth.',
                'Logger clocks and their recording coverage remain separate unresolved problems.',
                'Instrument coincidence gates, calibration, firmware, and physical event identity remain unverified.',
                'A candidate local map is valid only for its own recording and source hashes.']}
    (out/'refinement_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    _plot(anchors,fresh_rows,models,baseline,out,run_id,reference_id)
    return report,fresh_rows,out
