"""Optional real-data negative control; run from the repository root.

Independently jitter every target event within +/-0.5 s, preserving broad rates
while destroying narrow shared timing. The six checks are diagnostics, not an
estimate of a universal false-positive rate.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from align_clocks import read_times,local_lag

def main():
    data=ROOT/'raw/ground_run_01'
    ref,_=read_times(data/'top.txt')
    rng=np.random.default_rng(20261116)
    rows=[]
    for device in ['mid','bot']:
        times,_=read_times(data/f'{device}.txt')
        for h in [6.5,102.5,186.5]:
            lo,hi=h*3600-420,h*3600+420
            q=times[np.searchsorted(times,lo-1):np.searchsorted(times,hi+1)].copy()
            q=np.sort(q+rng.uniform(-.5,.5,len(q)))
            row,_=local_lag(ref,q,h*3600)
            row.update(device=device,control='independent_per_event_jitter_plus_minus_0.5s')
            rows.append(row)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results/ground_run_01'/f'{stamp}_clock_controls'
    out.mkdir(parents=True,exist_ok=False)
    (out/'jitter_controls.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    accepted=sum(r['timing_signal_supported'] for r in rows)
    print(f'Jittered controls accepted by evidence screen: {accepted} / {len(rows)}')
    print(out)
    if accepted:
        raise RuntimeError('A jittered control passed. Investigate before trusting alignment screens.')

if __name__=='__main__':
    main()
