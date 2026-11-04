# Choose the failure-threshold rule on the ARCHIVE wheels only (leave one out), never on SAT_2.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'fed'); import rul
FLEET = rul.FLEET; A = {}
for a in ('ARC_1', 'ARC_2', 'ARC_3'):
    raw = f'{FLEET}/archive/{a}'; wj = json.load(open(f'{raw}/truth/wear.json'))
    H, _, _ = rul.hi(rul.daily_ratio(raw), rul.REF_DAYS['ARC']); wn = f"rw{wj['wheel'] + 1}_current"
    A[a] = dict(h=H[wn].values, f=wj['t_fail'] / 86400, last=float(H[wn].iloc[int(wj['t_fail'] // 86400) - 1]))
rules = dict(min=np.min, mean=np.mean, median=np.median, max=np.max)
print('archive HI at failure', {k: round(v['last'], 1) for k, v in A.items()})
for win in (15, 30):
    for name, fn in rules.items():
        errs = []
        for a, v in A.items():
            th = fn([w['last'] for k, w in A.items() if k != a])
            for t in range(int(v['f']) - 40, int(v['f'])):
                tt = np.arange(t - win, t + 1); p = rul.fit_rul(v['h'][t - win:t + 1], tt, th, 'exp')
                if np.isfinite(p): errs.append(p - (v['f'] - t))
        e = np.array(errs); phm = np.mean(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)); print(f'window {win:2d} rule {name:6s}: PHM/pred {phm:.4f}  MAE {np.abs(e).mean():5.2f}  bias {e.mean():+5.2f}  late {100 * (e > 0).mean():3.0f}%')
