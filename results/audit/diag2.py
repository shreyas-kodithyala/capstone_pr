import sys, os, json
sat, arm = sys.argv[1], sys.argv[2]; sys.argv = sys.argv[:1]
sys.path.insert(0, 'fed'); os.environ.setdefault('FED_OUT', 'fed/out_v2')
import numpy as np, pandas as pd, fedsim as F
from core.model import GDN2; from core.config import MODEL; from core import detect as D
import config as C
d = F.inputs(sat); E = np.load(f'fed/out_v2/errors/{arm}_{sat}.npy'); rf = GDN2(20, 0, 12, **MODEL).rf
PER = F.periods(d, E, rf); names = [d['names'][c] for c in d['tg']]
tcname = {f'telecommand_{i+1}': n for i, n in enumerate(C.TC_NAMES)}
cmds = sorted((t, tcname[c]) for c, _, ts in d['cmds'] for t in ts); ct = np.array([c[0] for c in cmds]); cn = [c[1] for c in cmds]
sa = d['vals'][:, d['chs'].index([c for c in d['tg'] if d['names'][c] == 'sa_power'][0])]
lit = sa > 5; tr = np.flatnonzero(np.diff(lit.astype(np.int8)) != 0)
out = [f'##### {sat} {arm}']
for pn, per in PER.items():
    P, ts, y = per; t = ts.values; n = len(t); off = {'val': d['v0'], 'test': d['ntr']}[pn]
    gt = np.zeros(n, bool)
    for a, b in zip(y.StartTime.values, y.EndTime.values):
        i0, i1 = np.searchsorted(t, [a, b]); gt[i0:i1 + 1] = True
    # wide event mask: 6 h after any labelled event, to see settling tails
    wide = np.zeros(n, bool)
    for a, b in zip(y.StartTime.values, y.EndTime.values):
        i0, i1 = np.searchsorted(t, [a, b + np.timedelta64(6, 'h')]); wide[i0:i1 + 1] = True
    cfg = dict(F.START, thr='a20', close=10, minlen=30, cross=0, mask=30)
    A = F.alarms(P, cfg); g = A.any(1); r = D.runs_of(g); fp = [(s, e) for s, e in r if not gt[s:e].any()]
    days = n / F.RPD; rows = []
    Z = P.zs[('all', 's15')][0]
    for s, e in fp:
        k = np.searchsorted(ct, t[s]) - 1
        q = np.searchsorted(tr, s + off); dte = min(abs(s + off - tr[max(q - 1, 0)]), abs(tr[min(q, len(tr) - 1)] - s - off)) / 2
        rows.append(dict(ch=names[int(np.argmax(A[s:e].sum(0)))], cm=cn[k], dt=(t[s] - ct[k]) / np.timedelta64(1, 'm'), dur=(e - s) / 2, ecl=dte, after=bool(wide[s:e].any()), peak=float(Z[s:e].max()), day=(s + off) / F.RPD))
    q = pd.DataFrame(rows)
    out.append(f'[{pn}] FP {len(fp)} ({len(fp)/days*30:.1f}/30d) alarm-time {100*(g&~gt).mean():.2f}%')
    if len(q):
        out.append(f'   by channel {q.ch.value_counts().head(7).to_dict()}')
        out.append(f'   last cmd {q.cm.value_counts().head(6).to_dict()} med since {q.dt.median():.0f} min | <45min {100*(q.dt<45).mean():.0f}% | within 6h after event {100*q.after.mean():.0f}% | within 5min of eclipse edge {100*(q.ecl<=5).mean():.0f}% | dur med {q.dur.median():.0f} max {q.dur.max():.0f} min | peak z med {q.peak.median():.0f}')
        if pn == 'test':
            out.append(f'   FP per 30-day bin {np.histogram(q.day, bins=np.arange(182, 372, 30))[0].tolist()}')
print('\n'.join(out))
