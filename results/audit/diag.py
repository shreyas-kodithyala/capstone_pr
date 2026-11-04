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
tru = pd.read_csv(f"{d['raw']}/truth/events_truth.csv"); tru = tru[tru.Visible]
idn = pd.read_csv(f"{d['raw']}/channels.csv").set_index('Channel').Name.to_dict()
out = [f'##### {sat} {arm}']
evs = []
for pn, per in PER.items():
    P, ts, y = per; Z = P.zs[('all', 's15')][0]; t = ts.values; n = len(t)
    gt = np.zeros(n, bool)
    for a, b in zip(y.StartTime.values, y.EndTime.values):
        i0, i1 = np.searchsorted(t, [a, b]); gt[i0:i1 + 1] = True
    rows = []
    for _, e in tru[tru.Category == 'Anomaly'].iterrows():
        a = np.datetime64(pd.Timestamp(e.FirstVisible).tz_localize(None)); b = np.datetime64(pd.Timestamp(e.LastVisible).tz_localize(None))
        if a < t[0] or b > t[-1]: continue
        i0, i1 = np.searchsorted(t, [a, b]); i1 = max(i1, i0 + 1)
        lab = [names.index(idn[c]) for c in str(e.Channels).split() if idn[c] in names]
        zl = Z[i0:i1][:, lab].max(0); j = int(np.argmax(zl)); zall = Z[i0:i1].max(0)
        rows.append(dict(per=pn, code=e.Code, fam=e.FaultFamily, hrs=(i1 - i0) / 120, zlab=float(zl[j]), ch=names[lab[j]], zany=float(zall.max()), chany=names[int(np.argmax(zall))], i0=i0, i1=i1))
    ev = pd.DataFrame(rows); evs.append(ev)
    days = n / F.RPD
    line = []
    for tau in (10, 20, 30, 50, 80, 120, 200):
        A = (Z > tau); g = D.open_runs(D.close_runs(A.any(1).astype(np.uint8), 10), 10)
        r = D.runs_of(g); fp = [(s, e_) for s, e_ in r if not gt[s:e_].any()]
        hit = sum(bool(g[x.i0:x.i1].any()) for x in ev.itertuples())
        line.append(f'{tau}: FP {len(fp)} ({len(fp)/days*30:.1f}/30d) hit {hit}/{len(ev)}')
        if tau == 30:
            ch, cm, dt = [], [], []
            for s, e_ in fp:
                ch.append(names[int(np.argmax(Z[s:e_].max(0)))])
                k = np.searchsorted(ct, t[s]) - 1
                cm.append(cn[k]); dt.append((t[s] - ct[k]) / np.timedelta64(1, 'm'))
            fpd = pd.DataFrame(dict(ch=ch, cm=cm, dt=dt, dur=[(e_ - s) / 2 for s, e_ in fp]))
    out.append(f'[{pn}] days {days:.0f}: ' + ' | '.join(line))
    if len(fpd):
        out.append(f'   FP@30 channels {fpd.ch.value_counts().head(6).to_dict()}')
        out.append(f'   FP@30 last cmd {fpd.cm.value_counts().head(6).to_dict()}; median min since cmd {fpd.dt.median():.0f}; within 20min {100*(fpd.dt<20).mean():.0f}%; median dur {fpd.dur.median():.0f} min')
ev = pd.concat(evs)
if len(ev):
    q = ev.groupby(['per', 'code']).agg(n=('zlab', 'size'), hrs=('hrs', 'median'), zlab_min=('zlab', 'min'), zlab_med=('zlab', 'median'), zlab_max=('zlab', 'max'), ch=('ch', lambda s: s.value_counts().index[0]))
    out.append(q.round(1).to_string())
print('\n'.join(out))
