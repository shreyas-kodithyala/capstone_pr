import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path.insert(0, 'fed'); os.environ['FED_OUT'] = os.path.abspath('fed/out_final')
import torch, fedsim as F
from core.model import GDN2; from core.config import MODEL; from core.train import series_errors, seg_bs
import config as C
out = {}
# (a) reproducibility: generator code hash equals the hash stored with the seeds
h = hashlib.sha256()
for f in ('config.py', 'envmodels.py', 'operations.py', 'physics.py', 'telemetry.py', 'products.py'):
    h.update(open(f'sim/{f}', 'rb').read())
fj = json.load(open('sim/fleet_final.json')); print('generator code unchanged since final seeds:', h.hexdigest() == fj['code_sha256'])
# (b,c,d) leakage
D = {s: F.inputs(s) for s in F.SATS}; rf = GDN2(20, 0, 12, **MODEL).rf; seg, bs = seg_bs(rf)
for s in F.SATS:
    d = D[s]; assert not any('is_anomaly' in str(c) for c in d['chs'])
    print(s, 'train rows end', d['es'] - 2, 'early-stop rows end', d['v0'] - 1, 'val window', d['v0'], d['ntr'], 'test starts', d['ntr'], '| mu/sd use rows <', d['v0'])
dev = torch.device('cpu'); s = 'SAT_2'; d = D[s]; m = F.new_model(d, dev); m.load_state_dict(torch.load('fed/out_final/models/pers_SAT_2.pt', map_location='cpu'))
a = d['ntr'] + 50000; b = a + 3000; t0 = a + 1500
E1 = series_errors(m, d['X'], d['cmd'], dev, a, b)
X2 = d['X'].copy(); X2[t0:b + 2] += np.random.default_rng(0).normal(0, 5, X2[t0:b + 2].shape).astype(np.float32)
E2 = series_errors(m, X2, d['cmd'], dev, a, b)
print('model causality: max change in errors before the perturbed row', float(np.abs(E1[:t0 - a] - E2[:t0 - a]).max()), '| first perturbed row change', float(np.abs(E1[t0 - a + 1] - E2[t0 - a + 1]).max()))
# (e) ESA precision identity on the reported counts
R = json.load(open('fed/out_final/results.json'))
for s in F.SATS:
    r = R['pers'][s]['federated']; tp = r['n_detected']; fp = r['n_false_alarm_events']; p = tp / max(tp + fp, 1) * (1 - r['false_alarm_time_share'])
    print(s, 'EW precision from counts %.3f vs ESA metric %.3f' % (p, r['anomalies']['global_EW_precision']))
# (g) bootstrap
rng = np.random.default_rng(0)
def fb(p, r, b=.5): return 0 if p + r == 0 else (1 + b * b) * p * r / (b * b * p + r)
def sample(arm, sel):
    tp = fn = fp = 0; per = []
    for s in F.SATS:
        r = R[arm][s][sel]; ev = np.array([e['detected'] for e in r['events']]); k = rng.choice(len(ev), len(ev)); t = ev[k].sum(); f = len(ev) - t; q = rng.poisson(r['n_false_alarm_events'])
        P = t / max(t + q, 1) * (1 - r['false_alarm_time_share']); per.append(fb(P, t / len(ev))); tp += t; fn += f; fp += q
    return np.mean(per), tp, fp
B = {k: [] for k in ('solo', 'fed', 'pers')}
for _ in range(5000):
    st = rng.bit_generator.state; B['solo'].append(sample('solo', 'local'))
    rng.bit_generator.state = st; B['pers'].append(sample('pers', 'federated'))
    rng.bit_generator.state = st; B['fed'].append(sample('fed', 'federated'))
q = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 50, 97.5])]
for k in B: print('macro F0.5', k, q([b[0] for b in B[k]]))
for k in ('fed', 'pers'):
    dlt = np.array([x[0] - y[0] for x, y in zip(B[k], B['solo'])]); print(f'{k} - solo macro F0.5', q(dlt), 'P(gain>0)', round(float((dlt > 0).mean()), 3))
    dfp = np.array([x[2] - y[2] for x, y in zip(B[k], B['solo'])]); print(f'{k} - solo false alarms', q(dfp))
# (h) spread across the two fleets
V = json.load(open('fed/out_v2/reports/round5b_results.json'))
for arm, sel in (('solo', 'local'), ('fed', 'federated'), ('pers', 'federated')):
    print(arm, sel, 'dev', [round(V[arm][s][sel]['anomalies']['global_EW_F_0.50'], 3) for s in F.SATS], 'final', [round(R[arm][s][sel]['anomalies']['global_EW_F_0.50'], 3) for s in F.SATS])
# (f) controls and ablations on the final fleet: analysis only, nothing here feeds back into the design
def line(tag, arm, per, cfgs):
    v = []
    for s in F.SATS:
        c = cfgs[s] if isinstance(cfgs, dict) and s in cfgs else cfgs
        P, ts, y = per[s]['test']; A = F.alarms(P, c); ev = F.event_table(D[s], A, ts); fa = F.false_alarms(ts, A, y); sc = F.B.score(D[s]['raw'], ts, A, D[s]['tg'], F.B.SEL_ANOM)
        v.append((round(sc.get('global_EW_F_0.50', 0), 3), int(ev.detected.sum()), fa['n_false_alarm_events']))
    print(f'{tag:44s}', v, 'mean %.3f caught %d FA %d' % (np.mean([x[0] for x in v]), sum(x[1] for x in v), sum(x[2] for x in v)), flush=True)
for arm in ('solo', 'pers'):
    PER = F.load_periods(D, arm, rf); fc = R['pers']['fed_selection']['cfg']
    line(f'{arm}: reported local settings', arm, PER, {s: R[arm][s]['local']['cfg'] for s in F.SATS})
    line(f'{arm}: reported fleet settings', arm, PER, R[arm]['fed_selection']['cfg'])
    line(f'{arm}: no tuning (default k2)', arm, PER, F.START)
    line(f'{arm}: no tuning, k3', arm, PER, dict(F.START, thr='k3'))
    if arm == 'pers':
        line('pers fleet, no command mask', arm, PER, dict(fc, mask=0, smask=0))
        line('pers fleet, one absolute threshold a30', arm, PER, dict(fc, thr='a30'))
        line('pers fleet, own ceilings', arm, PER, dict(fc, ceil='own'))
        loso = {}
        for s in F.SATS:
            loso[s], _ = F.search([PER[q]['val'] for q in F.SATS if q != s], axes=dict(F.AXES, ceil=['own', 'fleet']))
        print('LOSO settings', {s: (c['thr'], c.get('ceil'), c.get('W')) for s, c in loso.items()})
        line('pers leave-one-satellite-out selection', arm, PER, loso)
