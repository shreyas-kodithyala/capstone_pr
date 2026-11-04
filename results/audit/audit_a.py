"""Audit A (bsk env): classifier two-sample test real vs simulated on catch22, physics numbers, ESA mix."""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, 'sim')
import pycatch22
v = json.load(open('results/final/validation_part1.json'))
for s, x in v['physics'].items():
    print(s, {k: (round(y, 4) if isinstance(y, float) else y) for k, y in x['battery'].items()} if isinstance(x.get('battery'), dict) else x.get('battery'))
    print('   ', json.dumps(x['eclipse'])[:170]); print('   ', json.dumps(x['actuator_limits'])[:170])
ref = json.load(open('sim/esa_ref_labels.json'))
print('ESA mix M1', ref['M1']['anomaly_type_mix'], 'point', ref['M1']['anomaly_point_frac'])
print('ESA mix M2', ref['M2']['anomaly_type_mix'], 'point', ref['M2']['anomaly_point_frac'])
rows = []
for s in ('SAT_1', 'SAT_2', 'SAT_3', 'SAT_4'):
    t = pd.read_csv(f'data/sim_fleet/{s}/anomaly_types.csv'); a = t[t.Category == 'Anomaly']
    tr = pd.read_csv(f'data/sim_fleet/{s}/truth/events_truth.csv'); tr = tr[(tr.Category == 'Anomaly') & tr.Visible]
    d = (pd.to_datetime(tr.LastVisible) - pd.to_datetime(tr.FirstVisible)).dt.total_seconds() / 3600
    rows.append(a.assign(dur=d.values[:len(a)] if len(d) == len(a) else np.nan))
A = pd.concat(rows); mix = (A.Dimensionality + '-' + A.Locality).value_counts(normalize=True).round(3).to_dict()
print('OUR mix', mix, 'point frac', round((A.Length == 'Point').mean(), 3), 'n', len(A))
print('duration h quantiles', A.dur.quantile([.1, .5, .9]).round(2).to_dict(), 'codes W excluded median')
