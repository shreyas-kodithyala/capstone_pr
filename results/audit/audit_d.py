import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp
R = np.load('sim/esa_ref_catch22.npy'); O = np.load('.localenv/tmp/sim_c22.npy'); ch = pd.read_csv('.localenv/tmp/sim_c22_ch.csv').iloc[:, 0].values
R[~np.isfinite(R)] = 0; O[~np.isfinite(O)] = 0
def c2st(a, b, seed=0):
    X = np.r_[a, b]; y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return cross_val_score(RF(300, class_weight='balanced', random_state=seed, n_jobs=4), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='roc_auc').mean()
print('C2ST AUC simulated vs real ESA (0.5 = indistinguishable, 1 = trivially separable): %.3f' % c2st(R, O))
print('C2ST AUC real M1 vs real M2 (same-domain reference): %.3f' % c2st(R[:103], R[-103:]))
grp = lambda n: 'wheel' if n.startswith('rw') else 'attitude' if n.startswith(('gyro', 'att')) else 'mag' if n.startswith('mag') else 'power/thermal'
g = np.array([grp(n) for n in ch])
for k in ('wheel', 'attitude', 'mag', 'power/thermal'):
    print(f'  C2ST {k:14s} n={int((g == k).sum()):3d} AUC %.3f' % c2st(R, O[g == k]))
ks = [ks_2samp(R[:, j], O[:, j]).statistic for j in range(22)]
print('per-feature KS: median %.2f, max %.2f, features with KS<0.2: %d of 22' % (np.median(ks), np.max(ks), sum(k < .2 for k in ks)))
# Wu & Keogh one-liner: an event is trivial if a single-channel one-liner fires higher inside it than anywhere outside all labels
print('\nOne-liner test on test anomalies (x, -x, diff, -diff, 1-hour mean)')
tot = {}
for s in ('SAT_1', 'SAT_2', 'SAT_3', 'SAT_4'):
    raw = f'data/sim_fleet/{s}'; df = pd.read_csv(f'{raw}/resampled_30s_test.csv.gz'); c = pd.read_csv(f'{raw}/channels.csv'); tg = c.Channel[c.Target == 'YES'].tolist()
    ts = pd.to_datetime(df.timestamp).dt.tz_localize(None).values; X = df[tg].astype(float); L = df[[f'is_anomaly_{k}' for k in tg]].values
    near = pd.Series((L > 0).any(1)).rolling(240, center=True, min_periods=1).max().values > 0     # labels +- 1 h
    Fn = {'x': X, '-x': -X, 'dx': X.diff().fillna(0), '-dx': -X.diff().fillna(0), 'm': X.rolling(120, min_periods=1).mean(), '-m': -X.rolling(120, min_periods=1).mean()}
    top = {k: v.values[~near].max(0) for k, v in Fn.items()}
    t = pd.read_csv(f'{raw}/truth/events_truth.csv'); t = t[(t.Split == 'test') & t.Visible & (t.Category == 'Anomaly')]
    for _, e in t.iterrows():
        a = np.datetime64(pd.Timestamp(e.FirstVisible).tz_localize(None)); b = np.datetime64(pd.Timestamp(e.LastVisible).tz_localize(None))
        i0, i1 = np.searchsorted(ts, [a, b]); i1 = max(i1, i0 + 1)
        triv = any((v.values[i0:i1].max(0) > top[k]).any() for k, v in Fn.items())
        tot.setdefault(e.Code, []).append(triv)
n = sum(len(v) for v in tot.values()); k = sum(sum(v) for v in tot.values())
print({c: f'{sum(v)}/{len(v)}' for c, v in sorted(tot.items())}); print(f'trivial events: {k}/{n} = {100 * k / n:.0f} %')
