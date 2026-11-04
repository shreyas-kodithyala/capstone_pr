# catch22 of simulated windows (bsk env, same sampling as validate.py); saved for the classifier test
import sys, numpy as np, pandas as pd, pycatch22
sys.path.insert(0, 'sim'); import config as C
rng = np.random.default_rng(1); win = 2880
tg = [C.CH_ID[c[0]] for c in C.CHANNELS if c[5]]; names = {C.CH_ID[c[0]]: c[0] for c in C.CHANNELS}
F, G = [], []
for s in ('SAT_1', 'SAT_2', 'SAT_3', 'SAT_4'):
    df = pd.concat([pd.read_csv(f'data/sim_fleet/{s}/resampled_30s_{p}.csv.gz', usecols=tg) for p in ('train', 'test')], ignore_index=True)
    for _ in range(150):
        c = tg[rng.integers(len(tg))]; i = rng.integers(0, len(df) - win); x = df[c].values[i:i + win].astype(float)
        if np.nanstd(x) < 1e-12: continue
        F.append(pycatch22.catch22_all(x.tolist())['values']); G.append(names[c])
np.save('.localenv/tmp/sim_c22.npy', np.array(F)); pd.Series(G).to_csv('.localenv/tmp/sim_c22_ch.csv', index=False)
print('windows', len(F), 'names', pycatch22.catch22_all(list(range(100)))['names'][:3])
