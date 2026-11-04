"""Validation part 2: difficulty / triviality check (Wu & Keogh 2023; ESA-ADB Table 11).

Runs ESA-ADB's simple baselines on every satellite's test split and scores them with ESA-ADB's
own metric code (vendored TimeEval ESA metrics, validated to 1e-12 against the official harness):
  * Out-of-limit   per channel, nominal train [min, max] widened by 10 % of the range (what
                   operations use today)
  * GlobalSTD3/5   per channel, |x - mean| > k std of nominal train data (ESA-ADB baseline)
  * iForest        multivariate Isolation Forest on the target channels, threshold at the train
                   annotated fraction, one global alarm (ESA-ADB unsupervised baseline)

  conda activate timeeval
  python baselines.py   -> data/sim_fleet/validation/baselines.json
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLEET = os.path.join(ROOT, "data", "sim_fleet")
NEW_RUNS = (os.environ.get("NEW_RUNS") or (os.path.join(ROOT, "ext", "new_runs") if os.path.isdir(os.path.join(ROOT, "ext", "new_runs")) else os.path.join(os.path.dirname(ROOT), "ESA", "Main", "new_runs")))
sys.path.insert(0, NEW_RUNS)
from core.esa import ESAScores, ChannelAwareFScore, ADTQC, BETA, series, build_labels_df, subsystems_mapping  # noqa: E402

SEL = {"Category": ["Rare Event", "Anomaly"]}
SEL_ANOM = {"Category": ["Anomaly"]}
SATS = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]


def score(raw, ts, P, chs, sel):
    ts = pd.to_datetime(pd.Series(ts)).dt.tz_localize(None).reset_index(drop=True)
    lab = build_labels_df(raw, ts)
    y = lab[lab["Channel"].isin(chs)].reset_index(drop=True)
    if not y.Category.isin(sel["Category"]).any():
        return {}
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for k, v in ESAScores(betas=BETA, select_labels=sel).score(y.drop(columns=["Channel"]), series(ts, P.max(1))).items():
            out[f"global_{k}"] = float(v)
        if P.shape[1] > 1 and not (P == P[:, :1]).all():
            yp = {c: series(ts, P[:, i]) for i, c in enumerate(chs)}
            try:
                for k, v in ChannelAwareFScore(beta=BETA, select_labels=sel).score(
                        y.copy(), {c: d.copy() for c, d in yp.items()}, subsystems_mapping(raw, chs)).items():
                    out[f"chan_{k}"] = float(v)
            except Exception as e:
                out["chan_error"] = repr(e)
            try:
                for k, v in ADTQC(select_labels=sel).score(y.copy(), {c: d.copy() for c, d in yp.items()}).items():
                    out[f"adtqc_{k}"] = float(v)
            except Exception as e:
                out["adtqc_error"] = repr(e)
    return out


def main():
    res = {}
    for sat in SATS:
        raw = f"{FLEET}/{sat}"
        ch = pd.read_csv(f"{raw}/channels.csv")
        tg = ch.Channel[ch.Target == "YES"].tolist()
        tr = pd.read_csv(f"{raw}/resampled_30s_train.csv.gz")
        te = pd.read_csv(f"{raw}/resampled_30s_test.csv.gz")
        Xtr, Xte = tr[tg].values.astype(float), te[tg].values.astype(float)
        Ltr = tr[[f"is_anomaly_{c}" for c in tg]].values
        nom = Ltr == 0
        r = {}
        # out-of-limit
        lo = np.array([np.nanmin(Xtr[nom[:, j], j]) for j in range(len(tg))])
        hi = np.array([np.nanmax(Xtr[nom[:, j], j]) for j in range(len(tg))])
        m = 0.1 * (hi - lo)
        P = ((Xte < lo - m) | (Xte > hi + m)).astype(np.uint8)
        r["out_of_limit"] = {"all": score(raw, te.timestamp, P, tg, SEL), "anomalies": score(raw, te.timestamp, P, tg, SEL_ANOM)}
        # GlobalSTD
        mu = np.array([Xtr[nom[:, j], j].mean() for j in range(len(tg))])
        sd = np.array([Xtr[nom[:, j], j].std() for j in range(len(tg))]) + 1e-12
        for k in (3, 5):
            P = (np.abs(Xte - mu) > k * sd).astype(np.uint8)
            r[f"GlobalSTD{k}"] = {"all": score(raw, te.timestamp, P, tg, SEL), "anomalies": score(raw, te.timestamp, P, tg, SEL_ANOM)}
        # iForest (unsupervised, global alarm)
        try:
            from sklearn.ensemble import IsolationForest
            rng = np.random.default_rng(0)
            Z = (Xtr - mu) / sd
            idx = rng.choice(len(Z), min(100000, len(Z)), replace=False)
            f = IsolationForest(n_estimators=100, random_state=42).fit(Z[idx])
            s_tr = -f.score_samples(Z[idx])
            cont = max((Ltr > 0).any(1).mean(), 1e-3)
            th = np.quantile(s_tr, 1 - cont)
            s_te = -f.score_samples((Xte - mu) / sd)
            g = (s_te > th).astype(np.uint8)
            P = np.repeat(g[:, None], len(tg), 1)
            r["iForest"] = {"all": score(raw, te.timestamp, P, tg, SEL), "anomalies": score(raw, te.timestamp, P, tg, SEL_ANOM)}
        except Exception as e:
            r["iForest"] = {"error": repr(e)}
        res[sat] = r
        print(sat, {k: round(v["all"].get("global_EW_F_0.50", float("nan")), 3) for k, v in r.items() if "all" in v}, flush=True)
    os.makedirs(f"{FLEET}/validation", exist_ok=True)
    json.dump(res, open(f"{FLEET}/validation/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
