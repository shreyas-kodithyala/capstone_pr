"""Remaining useful life of a reaction wheel from telemetry alone (C-MAPSS-style method).

Health indicator (no labels, no truth): for each wheel k and day, r = median |current_k| / median
|current| of the other three wheels (friction adds drive current on the worn wheel only).
HI = 7-day median of r / r_ref - 1 (excess current fraction; physical units, so archive wheels and
SAT_2 are comparable). Alerting uses the z-score of r against the satellite's reference period.

Thresholds (fixed before looking at SAT_2's test half):
  alert     HI > mean + 6 sd of the daily HI over the reference period, 3 days running
  failure   the highest HI any archived wheel (ARC_1..3, same design, run to failure) reached on its
            last day before failing. Rule chosen by leave-one-out PHM08 on the archive wheels only
            (max 0.152, mean 0.174, min 0.227 per prediction); the first rule, the lowest HI, is
            still reported as exp_min
RUL at day t (after the alert): fit log(HI) ~ a + b t over the last 30 days (exponential wear
path) and solve for the failure threshold; baseline arm: the same with a straight line on HI.
Scores vs truth/health.csv: alert lead time, RMSE / MAE over the last 60 days, PHM08 score
(late predictions penalised more), alpha-lambda accuracy (within +-20 % of true RUL).

  conda activate timeeval;  python rul.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.environ.get("SIM_FLEET") or os.path.join(os.path.dirname(HERE), "data", "sim_fleet")
OUT = os.environ.get("FED_OUT", os.path.join(HERE, "out_v2"))
REF_DAYS = dict(SAT=(0, 137), ARC=(0, 5))      # healthy reference: SAT = training rows before validation


def daily_ratio(raw):
    ch = pd.read_csv(f"{raw}/channels.csv")
    cur = {r.Name: r.Channel for r in ch.itertuples() if r.Name.endswith("_current")}
    parts = [pd.read_csv(f"{raw}/resampled_30s_{p}.csv.gz", usecols=["timestamp"] + list(cur.values()))
             for p in ("train", "test") if os.path.exists(f"{raw}/resampled_30s_{p}.csv.gz")]
    df = pd.concat(parts, ignore_index=True)
    df["day"] = (pd.to_datetime(df.timestamp) - pd.to_datetime(df.timestamp.iloc[0])).dt.days
    A = df[list(cur.values())].abs()
    A["day"] = df.day
    med = A.groupby("day").median()
    out = {}
    names = list(cur)
    for i, n in enumerate(names):
        others = med[[cur[m] for m in names if m != n]].median(1)
        out[n] = med[cur[n]] / others.clip(lower=1e-6)
    return pd.DataFrame(out)


def hi(ratio, ref):
    """-> (HI = 7-day median excess current fraction vs the wheel's own reference level (physical,
    comparable across wheels of one design), daily z-score vs the reference (alerting), ref stats)"""
    r = ratio.iloc[ref[0]:ref[1]]
    z = (ratio - r.mean()) / (r.std() + 1e-9)
    ex = ratio / r.median() - 1.0
    return ex.rolling(7, min_periods=3).median(), z, (r.mean(), r.std())


def fit_rul(h, t, th_fail, kind):
    """h: HI over the fit window (days t). Returns predicted days until HI reaches th_fail."""
    ok = h > 0.5 if kind == "exp" else np.isfinite(h)
    if ok.sum() < 5:
        return np.nan
    y = np.log(h[ok]) if kind == "exp" else h[ok]
    b, a = np.polyfit(t[ok], y, 1)
    if b <= 0:
        return np.inf
    target = np.log(th_fail) if kind == "exp" else th_fail
    return max((target - a) / b - t[-1], 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    # failure threshold from the archive wheels
    arc = {}
    for a in ("ARC_1", "ARC_2", "ARC_3"):
        raw = f"{FLEET}/archive/{a}"
        wj = json.load(open(f"{raw}/truth/wear.json"))
        H, _, _ = hi(daily_ratio(raw), REF_DAYS["ARC"])
        wn = f"rw{wj['wheel'] + 1}_current"
        fday = int(wj["t_fail"] // 86400)
        arc[a] = dict(wheel=wn, fail_day=fday, hi_last=float(H[wn].iloc[fday - 1]))
    th_fail = max(v["hi_last"] for v in arc.values())
    th_min = min(v["hi_last"] for v in arc.values())
    # SAT_2
    raw = f"{FLEET}/SAT_2"
    H, Z, (m, sd) = hi(daily_ratio(raw), REF_DAYS["SAT"])
    zr = Z.iloc[REF_DAYS["SAT"][0]:REF_DAYS["SAT"][1]]
    th_alert = 6.0                         # z units (mean + 6 sd of the reference period)
    Zs = Z.rolling(7, min_periods=3).median()
    wj = json.load(open(f"{raw}/truth/wear.json"))
    fday = wj["t_fail"] / 86400
    days = np.arange(len(H))
    above = (Zs > th_alert).astype(int)
    run3 = above.rolling(3).sum() >= 3
    first = {w: (int(np.argmax(run3[w].values)) if run3[w].any() else None) for w in H.columns}
    alert_wheel = min((d, w) for w, d in first.items() if d is not None) if any(v is not None for v in first.values()) else (None, None)
    a_day, w = alert_wheel
    res = dict(failure_threshold=th_fail, failure_threshold_min=th_min, archive=arc, alert_threshold=th_alert, first_alert_day=a_day,
               alert_wheel=w, true_wheel=f"rw{wj['wheel'] + 1}_current", true_fail_day=fday,
               lead_time_days=(fday - a_day) if a_day is not None else None,
               true_phase2_visible_day=(wj.get("t_phase2_visible") or 0) / 86400)
    rows = []
    if a_day is not None:
        h = H[w].values
        for t in range(a_day, int(fday)):
            lo = max(a_day - 10, t - 30)
            tt = days[lo:t + 1]
            hh = h[lo:t + 1]
            true = fday - t
            for kind, th in (("exp", th_fail), ("lin", th_fail), ("exp_min", th_min)):
                p = fit_rul(hh, tt, th, kind[:3])
                rows.append(dict(day=t, kind=kind, true_rul=true, pred_rul=p))
        R = pd.DataFrame(rows)
        R.to_csv(f"{OUT}/rul_predictions.csv", index=False)
        for kind in ("exp", "lin", "exp_min"):
            r = R[(R.kind == kind) & (R.true_rul <= 60) & np.isfinite(R.pred_rul)]
            e = r.pred_rul - r.true_rul
            phm = float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))
            res[kind] = dict(n=int(len(r)), rmse=float(np.sqrt((e ** 2).mean())), mae=float(e.abs().mean()),
                             phm08=phm, alpha_lambda_20=float((e.abs() <= 0.2 * r.true_rul).mean()),
                             late_frac=float((e > 0).mean()))
        # how early the date can be trusted: from here to failure every prediction is within +-20 %
        r = R[(R.kind == "exp")].sort_values("day")
        bad = ~((r.pred_rul - r.true_rul).abs() <= 0.2 * r.true_rul)
        tail = r.day[bad].max() + 1 if bad.any() else r.day.min()
        res["reliable_from_day"] = int(tail) if tail < fday else None
        res["reliable_lead_days"] = float(fday - tail) if tail < fday else None
        ms = {}
        for k in (90, 60, 30, 14, 7):
            q = r[(r.true_rul - k).abs() <= 0.5]
            if len(q):
                ms[k] = dict(pred=float(q.pred_rul.iloc[0]), err=float(q.pred_rul.iloc[0] - q.true_rul.iloc[0]))
        res["milestones"] = ms
        # figure
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for c in H.columns:
            ax[0].plot(days, H[c], lw=1.5 if c == w else 0.7, label=c)
        ax[0].axvline(a_day, ls="--", c="orange", label="alert (z > 6, 3 days)")
        ax[0].axhline(th_fail, ls="--", c="red", label="failure threshold (archive)")
        ax[0].axvline(fday, c="k", lw=0.8)
        ax[0].axvline(a_day, c="orange", lw=0.8)
        ax[0].set_yscale("symlog")
        ax[0].set_ylabel("excess current fraction")
        ax[0].legend(fontsize=7)
        for kind, c in (("exp", "C0"), ("lin", "C3")):
            r = R[R.kind == kind]
            ax[1].plot(r.day, r.pred_rul.clip(upper=200), c=c, label=f"predicted RUL ({kind})")
        r = R[R.kind == "exp"]
        ax[1].plot(r.day, r.true_rul, "k--", label="true RUL")
        ax[1].fill_between(r.day, 0.8 * r.true_rul, 1.2 * r.true_rul, color="grey", alpha=0.2, label="+-20 %")
        ax[1].set_ylabel("days")
        ax[1].set_xlabel("mission day")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{OUT}/rul_SAT_2.png", dpi=120)
    json.dump(res, open(f"{OUT}/rul.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
