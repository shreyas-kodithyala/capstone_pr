"""Past-only check (CLAUDE.md section 5). Recompute everything from data cut at row T (errors, trailing z,
masks, alarms) and compare with the full run: any difference before T - 70 rows is look-ahead.

  python causal_check.py <arm> <selection: local|federated> [out_dir]
"""
import json
import os
import sys

arm, sel = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    os.environ["FED_OUT"] = sys.argv[3]
sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import fedsim as F  # noqa: E402
from core.config import LATENCY, MODEL  # noqa: E402
from core.model import GDN2  # noqa: E402

R = json.load(open(f"{F.OUT}/results.json"))[arm]
rf = GDN2(20, 0, 12, **MODEL).rf
D = {s: F.inputs(s) for s in F.SATS}
PER = F.load_periods(D, arm, rf)
peers = [PER[s]["val"][0] for s in F.SATS]
out = {}
for s in F.SATS:
    cfg = R[s][sel]["cfg"]
    E = np.load(f"{F.OUT}/errors/{arm}_{s}.npy")
    full = F.alarms(PER[s]["test"][0], cfg)
    n = len(full)
    rows = []
    for frac in (0.3, 0.6, 0.85):
        cut = int(n * frac)
        P = F.periods(D[s], E[:D[s]["ntr"] + cut], rf)["test"][0]
        P.peers = peers
        P.ref = PER[s]["val"][0]
        A = F.alarms(P, cfg)
        safe = cut - LATENCY
        diff = int((A[:safe] != full[:safe]).sum())
        tail = int((A[safe:cut] != full[safe:cut]).sum())
        rows.append(dict(cut=cut, alarm_rows=int(full[:safe].any(1).sum()), diff_before_limit=diff, diff_in_last_70=tail))
    out[s] = dict(cfg=cfg, cuts=rows)
    print(s, rows, flush=True)
ok = all(c["diff_before_limit"] == 0 for v in out.values() for c in v["cuts"])
print("PAST-ONLY CHECK", "PASSED" if ok else "FAILED")
json.dump(dict(passed=ok, arm=arm, selection=sel, detail=out), open(f"{F.OUT}/causal_check_{arm}_{sel}.json", "w"), indent=1, default=str)
