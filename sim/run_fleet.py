"""Generate the whole fleet dataset with one command.

  conda activate bsk
  python run_fleet.py                     # all 5 satellites + 3 archived wheels -> ../data/sim_fleet
  python run_fleet.py --only SAT_2        # one satellite
  python run_fleet.py --workers 6         # physics processes in parallel (default 7)

Freeze: the first run draws fresh random seeds and records them, the code checksum (SHA-256 of
every sim/*.py) and the config in data/sim_fleet/fleet.json. Re-running with that fleet.json
reproduces the dataset bit-for-bit.

Resources: every worker runs at nice 10, one thread; physics workers use ~0.3 GB each, products
workers ~3 GB each (2 at a time). A memory guard pauses new jobs if free memory drops below 2.5 GB.
"""
import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "sim_fleet")
WORK = os.path.join(ROOT, "data", "sim_work")
sys.path.insert(0, HERE)


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        os.nice(10)
    except OSError:
        pass


def phys_job(name, spec, seed, twin=False, replay=None):
    _init()
    import config as C  # noqa
    from operations import Timeline
    from physics import SatSim
    tl = Timeline(name, spec, seed).build()
    d = os.path.join(WORK, name, "twin" if twin else "phys")
    if os.path.exists(f"{d}/physlog.json"):
        return json.load(open(f"{d}/physlog.json"))
    shutil.rmtree(d, ignore_errors=True)
    logf = open(os.path.join(OUT, "logs", f"{name}{'_twin' if twin else ''}.log"), "a")

    def log(s):
        logf.write(time.strftime("%H:%M:%S ") + s + "\n"); logf.flush()
    meta = SatSim(tl).run(d, twin=twin, replay=replay, log=log)
    log(f"done in {meta['wall_min']:.1f} min, auto dumps {len(meta['auto_dumps'])}")
    return meta


def prod_job(name, spec, seed, has_twin, out_root):
    _init()
    import products
    d = os.path.join(WORK, name)
    logf = open(os.path.join(OUT, "logs", f"{name}_products.log"), "a")

    def log(s):
        logf.write(time.strftime("%H:%M:%S ") + s + "\n"); logf.flush()
    t0 = time.time()
    ev = products.make(name, spec, seed, f"{d}/phys", f"{d}/twin" if has_twin else None, out_root, log=log)
    log(f"done in {(time.time() - t0) / 60:.1f} min, {len(ev)} events")
    return len(ev)


def free_gb():
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        pg = 16384
        f = {l.split(":")[0]: int(l.split(":")[1].strip().rstrip(".")) for l in vm.splitlines()[1:] if ":" in l}
        return (f.get("Pages free", 0) + f.get("Pages inactive", 0) + f.get("Pages speculative", 0)) * pg / 1e9
    except Exception:
        return 99.0


def code_hash():
    h = hashlib.sha256()
    for f in ("config.py", "envmodels.py", "operations.py", "physics.py", "telemetry.py", "products.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--prod-workers", type=int, default=2)
    a = ap.parse_args()
    import config as C
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    fj = os.path.join(OUT, "fleet.json")
    sats = {**C.FLEET, **C.ARCHIVE}
    if os.path.exists(fj):
        fleet = json.load(open(fj))
        if fleet["code_sha256"] != code_hash():
            print("WARNING: code changed since the seeds were frozen (fleet.json). Continuing with the recorded seeds.")
    else:
        fleet = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), code_sha256=code_hash(),
                     seeds={n: secrets.randbelow(10 ** 8) for n in sats},
                     epoch=C.EPOCH, days=C.DAYS, split_day=C.SPLIT_DAY, val_days=C.VAL_DAYS, grid_s=C.GRID,
                     satellites={n: s for n, s in sats.items()})
        json.dump(fleet, open(fj, "w"), indent=1)
    names = [n for n in sats if not a.only or n in a.only.split(",")]
    twins = {n for n in names if C.FLEET.get(n, {}).get("profile") in ("healthy", "wheel", "power", "ops")
             or n in C.ARCHIVE}
    t0 = time.time()

    def say(s):
        line = f"[{(time.time() - t0) / 60:6.1f} min] {s}"
        print(line, flush=True)
        open(os.path.join(OUT, "logs", "run.log"), "a").write(line + "\n")
    say(f"satellites {names}; twins {sorted(twins)}; physics workers {a.workers}")
    pp = ProcessPoolExecutor(a.workers, initializer=_init)
    qp = ProcessPoolExecutor(a.prod_workers, initializer=_init)
    pending = {}
    for n in sorted(names, key=lambda n: -sats[n].get("days", C.DAYS)):
        pending[pp.submit(phys_job, n, sats[n], fleet["seeds"][n])] = ("phys", n)
    done_phys, done_twin, prods = set(), set(), {}
    while pending:
        fin, _ = wait(list(pending), return_when=FIRST_COMPLETED, timeout=60)
        for f in fin:
            kind, n = pending.pop(f)
            try:
                res = f.result()
            except Exception as e:
                say(f"FAILED {kind} {n}: {e!r}")
                continue
            say(f"{kind} {n} finished")
            if kind == "phys":
                done_phys.add(n)
                if n in twins:
                    pending[pp.submit(phys_job, n, sats[n], fleet["seeds"][n], True, res["auto_dumps"])] = ("twin", n)
            elif kind == "twin":
                done_twin.add(n)
            if kind in ("phys", "twin") and n in done_phys and (n not in twins or n in done_twin):
                while free_gb() < 2.5:
                    say(f"memory guard: {free_gb():.1f} GB free, waiting"); time.sleep(30)
                root = OUT if n in C.FLEET else os.path.join(OUT, "archive")
                pending[qp.submit(prod_job, n, sats[n], fleet["seeds"][n], n in twins, root)] = ("products", n)
        if not fin:
            prog = []
            for n in names:
                for sub in ("phys", "twin"):
                    p = os.path.join(WORK, n, sub, "progress.json")
                    if os.path.exists(p):
                        g = json.load(open(p))
                        prog.append(f"{n}{'-t' if sub == 'twin' else ''}:{g['day']}/{g['of']}")
            say("progress " + " ".join(prog) + f" | free mem {free_gb():.1f} GB")
    pp.shutdown(); qp.shutdown()
    say("all done")


if __name__ == "__main__":
    main()
