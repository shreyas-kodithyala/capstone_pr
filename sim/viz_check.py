"""Decode a Vizard .bin file and compare it with the saved records. Vizard is not needed.

The message format is read from Basilisk's own libvizMessage.a.
  python viz_check.py ../viz/clips/<name>.bin
"""
import json
import mmap
import os
import resource
import sys
from datetime import datetime, timezone

import numpy as np
import Basilisk
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def varint(b, i):
    s = v = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 127) << s
        s += 7
        if c < 128:
            return v, i


def message_class():
    lib = open(os.path.join(os.path.dirname(Basilisk.__file__), "libvizMessage.a"), "rb").read()
    i = lib.find(b"\n\x10vizMessage.proto")
    j = i
    while True:                                    # walk the top-level fields of the file descriptor
        tag, k = varint(lib, j)
        field, wire = tag >> 3, tag & 7
        if wire != 2 or field == 0 or field > 20:
            break
        n, k = varint(lib, k)
        j = k + n
    fd = descriptor_pb2.FileDescriptorProto.FromString(lib[i:j])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("vizProtobufferMessage.VizMessage"))


def frames(path):
    """Stream frames from disk, so a large file never sits in memory."""
    M = message_class()
    with open(path, "rb") as fh:
        b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        i, n_all = 0, len(b)
        while i < n_all:
            n, i = varint(b, i)
            m = M()
            m.ParseFromString(b[i:i + n])
            i += n
            yield m
        b.close()


def verify(path, every=97):
    """Compare sampled frames with the saved records and the simulator's own geometry."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as C
    import viz_replay as V
    from envmodels import Orbit, gmst, mrp_to_dcm, rotz, shadow, sun_eci
    from operations import epoch_unix
    f = frames(path)
    first = next(f)
    ep = first.epoch
    t0 = (datetime(ep.year, ep.month, ep.day, ep.hours, ep.minutes, int(ep.seconds), tzinfo=timezone.utc) - V.EPOCH).total_seconds()
    names = [s.spacecraftName for s in first.spacecraft]
    popups, colours, n, rows = len(first.vizEventDialogs), set(), 1, []
    for i, m in enumerate(f, 1):
        n += 1
        popups += len(m.vizEventDialogs)
        for sc in m.spacecraft:
            colours.add(tuple(sc.trueTrajectoryLineColor))
        if i % every == 0:
            rows.append((t0 + m.currentTime.simTimeElapsed / 1e9, [list(sc.position) for sc in m.spacecraft], [list(sc.rotation) for sc in m.spacecraft],
                         [[w.wheelSpeed for w in sc.reactionWheels] for sc in m.spacecraft], list(m.celestialBodies[0].rotation), list(m.celestialBodies[1].position)))
    t = np.array([r[0] for r in rows])
    worst = dict(position_m=0.0, wheel_rad_s=0.0, earth_rotation=0.0, sun_m=0.0, eclipse=0.0)
    vertical = []
    tu = epoch_unix() + t
    for k, s in enumerate(names):
        R, W, have = np.zeros((len(t), 3)), np.zeros((len(t), 4)), np.zeros(len(t), bool)
        for fn in sorted(V.glob.glob(f"{V.WORK}/{s}/phys/truth_*.npz")):
            z = np.load(fn)
            tt = np.round(z["t"])
            j = np.minimum(np.searchsorted(tt, t), len(tt) - 1)
            ok = tt[j] == t
            if ok.any():
                R[ok], W[ok], have[ok] = z["r_BN"][j[ok]], z["Omega"][j[ok]], True
        assert have.all(), "frame time not in the saved record"
        o = Orbit(C.FLEET[s]["orbit"], epoch_unix())
        rN = np.array([r[1][k] for r in rows])
        sg = np.array([r[2][k] for r in rows])
        w = np.array([r[3][k] for r in rows])
        worst["position_m"] = max(worst["position_m"], float(np.abs(rotz(-o.d_omega(tu), rN) - R).max()))
        worst["wheel_rad_s"] = max(worst["wheel_rad_s"], float(np.abs(w - W).max()))
        sN, rau = sun_eci(tu)
        sP, _ = o.sun_P(tu)
        worst["eclipse"] = max(worst["eclipse"], float(np.abs(shadow(rN, sN, rau) - shadow(R, sP, rau)).max()))
        rb = np.einsum("nij,nj->ni", mrp_to_dcm(sg), rN / np.linalg.norm(rN, axis=1)[:, None])
        vertical.append(round(float(np.median(np.degrees(np.arccos(np.clip(rb[:, 0], -1, 1))))), 3))
    for i, r in enumerate(rows):
        g = float(gmst(tu[i]))
        true = np.array([[np.cos(g), np.sin(g), 0], [-np.sin(g), np.cos(g), 0], [0, 0, 1]])
        worst["earth_rotation"] = max(worst["earth_rotation"], float(np.abs(np.array(r[4]).reshape(3, 3) - true).max()))
        su, ra = sun_eci(tu[i])
        worst["sun_m"] = max(worst["sun_m"], float(np.abs(np.array(r[5]) - np.asarray(su) * float(ra) * 1.495978707e11).max()))
    return dict(file=os.path.basename(path), size_mb=round(os.path.getsize(path) / 1e6, 1), frames=n, satellites=names, first_day=round(t0 / 86400, 3),
                last_sampled_day=round(float(t[-1]) / 86400, 3), popups=popups, status_colours=len(colours), sampled=len(t), worst_difference=worst,
                median_deg_body_x_to_vertical=vertical)


if __name__ == "__main__":
    out = []
    for p_ in sys.argv[1:]:
        r = verify(p_)
        r["peak_memory_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz", "verification.json"), "w"), indent=1)
