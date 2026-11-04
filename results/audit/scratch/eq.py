import sys, os, json, time, shutil, numpy as np, multiprocessing as mp
sys.path.insert(0, 'sim')
W = '.localenv/tmp/t'; DAYS = 6
def job(kind):
    os.nice(10)
    import config as C, physics
    from operations import Timeline
    from physics import SatSim
    physics.H_AUTO = 0.25                       # force many autonomous dumps so the new path is exercised
    fj = json.load(open('sim/fleet_v2b.json'))
    tl = Timeline('SAT_2', C.FLEET['SAT_2'], fj['seeds']['SAT_2']).build()
    t0 = time.time(); T = DAYS * 86400.0
    if kind == 'phys':
        m = SatSim(tl).run(f'{W}/phys', t_end=T, log=lambda s: None)
    elif kind == 'follow':
        m = SatSim(tl).run(f'{W}/follow', twin=True, t_end=T, follow=f'{W}/phys', log=lambda s: None)
    else:
        rp = json.load(open(f'{W}/phys/physlog.json'))['auto_dumps']
        m = SatSim(tl).run(f'{W}/replay', twin=True, replay=rp, t_end=T, log=lambda s: None)
    print(kind, 'wall %.1fs' % (time.time() - t0), 'auto dumps', len(m['auto_dumps']), flush=True)
if __name__ == '__main__':
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    a = mp.Process(target=job, args=('phys',)); b = mp.Process(target=job, args=('follow',))
    a.start(); time.sleep(2); b.start(); a.join(); b.join()
    job('replay')
    import physics
    A, B = physics.load_truth(f'{W}/follow'), physics.load_truth(f'{W}/replay')
    same = {k: bool(np.array_equal(A[k], B[k])) for k in A}
    print('follow == replay bit-for-bit:', all(same.values()), same)
    P = physics.load_truth(f'{W}/phys'); print('rows', len(A['t']), 'dumps', len(json.load(open(f'{W}/phys/physlog.json'))['auto_dumps']))
