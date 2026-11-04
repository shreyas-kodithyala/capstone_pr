import sys, time, os
sys.argv, args = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, 'fed')
os.environ.setdefault('FED_OUT', 'fed/out_v2')
import torch, fedsim as F
sat, devn, n, th = args[0], args[1], int(args[2]), int(args[3])
torch.set_num_threads(th); dev = torch.device(devn)
d = F.inputs(sat); m = F.new_model(d, dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
F.local_steps(m, opt, d, 5, 0, dev)
t = time.time(); F.local_steps(m, opt, d, n, 1, dev)
if devn == 'mps': torch.mps.synchronize()
print(f'{sat} {devn} th={th}: {(time.time()-t)/n:.3f} s/batch', flush=True)
