import glob, hashlib, json, os, sys, tarfile, time
R='.localenv/release'
parts=sorted(glob.glob(R+'/capstone_data.tar.gz.part-*'))
class Cat:
    def __init__(s,ps): s.ps=list(ps); s.f=open(s.ps.pop(0),'rb')
    def read(s,n=-1):
        out=b''
        while n<0 or len(out)<n:
            b=s.f.read(n-len(out) if n>=0 else 1<<20)
            if b: out+=b; continue
            if not s.ps: break
            s.f.close(); s.f=open(s.ps.pop(0),'rb')
        return out
def sha(fobj):
    h=hashlib.sha256()
    while True:
        b=fobj.read(1<<22)
        if not b: return h.hexdigest()
        h.update(b)
t0=time.time(); seen=set(); bad=[]; n=0; byts=0
with tarfile.open(fileobj=Cat(parts),mode='r|gz') as t:
    for m in t:
        if not m.isfile(): continue
        seen.add(m.name); n+=1; byts+=m.size
        a=sha(t.extractfile(m))
        if not os.path.exists(m.name): bad.append(('missing_on_disk',m.name)); continue
        with open(m.name,'rb') as f: b=sha(f)
        if a!=b: bad.append(('differs',m.name))
        if n%200==0: print(n,round(byts/1e9,2),'GB',int(time.time()-t0),'s',flush=True)
disk=set()
for root in ['data','fed/out_final','fed/out_v2','fed/out_w60']:
    for d,_,fs in os.walk(root):
        for x in fs: disk.add(os.path.join(d,x))
miss=sorted(disk-seen)
print(json.dumps(dict(parts=len(parts),files_in_archive=n,bytes=byts,files_on_disk=len(disk),not_in_archive=miss[:50],n_not_in_archive=len(miss),bad=bad[:50],n_bad=len(bad),secs=int(time.time()-t0)),indent=1))
