import sys,pypdf
for p in sys.argv[1:]:
    r=pypdf.PdfReader(p); t="\n\f".join((x.extract_text() or "") for x in r.pages)
    open(p[:-4]+".txt","w").write(t); print(p,len(r.pages),len(t))
