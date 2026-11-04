import sys,fitz
d=fitz.open(sys.argv[1]); t="\n\f".join(p.get_text() for p in d); open(sys.argv[2],"w").write(t); print(len(d),len(t))
