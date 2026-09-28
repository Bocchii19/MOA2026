import sys,re
path,test=sys.argv[1],sys.argv[2]
sp=[]
for ln in open(path):
    m=re.match(r"SPAN (\S+) cl=(\d+) beg=(\d+) end=(\d+) dur=(\d+) name=(.*)",ln.strip())
    if m and m.group(1)==test and m.group(2)=="0":
        sp.append((int(m.group(3)),int(m.group(4)),m.group(6)))
end=max(e for b,e,n in sp)
def busy(kind):
    a=bytearray(end+1)
    for b,e,n in sp:
        if kind(n):
            for t in range(b,e): a[t]=1
    return a
dma=busy(lambda n:n=="DMA")
tu=busy(lambda n:n.startswith("Renegade::"))
both=sum(1 for t in range(end) if dma[t] and tu[t])
donly=sum(1 for t in range(end) if dma[t] and not tu[t])
tonly=sum(1 for t in range(end) if tu[t] and not dma[t])
idle=end-both-donly-tonly
print(f"{test}: window {end}")
for k,v in (("DMA only",donly),("compute only",tonly),("both",both),("neither",idle)):
    print(f"   {k:13} {v:7}  {100*v/end:5.1f}%")
# phases: contiguous runs of each state longer than 1500
state=lambda t:("D" if dma[t] else "")+("C" if tu[t] else "")
cur=state(0); st=0; runs=[]
for t in range(1,end):
    s=state(t)
    if s!=cur:
        runs.append((st,t,cur)); cur=s; st=t
runs.append((st,end,cur))
big=[r for r in runs if r[1]-r[0]>=1200 and r[2]!="DC"]
print("   longest single-engine / idle stretches:")
for b,e,s in sorted(big,key=lambda r:r[0]-r[1])[:10]:
    lbl={"D":"DMA only","C":"compute only","":"idle"}[s]
    print(f"     {b:7}->{e:7} {e-b:6}  {lbl}")
