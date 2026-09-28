import argparse, pathlib, subprocess, re, shutil, json, hashlib, time, statistics
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--fixture',default='/tmp/codex-k3-190k/base/ref/fixtures.safetensors');p.add_argument('--rounds',type=int,default=3);p.add_argument('--profile',default='info');p.add_argument('--only',default='decoder_feedforward');p.add_argument('--reverse',action='store_true');a=p.parse_args()
d=pathlib.Path('/tmp/codex-k3-190k/runs')/a.tag;d.mkdir(parents=True,exist_ok=False)
for name,src in [('bin_A',a.baseline),('bin_B',a.candidate),('fixtures.safetensors',a.fixture)]:shutil.copy2(src,d/name)
entry='''#!/bin/sh
cd "$(dirname "$0")" || exit 1
cp bin_A ./k3190_A.exec || exit 126
cp bin_B ./k3190_B.exec || exit 126
chmod +x ./k3190_A.exec ./k3190_B.exec || exit 126
'''
for r in range(1,a.rounds+1):
 for arm in ('BA' if a.reverse else 'AB'):
  entry+=f'echo "@@AB_BEGIN {r} {arm}@@"\nGEMMA4_FIXTURE="$PWD/fixtures.safetensors" GEMMA4_ONLY="{a.only}" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE={a.profile} '+('GEMMA4_DUMP_SPANS=1 ' if a.profile=='trace' else '')+f'./k3190_{arm}.exec\nrc=$?\necho "@@AB_END {r} {arm} rc=$rc@@"\n'
(d/'entry.sh').write_text(entry);(d/'entry.sh').chmod(0o755)
meta={'tag':a.tag,'rounds':a.rounds,'profile':a.profile,'only':a.only,'reverse':a.reverse,'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir() if f.is_file()}}
(d/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
cmd=['furiosa-arena','submit',str(d/'entry.sh'),str(d/'bin_A'),str(d/'bin_B'),str(d/'fixtures.safetensors'),'--name',a.tag,'--entrypoint','entry.sh','--timeout','70']
r=subprocess.run(cmd,text=True,capture_output=True);out=r.stdout+r.stderr;(d/'submit.log').write_text(out);print(out,flush=True)
m=re.search(r'submitted job (\d+)',out);assert r.returncode==0 and m,out
job=m[1];meta['job_id']=int(job);(d/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for _ in range(90):
 r=subprocess.run(['furiosa-arena','status',job],capture_output=True,text=True);out=r.stdout+r.stderr;(d/'status.log').write_text(out)
 if any(x in out for x in ['SUCCEEDED','FAILED','TIMEOUT','CANCELLED','CANCELED']):break
 time.sleep(10)
else: raise RuntimeError('observation timeout; do not resubmit, job '+job)
r=subprocess.run(['furiosa-arena','logs',job],capture_output=True,text=True);log=r.stdout+r.stderr;(d/'log.txt').write_text(log)
arm=None;kernel=None;runs={x:{} for x in 'AB'};ends=[];begins=[];errors=[]
for line in log.splitlines():
 m=re.fullmatch(r'@@AB_BEGIN (\d+) ([AB])@@',line)
 if m:arm=m[2];begins.append((int(m[1]),arm));continue
 m=re.fullmatch(r'@@AB_END (\d+) ([AB]) rc=(\d+)@@',line)
 if m:ends.append((int(m[1]),m[2],int(m[3])));arm=None;continue
 if arm is None:continue
 m=re.match(r'==>\s*(\S+)',line)
 if m:kernel=m[1]
 if 'FAIL' in line or 'panicked' in line:errors.append(line)
 m=re.search(r'\(of (\d+) runs: \[([\d, ]+)\]\)',line)
 if m and kernel:
  v=list(map(int,m[2].split(',')));assert len(v)==int(m[1]);runs[arm].setdefault(kernel,[]).extend(v)
if len(ends)!=2*a.rounds or [x[:2] for x in ends]!=begins or any(x[2] for x in ends):errors.append('incomplete/failed invocations')
summary={'job_id':int(job),'valid':not errors,'errors':errors,'samples':runs,'kernels':{}}
for k in sorted(set(runs['A'])|set(runs['B'])):
 med={x:statistics.median(runs[x][k]) for x in 'AB'};summary['kernels'][k]={'median_cycles':med,'B_minus_A':med['B']-med['A'],'n':{x:len(runs[x][k]) for x in 'AB'}}
(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='samples'},indent=2),flush=True)
if errors:raise SystemExit(1)
