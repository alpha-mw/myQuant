"""Run separately pinned integration helpers against the current frozen installation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE=Path('/Users/maxwell/mySpace/myQuant')
ROOT=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((ROOT/'fixture-receipt.json').read_bytes())
if (ROOT/'configured-driver-status.json').exists() or (ROOT/'factor-workspace').exists():
 raise ValueError('native scenario already started; inspect exact retained state')
repository=ROOT/'repository'
runtime=[p for d in ('quant_investor','scripts') for p in (SOURCE/d).rglob('*.py') if '__pycache__' not in p.parts]
differences=[str(p.relative_to(SOURCE)) for p in runtime if not (repository/p.relative_to(SOURCE)).is_file() or p.read_bytes()!=(repository/p.relative_to(SOURCE)).read_bytes()]
if differences: raise ValueError('current runtime differs from frozen install: '+repr(differences))
directory=ROOT/'validation-helpers'
directory.mkdir(mode=0o700)
names=['_native_full_dag_scenario.py','_native_daily_maintenance_fixture.py','_native_synthetic_clock.py','_native_configured_daily_scenario.py']
pins={}
for name in names:
 raw=(SOURCE/'tests/unit'/name).read_bytes()
 target=directory/name
 target.write_bytes(raw)
 target.chmod(0o600)
 pins[name]=hashlib.sha256(raw).hexdigest()
(ROOT/'validation-helper-manifest.json').write_text(json.dumps({'helper_sha256':pins,'runtime_files_compared':len(runtime),'runtime_differences':differences,'runtime_commit':receipt['commit']},indent=2)+'\n')
state={'state':'RUNNING','root':str(ROOT),'producer_commit':receipt['commit'],'started_at':datetime.now(timezone.utc).isoformat(),'production_deployed':False}
status=ROOT/'configured-driver-status.json'
program='''import runpy,socket,sys,os,signal,faulthandler,json
from pathlib import Path
root=Path(sys.argv[3])
trace=(root/'configured-native.stack.txt').open('a')
os.chmod(trace.name,0o600)
faulthandler.register(signal.SIGUSR1,file=trace,all_threads=True)
(root/'configured-native-trace-ready.json').write_text(json.dumps({'pid':os.getpid(),'signal':'SIGUSR1','trace':trace.name})+chr(10))
original=socket.socket.connect
def guard(sock,address):
 if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
  raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
 return original(sock,address)
socket.socket.connect=guard
sys.path.insert(0,sys.argv[1])
script=sys.argv[2]
sys.argv=sys.argv[2:]
runpy.run_path(script,run_name='__main__')
'''
env=dict(os.environ)
env.pop('PYTHONPATH',None)
with (ROOT/'configured-native.log').open('w') as stream:
 child=subprocess.Popen([receipt['python'],'-I','-c',program,str(directory),str(directory/names[-1]),str(ROOT),str(SOURCE/'.venv/lib/python3.13/site-packages')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 state['child_pid']=child.pid
 status.write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps(state),flush=True)
 code=child.wait()
state.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
state.pop('child_pid',None)
state['helper_drift']=[name for name,digest in pins.items() if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest]
if state['helper_drift']: state['state']='FAIL'
status.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2),flush=True)
raise SystemExit(0 if state['state']=='PASS' else 1)
