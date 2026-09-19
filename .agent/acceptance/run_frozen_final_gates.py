"""Run final CI gates once against an already verified frozen candidate source."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

source=Path('/Users/maxwell/mySpace/myQuant')
root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
repository=root/'repository'
assert receipt['runtime_verification']['state']=='PASS'
assert subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()==receipt['commit']
output=root/'final-gates'
output.mkdir(mode=0o700,exist_ok=False)
files=[p for d in ('quant_investor','scripts','tests','portfolio_dashboard') for p in (repository/d).rglob('*') if p.is_file() and p.suffix in {'.py','.js','.html','.css','.json'} and '__pycache__' not in p.parts]
files.extend(repository/p for p in ('pyproject.toml','uv.lock','.github/workflows/ci-cd.yml'))
inventory={str(p.relative_to(repository)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(output/'source-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
for relative,sha in inventory.items():
 if relative.startswith(('quant_investor/','scripts/')) and relative.endswith('.py'):
  assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==sha,relative
python=str(source/'.venv/bin/python'); node=shutil.which('node'); assert node
scope=['quant_investor/contracts','quant_investor/system','quant_investor/factors/governance','quant_investor/intelligence','quant_investor/mainline','quant_investor/cli']
commands=[
 ('access',[python,'scripts/check_strategy_record_access.py']),
 ('fatal-lint',[python,'-m','flake8','quant_investor','--count','--select=E9,F63,F7,F82','--show-source','--statistics']),
 ('stable-lint',[python,'-m','flake8',*scope,'--max-complexity=10','--max-line-length=100']),
 ('stable-black',[python,'-m','black','--check',*scope]),
 ('stable-mypy',[python,'-m','mypy',*scope,'--ignore-missing-imports']),
 ('dashboard-v1',[node,'portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v1.test.js']),
 ('dashboard-v2',[node,'portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v2.test.js']),
 ('dashboard-daily',[node,'portfolio_dashboard/tests/cn_daily_dashboard_contract_v1.test.js']),
 *[('syntax-'+Path(p).stem,[node,'--check',p]) for p in ('portfolio_dashboard/app.js','portfolio_dashboard/js/cn_aggressive_input.js','portfolio_dashboard/js/cn_aggressive_dashboard_contract_v1.js')],
 ('unit',[python,'-m','pytest','tests/unit','-q','-ra']),
]
environment=dict(os.environ); environment['PYTHONPATH']=str(repository);environment['PYTHONDONTWRITEBYTECODE']='1'
origin=subprocess.check_output([python,'-c','import quant_investor;print(quant_investor.__file__)'],cwd=repository,env=environment,text=True).strip()
assert Path(origin).resolve()==(repository/'quant_investor/__init__.py').resolve()
state={'state':'RUNNING','started_at':datetime.now(timezone.utc).isoformat(),'root':str(root),'runtime_commit':receipt['commit'],'source_import_origin':origin,'scope':'FROZEN_SOURCE_CI_EQUIVALENT','unit_includes_unified_subset':True,'installed_runtime_proof_separate':True,'steps':[],'automatic_retry':False,'production_deployed':False}
status=output/'status.json'
for label,command in commands:
 state['active_step']=label
 started=time.monotonic()
 with (output/(label+'.log')).open('w') as stream:
  child=subprocess.Popen(command,cwd=repository,env=environment,stdout=stream,stderr=subprocess.STDOUT)
  state['child_pid']=child.pid;status.write_text(json.dumps(state,indent=2)+'\n')
  print('START',label,'pid',child.pid,flush=True)
  code=child.wait()
 state['steps'].append({'step':label,'exit_code':code,'seconds':round(time.monotonic()-started,2),'command':command,'log':str(output/(label+'.log'))})
 print('EXIT',label,code,flush=True)
 if code:
  state['state']='FAIL';break
else:state['state']='PASS'
state.pop('active_step',None);state.pop('child_pid',None)
state['source_drift']=[name for name,sha in inventory.items() if not (repository/name).is_file() or hashlib.sha256((repository/name).read_bytes()).hexdigest()!=sha]
if state['source_drift']:state['state']='FAIL'
state['finished_at']=datetime.now(timezone.utc).isoformat();status.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
raise SystemExit(0 if state['state']=='PASS' else 1)
