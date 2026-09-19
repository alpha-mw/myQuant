"""Real installed Calendar component, explicit offline HTTPS/guard seam, no EOD claim."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
assert receipt['runtime_verification']['state']=='PASS'
helpers=root/'production-calendar-offline-helpers'
helpers.mkdir(mode=0o700,exist_ok=False)
shutil.copytree(root/'repository/tests',helpers/'tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
program=r'''import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]);helpers=Path(sys.argv[2])
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
import quant_investor
assert str(Path(quant_investor.__file__).resolve())==receipt['runtime_verification']['import_origin']
sys.path.insert(0,str(helpers/'tests/unit'));sys.path.insert(1,str(helpers));sys.path.append(sys.argv[3])
import pytest
from _native_production_calendar_fixture import offline_https
from quant_investor.market.future_calendar_producer import new_future_state,bind_future_calendar
from quant_investor.market.future_calendar_context import PRODUCTION_CONTEXT_SCHEMA
from quant_investor.market.next_session_proof import read_next_session_proof
from quant_investor.contracts import canonical_json_bytes
workspace=root/'production-calendar-offline-workspace';workspace.mkdir(mode=0o700,exist_ok=False)
raw=(root/'release-input.json').read_bytes();path=workspace/'release.json';path.write_bytes(raw);path.chmod(0o600)
ref={'path':path.name,'sha256':hashlib.sha256(raw).hexdigest()}
context={'schema_version':PRODUCTION_CONTEXT_SCHEMA,'release_install_input_ref':ref,
 'release_repository_root':receipt['repository'],'release_commit':receipt['commit'],
 'calendar_capture_parent':str(workspace/'baseline-calendar'),'initial_calendar_receipt_ref':None,
 'next_session_calendar_mode':'PRODUCTION_INSTALLED_CAPTURE'}
# Shape-only Core refs: this component run does not execute Factor/Core/EOD.
shape={'path':'component-only-not-core-evidence.json','sha256':'a'*64}
state=new_future_state(day='20260916',context_sha=hashlib.sha256(canonical_json_bytes(context)).hexdigest(),
 context_schema=PRODUCTION_CONTEXT_SCHEMA,state={'schema_version':'cn-daily-factor-state.v1',
 'calendar_receipt_ref':shape,'core_checkpoint_ref':shape,'core_observation_refs':{'LOW':shape,'W80':shape}})
phases=[]
with pytest.MonkeyPatch.context() as patch:
 calls=offline_https(patch)
 result=bind_future_calendar(workspace=workspace,context=context,context_sha=state['context_sha256'],state=state,save_state=lambda s:phases.append(s['phase']))
 proof=read_next_session_proof(workspace=str(workspace),eod_trade_date=state['trade_date'],publication_ref=result['next_session_calendar_proof_ref'])
 assert calls==['DOCUMENTATION','SSE','SZSE','BSE']
 assert proof['live_eligible'] is True and proof['consumer_admission'] is False
output={'state':'PASS','scope':'INSTALLED_CALENDAR_COMPONENT_WITH_OFFLINE_HTTPS_AND_GUARD_TEST_SEAMS',
 'commit':receipt['commit'],'import_origin':quant_investor.__file__,'installed_verifier_mocked':False,
 'real_provider_calls':False,'production_deployed':False,'core_inputs_shape_only':True,'full_eod_proof':False,
 'morning_proof':False,'phases':phases,'next_open_session':proof['proof']['next_open_session'],
 'workspace':str(workspace),'publication_ref':result['next_session_calendar_proof_ref'],
 'transport_evidence_ref':result['transport_evidence_ref'],'finished_at':datetime.now(timezone.utc).isoformat()}
(root/'production-calendar-installed-smoke-proof.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output),flush=True)
'''
env=dict(os.environ);env.pop('PYTHONPATH',None);env['UV_OFFLINE']='1'
status={'state':'RUNNING','scope':'INSTALLED_OFFLINE_CALENDAR_COMPONENT','started_at':datetime.now(timezone.utc).isoformat(),'root':str(root),'production_deployed':False}
with (root/'production-calendar-installed-smoke.log').open('w') as output:
 child=subprocess.Popen([receipt['python'],'-I','-c',program,str(root),str(helpers),sysconfig.get_paths()['purelib']],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 status['pid']=child.pid
 (root/'production-calendar-installed-smoke-status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status),flush=True)
 code=child.wait()
status.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
(root/'production-calendar-installed-smoke-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
raise SystemExit(code)
