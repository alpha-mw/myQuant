"""Continue exact Case4 inputs with the original offline capability, no new capture."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path
source=Path('/Users/maxwell/mySpace/myQuant');root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
for name in ('missing-store-driver-status.json','missing-store-resume-driver-status.json'):
 previous=json.loads((root/name).read_bytes());assert previous['state']=='FAIL' and previous['helper_drift']==[]
status_path=root/'missing-store-scoped-resume-driver-status.json'
if status_path.exists():raise SystemExit('Scoped retained recovery already attempted')
helpers=root/'scoped-recovery-validation-helpers';helpers.mkdir(mode=0o700,exist_ok=False)
old_pins=json.loads((root/'validation-helper-manifest.json').read_bytes())['helper_sha256']
pins={}
for name in ('_native_missing_store_case.py','_native_synthetic_clock.py'):
 raw=(root/'validation-helpers'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==old_pins[name]
 (helpers/name).write_bytes(raw);(helpers/name).chmod(0o600);pins[name]=old_pins[name]
program='''import json,sys,socket
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import quant_investor
root=Path(sys.argv[1]);receipt=json.loads((root/'fixture-receipt.json').read_bytes())
assert str(Path(quant_investor.__file__).resolve())==receipt['runtime_verification']['import_origin']
sys.path[:0]=[str(root/'scoped-recovery-validation-helpers'),str(root/'repository'),str(root/'repository/scripts'),str(root/'repository/tests/unit')]
sys.path.append(sys.argv[2])
def forbidden(*args,**kwargs):raise AssertionError('CASE4_RETAINED_RECOVERY_ATTEMPTED_NEW_ACQUISITION')
socket.socket.connect=forbidden
from _native_daily_calendar_fixture import configured_future_calendar_scope
from _native_missing_store_case import dispatch_case
from quant_investor.market import future_calendar_producer,next_session_acquisition,tushare_calendar_authority
source=json.loads((root/'configured-source-result.json').read_bytes())
with configured_future_calendar_scope(root=root,trade_date='20260827'),ExitStack() as stack:
 stack.enter_context(patch.object(future_calendar_producer,'capture_next_session_calendar',forbidden))
 stack.enter_context(patch.object(next_session_acquisition,'capture_next_session_calendar',forbidden))
 stack.enter_context(patch.object(tushare_calendar_authority,'capture_trusted_provider_calendar_evidence',forbidden))
 dispatch_case(root=root,workspace=root/'factor-workspace',request_ref=source['request_ref'],config_ref=source['config_ref'],release_install_ref=source['release_install_ref'],resume_retained=True)
'''
(root/'scoped-recovery-helper-manifest.json').write_text(json.dumps({'helper_sha256':pins,'runtime_commit':receipt['commit'],'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'change':'Re-enter exact original offline fixture capability; original helpers unchanged; forbid all capture entries'},indent=2)+'\n')
state={'state':'RUNNING','case':4,'root':str(root),'producer_commit':receipt['commit'],'started_at':datetime.now(timezone.utc).isoformat(),'production_deployed':False,'continued_from_retained_test_failure':True,'new_acquisition_forbidden':True}
env=dict(os.environ);env.pop('PYTHONPATH',None)
with (root/'missing-store-scoped-resume-native.log').open('w') as log:
 child=subprocess.Popen([receipt['python'],'-I','-B','-c',program,str(root),str(source/'.venv/lib/python3.13/site-packages')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 state['child_pid']=child.pid;status_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
 code=child.wait()
state.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat(),helper_drift=[name for name,sha in pins.items() if hashlib.sha256((helpers/name).read_bytes()).hexdigest()!=sha])
if state['helper_drift']:state['state']='FAIL'
status_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
raise SystemExit(0 if state['state']=='PASS' else 1)
