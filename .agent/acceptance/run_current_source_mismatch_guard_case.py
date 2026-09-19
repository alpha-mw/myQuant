"""Run one reviewed Case8 helper only after a complete current installed first day."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path
source=Path('/Users/maxwell/mySpace/myQuant');root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
status_path=root/'current-source-mismatch-guard-driver-status.json'
if status_path.exists():raise SystemExit('Case8 already attempted; inspect retained evidence')
first=json.loads((root/'configured-driver-status.json').read_bytes())
assert first['state']=='PASS' and first['producer_commit']==receipt['commit']
assert (root/'configured-native-proof.json').is_file()
helper_root=root/'case8-guard-helpers';helper_root.mkdir(mode=0o700,exist_ok=False)
name='current_source_mismatch_reader_case.py';raw=(source/'.agent/acceptance'/name).read_bytes()
compile(raw,name,'exec');(helper_root/name).write_bytes(raw);(helper_root/name).chmod(0o600)
pin=hashlib.sha256(raw).hexdigest()
(root/'case8-guard-helper-manifest.json').write_text(json.dumps({'path':str(helper_root/name),'sha256':pin,'runtime_commit':receipt['commit'],'scope':'PHYSICAL_FAULT_EXACT_READ_ROUTING_TEST_SEAM'},indent=2)+'\n')
program='''import sys,json
from pathlib import Path
import quant_investor
root=Path(sys.argv[1]);receipt=json.loads((root/'fixture-receipt.json').read_bytes())
assert str(Path(quant_investor.__file__).resolve())==receipt['runtime_verification']['import_origin']
sys.path[:0]=[str(root/'case8-guard-helpers'),str(root/'repository'),str(root/'repository/scripts'),str(root/'repository/tests/unit')]
from current_source_mismatch_reader_case import run
run(root)
'''
env=dict(os.environ);env.pop('PYTHONPATH',None)
state={'state':'RUNNING','case':8,'root':str(root),'runtime_commit':receipt['commit'],'started_at':datetime.now(timezone.utc).isoformat(),'production_deployed':False,'automatic_retry':False,'harness_sha256':pin}
with (root/'current-source-mismatch-guard-native.log').open('w') as output:
 child=subprocess.Popen([receipt['python'],'-I','-B','-c',program,str(root)],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 state['child_pid']=child.pid;status_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
 code=child.wait()
state.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat(),helper_unchanged=hashlib.sha256((helper_root/name).read_bytes()).hexdigest()==pin)
if not state['helper_unchanged']:state['state']='FAIL'
status_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
raise SystemExit(0 if state['state']=='PASS' else 1)
