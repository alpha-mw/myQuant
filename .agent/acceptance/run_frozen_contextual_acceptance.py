"""Run the original failing test once against a verified frozen repaired candidate."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path
source=Path('/Users/maxwell/mySpace/myQuant');root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes());assert receipt['runtime_verification']['state']=='PASS'
out=root/'contextual-original-test';out.mkdir(mode=0o700,exist_ok=False)
repo=root/'repository';target='tests/unit/test_unified_factor_store_contextual.py::test_full_prospective_store_context_replays_1442_raw_sources_without_activation'
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1')
state={'state':'RUNNING','root':str(root),'commit':receipt['commit'],'scope':'ORIGINAL_CONTEXTUAL_TEST_FROZEN_SOURCE','native_limit_seconds':180,'timeout_changed':False,'profile_enabled':False,'started_at':datetime.now(timezone.utc).isoformat()}
with (out/'pytest.log').open('w') as stream:
 child=subprocess.Popen([str(source/'.venv/bin/python'),'-m','pytest',target,'-q','-ra','-p','no:cacheprovider','--basetemp',str(out/'pytest-retained')],cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
 state['pid']=child.pid;(out/'status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
 code=child.wait()
state.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256((out/'pytest.log').read_bytes()).hexdigest())
(out/'status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
raise SystemExit(code)
