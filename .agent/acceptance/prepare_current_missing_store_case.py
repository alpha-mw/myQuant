"""Prepare a distinct installed Case4 workspace at the exact already-tested commit."""
from datetime import datetime,timezone
import json,os,subprocess,sys
from pathlib import Path
source=Path('/Users/maxwell/mySpace/myQuant')
primary=Path('/private/tmp/myquant-canonical-scan-20260916T152008Z')
base=json.loads((primary/'fixture-receipt.json').read_bytes())
assert json.loads((primary/'final-gates/status.json').read_bytes())['state']=='PASS'
location=source/'.agent/acceptance/phase15-current-case4-location.json'
if location.exists():raise SystemExit('Case4 already recorded; inspect instead of duplicating')
root=Path('/private/tmp/myquant-current-case4-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir(mode=0o700)
state={'state':'BUILDING','case':4,'root':str(root),'commit':base['commit'],'pid':os.getpid(),'production_deployed':False}
location.write_text(json.dumps(state,indent=2)+'\n')
os.environ['UV_OFFLINE']='1';os.environ['UV_CACHE_DIR']='/Users/maxwell/.cache/uv'
repository=root/'repository';subprocess.run(['git','clone','--quiet','--shared',str(primary/'repository'),str(repository)],check=True)
subprocess.run(['git','-C',str(repository),'checkout','--detach','--quiet',base['commit']],check=True)
def git(*args):return subprocess.check_output(['git','-C',str(repository),*args],text=True).strip()
assert git('rev-parse','HEAD')==base['commit']
from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.release_install import prepare_operational_release
release_root=root/'release';release_root.mkdir(mode=0o700)
try:
 prepared=prepare_operational_release(repository_root=repository,release_root=release_root,final_commit=base['commit'],final_tree=git('rev-parse','HEAD^{tree}'),created_at=None)
 raw=canonical_json_bytes({'deployed_release':prepared['release'],'release_install_evidence':prepared['release_install_evidence']})
 path=root/'release-input.json';path.write_bytes(raw);path.chmod(0o600)
 python=prepared['release_install_evidence']['payload']['python_executable']
 program='import sys,json;from pathlib import Path;from quant_investor.system.release_install import verify_running_release_install_input;print(json.dumps(verify_running_release_install_input(Path(sys.argv[1]).read_bytes(),repository_root=sys.argv[2])))'
 env=dict(os.environ);env.pop('PYTHONPATH',None)
 verified=json.loads(subprocess.check_output([python,'-I','-c',program,str(path),str(repository)],cwd=root,env=env,text=True));assert verified['state']=='PASS'
 receipt={'synthetic_test_environment':True,'repository':str(repository),'release_root':str(release_root),'release_input':str(path),'python':python,'commit':base['commit'],'runtime_verification':verified,'full_dag_proof':False,'production_deployed':False,'same_commit_as_primary':str(primary),'independent_case':4}
 (root/'fixture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 state.update(state='INSTALLED_VERIFIED',finished_at=datetime.now(timezone.utc).isoformat())
except BaseException as exc:
 state.update(state='FAIL',error_type=type(exc).__name__,error=str(exc));location.write_text(json.dumps(state,indent=2)+'\n');raise
location.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
