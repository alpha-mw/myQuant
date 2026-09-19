"""Exercise the retained native callback under unchanged limits after the narrow repair."""
from datetime import datetime,timezone
import hashlib,json,os,sys,time
from pathlib import Path

source=Path('/Users/maxwell/mySpace/myQuant');sys.path.insert(0,str(source))
root=Path('/private/tmp/myquant-contextual-timeout-repro-20260916T144304Z')
output=root/'retained-callback-optimized';output.mkdir(mode=0o700,exist_ok=False)
fixture=root/'pytest-retained/test_full_prospective_store_co0';workspace=fixture/'workspace'
def inventory():
 return {str(p.relative_to(fixture)):(p.stat().st_mode,p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in fixture.rglob('*') if p.is_file()}
before=inventory()
from quant_investor.contracts import parse_canonical_json_bytes,contract_catalog_sha256
from quant_investor.system.store import SystemStore
from quant_investor.system.validation import _run_callback_worker,MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS
from quant_investor.factors.governance.contextual import validate_prospective_contextual_run
request_path=workspace/'results/system/objects/system.validation_run_request/a4d10ebcc757c4610c603b8a98f7207a75e9cbdf9a457eee11b418b8f54db048.json'
request=parse_canonical_json_bytes(request_path.read_bytes())
store=SystemStore(workspace,source_root=fixture/'source',source_root_id='factor-full-prospective-source-root')
state={'state':'RUNNING','pid':os.getpid(),'scope':'RETAINED_NATIVE_CALLBACK_UNPROFILED_NOT_FULL_TEST',
 'native_limit_seconds':MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS,'timeout_changed':False,
 'started_at':datetime.now(timezone.utc).isoformat(),'catalog_sha256':contract_catalog_sha256(),
 'core_source_sha256':hashlib.sha256((source/'quant_investor/contracts/core.py').read_bytes()).hexdigest()}
(output/'status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
started=time.monotonic()
try:
 result,peak=_run_callback_worker(validate_prospective_contextual_run,store=store,validation_request=request,
 trusted_at=request['created_at'],maximum_seconds=MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS)
 state.update(state='PASS',peak_rss=peak,lane=result['lane'],raw_source_count=len(result['source_object_refs']),custody_records=len(result['custody_record_refs']),validated=result['validated'])
except Exception as exc:
 state.update(state='FAIL',error_type=type(exc).__name__,error=str(exc))
state.update(seconds=round(time.monotonic()-started,3),protected_files=len(before),retained_inventory_unchanged=inventory()==before,finished_at=datetime.now(timezone.utc).isoformat())
if not state['retained_inventory_unchanged']:state['state']='FAIL'
(output/'status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
raise SystemExit(0 if state['state']=='PASS' else 1)
