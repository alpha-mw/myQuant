"""New immutable fixture input revision; preserve failed producer request/core."""
from pathlib import Path
import hashlib
import json
import socket
import sys
import quant_investor

root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'fixture-receipt.json').read_bytes())
assert Path(quant_investor.__file__).resolve()==Path(receipt['runtime_verification']['import_origin']).resolve()
repo=root/'repository'
sys.path.extend([str(repo),str(repo/'scripts'),str(repo/'tests/unit'),'/Users/maxwell/mySpace/myQuant/.venv/lib/python3.13/site-packages'])
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.unified import system_calendar_capture
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
from quant_investor.system.release_install import verify_running_release_install_input
from _native_daily_calendar_fixture import synthetic_calendar_transport
w=root/'factor-workspace'
status=json.loads((root/'historical-upstream-run-status.json').read_bytes())
assert status['status']=='FAILED' and status['attempt']==2
assert verify_running_release_install_input((root/'release-input.json').read_bytes(),repository_root=repo)['state']=='PASS'

original_connect=socket.socket.connect
def connect(sock,address):
    if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
        raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
    return original_connect(sock,address)
socket.socket.connect=connect


def ref(path):
    return {'path':str(path.relative_to(w)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def read(reference):
    path=w/reference['path']
    assert path.resolve(strict=True).is_relative_to(w)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
    return json.loads(path.read_bytes())


def put(name,value):
    path=w/'historical-fixture'/name
    raw=canonical_json_bytes(value)
    if path.exists():
        assert path.read_bytes()==raw
    else:
        path.write_bytes(raw);path.chmod(0o600)
    return ref(path)

old_ref=ref(w/'historical-fixture/request.json')
old=read(old_ref)
collection=read(old['recipe_ref'])
context_ref=collection['recipes']['20260828']['factor_loop_context_ref']
context=read(context_ref)
assert 'release_commit' not in context and 'calendar_capture_parent' not in context
parent=w/'data/private/historical-factor-calendar-captures'
new_context={**context,'release_commit':receipt['commit'],'calendar_capture_parent':str(parent)}
new_context_ref=put('loop-context-complete.json',new_context)
new_collection=json.loads(json.dumps(collection))
for recipe in new_collection['recipes'].values():
    assert recipe['factor_loop_context_ref']==context_ref
    recipe['factor_loop_context_ref']=new_context_ref
new_collection_ref=put('recipes-complete-context.json',new_collection)
new_request={**old,'recipe_ref':new_collection_ref}
new_request_ref=put('request-complete-context.json',new_request)
maintenance=json.loads((root/'successor-maintenance-result-2026-08-28.json').read_bytes())
core=maintenance['core_completion_ref']
verified=validate_daily_maintenance_receipt(workspace_root=w,receipt_path=core['path'],expected_receipt_sha256=core['sha256'])
assert verified['target_date']=='20260828' and verified['historical_session']['prospective'] is False
baseline_ref=json.loads((root/'full-completion-ref.json').read_bytes())
baseline=read(baseline_ref)
assert baseline['trade_date']=='20260827'
baseline_inputs=read(baseline['native_inputs_ref'])
assert hashlib.sha256((w/'results/factors/_active.json').read_bytes()).hexdigest()==baseline_inputs['factor_pointer_sha256']
name=f"tushare-calendar-20260828-{receipt['commit'][:7]}-{core['sha256'][:12]}"
assert not (parent/name).exists(), 'inspect retained Calendar attempt rather than repeat acquisition'
parent.mkdir(parents=True,mode=0o700,exist_ok=True)
print('START native synthetic Calendar input preparation for retained core',flush=True)
with synthetic_calendar_transport(fixture_source=repo/'tests/unit/test_tushare_calendar_authority.py',cutoff='2026-08-28'):
    captured=system_calendar_capture(workspace_root=str(w),capture_parent=str(parent),capture_root_name=name,
        cutoff_date='2026-08-28',release_repository_root=str(repo),
        release_install_input_path=new_context['release_install_input_ref']['path'],
        expected_release_install_input_sha256=new_context['release_install_input_ref']['sha256'])
proof={'synthetic':True,'revision_reason':'MISSING_PRODUCER_CONTEXT_FIELDS',
    'old_request_ref':old_ref,'new_request_ref':new_request_ref,'old_context_ref':context_ref,
    'new_context_ref':new_context_ref,'original_core_ref':core,'calendar_capture':captured,
    'original_inputs_unchanged':ref(w/old_ref['path'])==old_ref and ref(w/context_ref['path'])==context_ref,
    'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'note':'New input revision and separately prepared native synthetic Calendar; no historical failure relabeled as success.'}
assert proof['original_inputs_unchanged']
(root/'historical-context-revision-proof.json').write_text(json.dumps(proof,indent=2,default=str)+'\n')
print('PASS new immutable producer context and native Calendar source prepared',flush=True)
