"""Native Dashboard replay; outer full-EOD and Store-to-EOD association are explicit seams."""
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone
import hashlib
import json

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.dashboard_evidence import DashboardEvidenceSources
from scripts import daily_completion_dashboard as replay
from scripts import daily_completion_store as store_replay
from scripts.cn_official_close_batch import inspect_frozen_close_commit

# Native Market manifests intentionally retain absolute roots. A copied fixture is
# not relocatable; create one new isolated fixture through its native builder.
import importlib.util
spec = importlib.util.spec_from_file_location("phase9_source_fixture", ".agent/acceptance/build_phase9_source_fixture.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root, node, native_request, outcome = module.build()
source = {"root":str(root), "day":node.journal.trade_date, "outputs":outcome.output_refs}
print("native-fixture-root", root, flush=True)
day = source['day']
base = f'results/operations/daily_production/CN/{day}'

def read(ref):
    raw=(root/ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    return json.loads(raw)

def put(path, value):
    raw=canonical_json_bytes(value); p=root/path
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o600)
    return {'path':path,'sha256':hashlib.sha256(raw).hexdigest()}

capture=read(source['outputs']['capture']);request=read(capture['request_ref'])
terminal_path=f"{base}/nodes/dashboard/{capture['request_ref']['sha256']}/attempt-0001/terminal.json"
terminal_ref={'path':terminal_path,'sha256':hashlib.sha256((root/terminal_path).read_bytes()).hexdigest()}
refs={name:request['input_refs']['authority.'+name] for name in ('store','factor','top100','theme','decision')}
plan=read(request['input_refs']['store_plan'])
frozen=inspect_frozen_close_commit(
    record_root=root/'results/strategy_records/CN/aggressive_tech_manufacturing',
    transaction_id=plan['transaction_id'], expected_plan_sha=request['input_refs']['store_plan']['sha256'],
    expected_source_pointer_sha=plan['preimages']['store_pointer_sha256'],
    expected_target=plan['requested_target'],
)
inputs={'schema_version':'cn-daily-native-inputs.v5','publish_current_dashboard':True,
        'dashboard_publication_policy':'native-eod-first.v1',
        **{target:request['input_refs'][key] for target,key in (
            ('store_plan_ref','store_plan'),('market_snapshot_ref','market'),
            ('benchmark_ref','benchmark'),('risk_free_ref','risk_free'))}}
input_ref=put(f'{base}/dashboard/fixture-replay-inputs.json',inputs)
recorded={'native_inputs_ref':input_ref,'release_ref':request['release_ref'],
          'node_terminal_refs':{**refs,'dashboard':terminal_ref}}
completion_ref=put(f'{base}/fixture-replay-envelope.json',recorded)
store_outputs=read(refs['store'])['output_refs']
collector=DashboardEvidenceSources(workspace=root,trade_date=day,release_ref=request['release_ref'],terminal_refs=refs)
recipe=read(request['input_refs']['daily_evidence_recipe'])
assert collector.build(recipe['created_at'])==read(source['outputs']['daily_evidence'])

def inventory():
    return {str(p.relative_to(root)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}

with patch.object(replay,'inspect_recorded_completion',return_value={'recorded_completion':recorded}), patch.object(
    store_replay,'inspect_recorded_completion',return_value={'recorded_completion':recorded}):
    before=inventory()
    result=replay.replay_completed_dashboard(workspace=str(root),trade_date=day,completion_ref=completion_ref)
    assert result['daily_evidence']==read(source['outputs']['daily_evidence'])
    assert inventory()==before
    # Serving files are mutable presentation state; immutable EOD replay must ignore them.
    head=root/'portfolio_dashboard/private/generated/cn_daily_completed_head.v1.json'
    head.parent.mkdir(parents=True,exist_ok=True)
    head.write_bytes(b'controlled invalid later serving head')
    head.chmod(0o600)
    before=inventory()
    again=replay.replay_completed_dashboard(workspace=str(root),trade_date=day,completion_ref=completion_ref)
    assert again==result and inventory()==before
receipt={'observed_at':datetime.now(timezone.utc).isoformat(),'scope':'NATIVE_DASHBOARD_V5_REBUILD_AND_FIVE_DOMAIN_COLLECTOR',
         'result':'PASS','root':str(root),'full_native_eod_admission':False,'store_frozen_commit_inspected':True,
         'outer_seams':['EOD recorded-completion admission','Selected source journal wrappers are synthetic; Store replay and frozen commit are native'],
         'no_writes_during_replay':True,'serving_head_independent':True,
         'count':result['daily_evidence']['payload']['top100_count']}
Path('.agent/acceptance/phase9-v5-native-dashboard-replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
