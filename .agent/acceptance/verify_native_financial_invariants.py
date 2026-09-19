"""Read-only financial/authority audit of an already completed native five-day proof."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

root = Path(sys.argv[1]).resolve(strict=True)
workspace = root / 'factor-workspace'
proof_path = root / 'five-native-trading-day-proof.json'
proof = json.loads(proof_path.read_bytes())
assert proof['full_five_day_proof'] is True and proof['synthetic'] is True
assert proof['single_attempt_all_nodes'] is True and proof['calendar_consecutive'] is True
marker = json.loads((workspace / 'SYNTHETIC-FIXTURE.json').read_bytes())
assert marker['synthetic'] is True and marker['real_account'] is False
record_root = workspace / 'results/strategy_records/CN/aggressive_tech_manufacturing'
observed = {}


def raw_ref(reference):
    relative = Path(reference['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = workspace / relative
    assert path.resolve(strict=True) == path and path.is_file()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference['sha256']
    observed[str(path)] = (hashlib.sha256(raw).hexdigest(), path.stat().st_mtime_ns)
    return raw


def doc(reference):
    return json.loads(raw_ref(reference))


rows = []
for day in proof['trade_dates']:
    completion = doc(proof['completion_refs'][day])
    assert completion['trade_date'] == day and completion['synthetic'] is True
    assert completion['status'] == 'SUCCEEDED'
    assert all(value is False for value in completion['authority'].values())
    terminals = {node: doc(reference) for node, reference in completion['node_terminal_refs'].items()}
    assert len(terminals) == 16
    assert all(t['state'] == 'SUCCEEDED' and all(v is False for v in t['authority'].values()) for t in terminals.values())
    output = terminals['store']['output_refs']
    manual = doc(output['manual'])
    raw_ref(output['ledger'])
    source = record_root / manual['source_record']
    assert source.parent == record_root and source.resolve(strict=True) == source
    source_ledger = source / 'ledger_after_manual_switch.parquet'
    assert hashlib.sha256(source_ledger.read_bytes()).hexdigest() == manual['source_contained_ledger_sha256']
    observed[str(source_ledger)] = (hashlib.sha256(source_ledger.read_bytes()).hexdigest(), source_ledger.stat().st_mtime_ns)
    previous = pd.read_parquet(source_ledger)
    current = pd.read_parquet(workspace / output['ledger']['path'])
    identity = ['symbol', 'shares', 'avg_cost', 'cost_basis']
    pd.testing.assert_frame_equal(previous[identity].sort_values('symbol').reset_index(drop=True),
        current[identity].sort_values('symbol').reset_index(drop=True), check_exact=True)
    assert manual['cash_after'] == manual['cash_before']
    assert manual['no_trade_performed'] is True and manual['trade_count'] == 0
    assert manual['applied_local_trades'] == manual['applied_owner_declared_trades'] == []
    assert manual['broker_order_trade_authority'] is False
    assert manual['valuation_trade_date'].replace('-', '') == day
    source_manual = source / 'manual_execution_manifest.json'
    old_raw = source_manual.read_bytes()
    assert hashlib.sha256(old_raw).hexdigest() == manual['source_manual_manifest_sha256']
    observed[str(source_manual)] = (hashlib.sha256(old_raw).hexdigest(), source_manual.stat().st_mtime_ns)
    old_manual = json.loads(old_raw)
    assert manual['market_value_after'] != old_manual['market_value_after']
    assert manual['financial_state_sha256'] != old_manual['financial_state_sha256']
    rows.append({'trade_date': day, 'completion_ref': proof['completion_refs'][day],
        'store_manual_ref': output['manual'], 'source_record': manual['source_record'],
        'shares_cost_cash_unchanged': True, 'new_strict_close_valuation': True,
        'no_trades': True, 'all_16_node_authorities_false': True})
for name in ('results/system/_active.json', 'results/mainline/_active.json'):
    assert not (workspace / name).exists(), 'unexpected System/Mainline activation'
for path, (digest, mtime) in observed.items():
    p = Path(path)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest and p.stat().st_mtime_ns == mtime
result = {'verified_at': datetime.now(timezone.utc).isoformat(), 'synthetic': True,
    'producer_commit': proof['producer_commit'], 'full_dag_proof_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'trade_dates': proof['trade_dates'], 'days': rows, 'read_sources_unchanged': True,
    'no_system_or_mainline_activation': True}
output = root / 'five-day-financial-invariants.json'
raw = (json.dumps(result, indent=2) + '\n').encode()
if output.exists():
    raise RuntimeError('financial audit already exists; retain original receipt')
output.write_bytes(raw)
print('PASS five-day cash/shares/cost/valuation and authority invariants')
