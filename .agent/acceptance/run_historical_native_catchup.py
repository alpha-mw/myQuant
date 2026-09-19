"""Installed historical batch acceptance; no native validators/results are replaced.

Uses exact historical synthetic Theme/Industry/Exposure inputs from a previous
native fixture, while rebuilding both days' Market/PIT/Factor/Store/Decision/EOD.
Provider inputs and clocks are synthetic and remain ineligible for prospective OOS.
"""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import sys
from unittest.mock import patch

import quant_investor

CASE = Path(sys.argv[1]).resolve(strict=True)
receipt = json.loads((CASE / 'fixture-receipt.json').read_bytes())
assert Path(quant_investor.__file__).resolve() == Path(receipt['runtime_verification']['import_origin']).resolve()
REPOSITORY = CASE / 'repository'
sys.path.extend([str(REPOSITORY), str(REPOSITORY / 'scripts'), str(REPOSITORY / 'tests/unit'),
    '/Users/maxwell/mySpace/myQuant/.venv/lib/python3.13/site-packages'])

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.release_install import verify_running_release_install_input
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from quant_investor.operations.catchup_binding import COLLECTION_SCHEMA, read_catchup_binding
from quant_investor.market import daily_maintenance as maintenance_native
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.cn_history_audit import run_cn_history_audit
from quant_investor.strategy_records import event_store
from quant_investor.market import cn_benchmark_store as benchmark
from scripts import daily_materialization
from scripts.daily_production import dispatch_daily_request
from scripts.daily_completion_replay import replay_native_completion
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_daily_calendar_fixture import synthetic_calendar_transport, publish_daily_future_proof
import _native_daily_maintenance_fixture as fixture_maintenance
from _native_shared_macro_fixture import build_shared_macro
from test_fundamental_generation_promotion import _publish_verified_primary
from test_daily_evidence_requested_session import capture
from test_cn_dashboard_export import _write_risk_free
import pandas as pd

WORKSPACE = CASE / 'factor-workspace'
OLD = Path('/private/tmp/myquant-native-v2-five-20260908T135108Z/factor-workspace')
DAYS = ['20260828', '20260831']
assert verify_running_release_install_input((CASE / 'release-input.json').read_bytes(),
    repository_root=REPOSITORY)['state'] == 'PASS'
assert json.loads((WORKSPACE / 'SYNTHETIC-DATA-CLOCK.json').read_bytes())['synthetic'] is True
_original_connect = socket.socket.connect

def connect(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
    return _original_connect(sock, address)
socket.socket.connect = connect


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(WORKSPACE)), 'sha256': sha(path)}


def put(path, value):
    path = WORKSPACE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent
    while parent != WORKSPACE:
        parent.chmod(0o700)
        parent = parent.parent
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    if path.exists():
        assert path.read_bytes() == raw, 'synthetic fixture cannot overwrite different input'
    else:
        path.write_bytes(raw)
        path.chmod(0o600)
    return ref(path)


def read(reference, workspace=WORKSPACE):
    path = workspace / reference['path']
    assert path.resolve(strict=True).is_relative_to(workspace.resolve(strict=True))
    assert sha(path) == reference['sha256']
    return json.loads(path.read_bytes())


copied = {}
def copy_source(reference):
    relative = PurePosixPath(reference['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = OLD / relative
    raw = path.read_bytes()
    assert sha(path) == reference['sha256']
    copied[reference['path']] = reference['sha256']
    saved = put(str(relative), raw)
    assert saved == reference
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError):
        return saved
    def visit(value):
        if type(value) is dict:
            if set(value) == {'path', 'sha256'}:
                if value['path'] not in copied:
                    copy_source(value)
            else:
                for child in value.values():
                    visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)
    visit(document)
    return saved


def prepare_request():
    previous_ref = json.loads((CASE / 'full-completion-ref.json').read_bytes())
    checked = replay_native_completion(workspace=str(WORKSPACE), trade_date='20260827', completion_ref=previous_ref)
    assert checked['native_replay_validated'] and checked['synthetic']
    previous = inspect_recorded_completion(workspace=str(WORKSPACE), trade_date='20260827', completion_ref=previous_ref)
    recipe = previous['completed_handoff_snapshot'].document('recipe')
    book = WORKSPACE / 'results/strategy_records/CN/aggressive_tech_manufacturing'
    events = event_store.load_generation(book / '_event_store')
    closures = list(events['closures'])
    for day in DAYS:
        iso = f'{day[:4]}-{day[4:6]}-{day[6:]}'
        declaration = put(f'historical-fixture/owner-empty-{day}.json', {
            'synthetic': True, 'trade_date': iso,
            'dimensions': dict.fromkeys(event_store.EVENT_DIMENSIONS, [])})
        closures.append(event_store.build_empty_closure(trade_date=iso, sealed_at=iso+'T10:00:00Z',
            cutoff_at=iso+'T07:30:00Z', policy_ref=recipe['policy_refs']['store'],
            owner_declaration_ref=declaration, source_receipt_ref=None))
    event_store.publish_generation(book / '_event_store', generation_id='event-historical-fixture-20260831',
        generated_at='2026-09-01T10:00:00Z', expected_pointer_sha256=events['pointer_sha256'],
        closures=closures, policy_ref=recipe['policy_refs']['store'])
    days = ['2026-08-19', '2026-08-20', '2026-08-21'] + [row['trade_date'] for row in closures]
    old_benchmark = benchmark.load_generation(WORKSPACE / 'data/parquet/cn/benchmarks')
    capture_ref = put('historical-fixture/benchmark-source.json', {'synthetic': True, 'days': days})
    bm = benchmark.publish_generation(WORKSPACE / 'data/parquet/cn/benchmarks',
        generation_id='benchmark-historical-fixture-20260831', captured_at='2026-09-01T10:00:00Z',
        expected_pointer_sha256=old_benchmark['pointer_sha256'], acquisition_receipt_ref=capture_ref,
        rows=[{'date': day, 'ts_code': code, 'close': 1000.0+i+j, 'source_system': 'tushare.index_daily',
            'coverage': 'exact_close', 'value_date': day}
            for j, day in enumerate(days) for i, code in enumerate(benchmark.REQUIRED_CODES)])
    (WORKSPACE / 'portfolio_dashboard/inputs/cn_index_benchmark.csv').write_bytes(benchmark.compatibility_csv_bytes(bm['rows']))
    _write_risk_free(WORKSPACE, days)
    for name in ('cn_index_benchmark.csv', 'cn_govt_bond_yield.csv'):
        (WORKSPACE / 'portfolio_dashboard/inputs' / name).chmod(0o600)
    template_map = {}
    for day in DAYS:
        old_completion = json.loads((OLD / f'results/operations/daily_production/CN/{day}/completion.v1.json').read_bytes())
        old_materialized = read(old_completion['materialization_ref'], OLD)
        old_handoff = read(old_materialized['maintenance_handoff_ref'], OLD)
        old_recipe = read(old_handoff['recipe_ref'], OLD)
        value = deepcopy(recipe)
        value['target_trade_date'] = day
        value['previous_completion_ref'] = value['bootstrap_ref'] = None
        value['store_preimages'] = {'store_pointer_ref': None,
            'event_pointer_ref': ref(book / '_event_store/current.v1.json'),
            'benchmark_pointer_ref': ref(WORKSPACE / 'data/parquet/cn/benchmarks/_latest.json')}
        value['dashboard_sources'] = {
            'benchmark_ref': ref(WORKSPACE / 'portfolio_dashboard/inputs/cn_index_benchmark.csv'),
            'risk_free_ref': ref(WORKSPACE / 'portfolio_dashboard/inputs/cn_govt_bond_yield.csv')}
        value['research_sources']['as_of'] = f'{day[:4]}-{day[4:6]}-{day[6:]}T13:30:00Z'
        for name in ('industry_source_ref', 'theme_source_ref', 'exposure_rows_ref'):
            value['research_sources'][name] = copy_source(old_recipe['research_sources'][name])
        for name in ('fundamental', 'macro'):
            value['research_sources'][name] = {'mode': 'MAINTENANCE_STAGE', 'source_ref': None}
        template_map[day] = value
    templates = put('historical-fixture/recipes.json', {'schema_version': COLLECTION_SCHEMA, 'recipes': template_map})
    calendar = capture('2026-09-01T20:20:00+08:00')
    raw_ref = put('historical-fixture/calendar.raw.json', calendar.raw_response_bytes)
    calendar_ref = put('historical-fixture/calendar.json', {
        **calendar.receipt, 'raw_response_path': str(WORKSPACE / raw_ref['path'])})
    request = {'schema_version': 'cn-daily-production-request.v1',
        **{k: recipe[k] for k in ('market','strategy_id','graph_sha256','release_install_ref')},
        'action': 'CATCH_UP', 'target_trade_date': DAYS[-1], 'recipe_ref': templates,
        'maintenance_handoff_ref': None, 'calendar_ref': calendar_ref, 'raw_calendar_ref': raw_ref,
        'previous_completion_ref': previous_ref, 'day_input_refs': {}}
    put('historical-fixture/source-provenance.json', {'synthetic': True,
        'historical_synthetic_source_workspace': str(OLD), 'copied_input_sha256': copied,
        'production_results_copied': False, 'prospective': False})
    return put('historical-fixture/request.json', request), request


native_runner = maintenance_native.run_cn_daily_maintenance
native_execute = daily_materialization.execute_daily_recipe
maintenance_calls = []

def maintained(**kwargs):
    historical = kwargs['_historical_calendar_input']
    day = historical['target_trade_date']
    assert day in DAYS
    iso = f'{day[:4]}-{day[4:6]}-{day[6:]}'
    maintenance_calls.append(day)
    fixture = NativeFactorInputs(WORKSPACE / 'synthetic-inputs', extra_future_sessions=3)
    arguments = fixture.day(4 + DAYS.index(day), extra_history=9)
    assert arguments['as_of'] == day
    strict_market_from_factor_inputs(WORKSPACE, arguments, macro_ready_layout=True,
        pit_observed_at=iso+'T00:00:00Z', simulated_available_at=iso+'T07:30:00Z')
    dates = pd.read_parquet(arguments['exchange_calendar_path'])['open_session'].tolist()
    audit, path = run_cn_history_audit(data_root=WORKSPACE/'data', output_root=WORKSPACE/'data/private/synthetic-history',
        days=100, end_date=day, allow_online=False, trade_dates=[d.strftime('%Y%m%d') for d in dates[-100:]])
    (CASE / ('successor-market-audit-'+iso+'.json')).write_text(json.dumps({'history': audit, 'audit_path': str(path)}, indent=2)+'\n')
    publish_daily_future_proof(release_root=CASE, workspace=WORKSPACE, trade_date=day)
    def fundamental(ctx):
        canonical = WORKSPACE / 'data/parquet/cn'
        pointer = canonical / '_fundamental_latest.json'
        _publish_verified_primary(canonical, run_id='native-history-'+day, symbols_override=fixture.symbols,
            evidence_namespace='native-history-'+day, expected_pointer_sha256=sha(pointer))
        result = maintenance_native._fundamental_health(ctx)
        assert result['evidence'].get('research_source_ref'), result
        return result
    def macro(ctx):
        result = build_shared_macro(WORKSPACE, iso)
        source = put(str(ctx.attempt_root.relative_to(WORKSPACE) / 'macro-research-source.json'),
            {'classification': 'CANONICAL_MACRO_READY', 'source': result['closure_ref']})
        return {'status': 'READY', 'write_performed': True, 'blockers': [],
            'evidence': {'research_source_ref': source}}
    native_results = {}
    def routed(**parameters):
        components = parameters['components']
        parameters.update(now=datetime.now(timezone.utc), _expected_target_trade_date=None,
            _historical_calendar_input=historical, components=maintenance_native.MaintenanceComponents(
                pit=components.pit, market=components.market, history=components.history,
                fundamental=fundamental, macro_release=macro))
        native_results['result'] = native_runner(**parameters)
        return native_results['result']
    with synthetic_calendar_transport(fixture_source=REPOSITORY/'tests/unit/test_tushare_calendar_authority.py', cutoff=iso), \
         patch.object(fixture_maintenance, 'run_cn_daily_maintenance', routed):
        fixture_maintenance.maintenance(CASE, iso, core_completed=kwargs['core_completed'],
            core_replay_completed=kwargs['_core_replay_completed'])
    return native_results['result']


def main():
    relative = sys.argv[2] if len(sys.argv) > 2 else 'historical-fixture/request.json'
    selected = Path(relative)
    assert not selected.is_absolute() and '..' not in selected.parts
    path = WORKSPACE / selected
    assert path.resolve(strict=False) == path
    if len(sys.argv) > 2:
        assert path.is_file(), 'explicit request revision is missing'
    if path.exists():
        request_ref, request = ref(path), json.loads(path.read_bytes())
    else:
        request_ref, request = prepare_request()
    proof_path = CASE / 'public-historical-native-proof.json'
    assert not proof_path.exists(), 'completed acceptance must be replayed explicitly, not rerun'
    stop_path = CASE / 'intentional-interday-stop.json'
    def interday(**kwargs):
        bound = read_catchup_binding(workspace=str(WORKSPACE), binding_ref=kwargs['_catchup_binding_ref'])
        if bound['binding']['trade_date'] == DAYS[-1] and not stop_path.exists():
            stop_path.write_text(json.dumps({'synthetic_interruption': True, 'after_completed_date': DAYS[0],
                'before_execute_date': DAYS[-1], 'request_ref': request_ref})+'\n')
            raise InterruptedError('SYNTHETIC_INTERDAY_INTERRUPTION')
        return native_execute(**kwargs)
    with patch.object(maintenance_native, 'run_cn_daily_maintenance', maintained), \
         patch.object(daily_materialization, 'execute_daily_recipe', interday):
        if not stop_path.exists():
            try:
                first_result = dispatch_daily_request(workspace=str(WORKSPACE), request_ref=request_ref,
                    release_install_ref=request['release_install_ref'], synthetic=True)
            except InterruptedError as exc:
                assert str(exc) == 'SYNTHETIC_INTERDAY_INTERRUPTION'
            else:
                (CASE / 'historical-first-dispatch-result.json').write_text(
                    json.dumps(first_result, indent=2)+'\n')
                raise AssertionError('missing intended interday interruption')
        print('PASS first completed historical day; resume identical root request', flush=True)
        result = dispatch_daily_request(workspace=str(WORKSPACE), request_ref=request_ref,
            release_install_ref=request['release_install_ref'], synthetic=True)
    (CASE / 'public-historical-native-result.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['business_state'] == 'COMPLETE', result
    assert [row['execution_state'] for row in result['days']] == ['NO_ACTION', 'SUCCEEDED']
    before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in WORKSPACE.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('completed historical repeat invoked writer')
    with patch.object(maintenance_native, 'run_cn_daily_maintenance', forbidden), \
         patch.object(DailyJournal, 'locked', forbidden):
        repeated = dispatch_daily_request(workspace=str(WORKSPACE), request_ref=request_ref,
            release_install_ref=request['release_install_ref'], synthetic=True)
    assert repeated['execution_state'] == 'NO_ACTION'
    assert before == {str(p): (sha(p), p.stat().st_mtime_ns) for p in WORKSPACE.rglob('*') if p.is_file()}
    proof = {'synthetic': True, 'prospective': False, 'producer_commit': receipt['commit'],
        'driver_sha256': sha(Path(__file__)), 'trade_dates': DAYS, 'request_ref': request_ref,
        'maintenance_calls_this_process': maintenance_calls, 'first_day_preserved_on_resume': True,
        'no_write_completed_repeat': True, 'full_native_results': result,
        'public_cli_synthetic_guard_preserved': True, 'entry': 'installed internal dispatcher synthetic=True'}
    revision = CASE / 'historical-context-revision-proof.json'
    if revision.exists():
        previous = json.loads(revision.read_bytes())
        assert previous['new_request_ref'] == request_ref
        proof['prior_failed_request_ref'] = previous['old_request_ref']
        proof['input_revision_proof_sha256'] = sha(revision)
        proof['first_day_reused_retained_core'] = True
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    print('PASS INSTALLED HISTORICAL UPSTREAM CATCHUP', flush=True)

if __name__ == '__main__':
    main()
