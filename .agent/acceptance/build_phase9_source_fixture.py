"""Current collector/native-render acceptance over retained synthetic source artifacts.

Journal wrappers are synthetic and this fixture cannot pass full native EOD admission.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tempfile
from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.intelligence.pool_tabular import observation_bindings, encode_top100, tabular_documents
from quant_investor.intelligence.decision_report import build_decision_report, DOMAINS
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.daily_contract import GRAPH_SHA256, NodeState
from quant_investor.operations.portfolio_binding import freeze_portfolio_state
from _native_daily_store_fixture import NativeStoreFixture, DAYS
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter
from scripts.daily_dashboard_sealed import SealedDashboardAdapter
from quant_investor.market import cn_benchmark_store as benchmark

SOURCE = Path('/private/tmp/myquant-final-source-native-20260909T030031Z/factor-workspace')
DAY = '20260828'


def build(*, publish_current_dashboard=True, future_benchmark_day=None, prefix='phase9-source-collector-'):
    root=Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    completed=json.loads((SOURCE/f'results/operations/daily_production/CN/{DAY}/completion.v1.json').read_bytes())
    def old(node,key):
        term=json.loads((SOURCE/completed['node_terminal_refs'][node]['path']).read_bytes())
        ref=term['output_refs'][key];raw=(SOURCE/ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        return raw
    def put(name,raw):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        path.write_bytes(raw);path.chmod(0o600)
        return {'path':name,'sha256':hashlib.sha256(raw).hexdigest()}
    def doc(name,value):return put(name,canonical_json_bytes(value))
    book=NativeStoreFixture(root)
    for day in DAYS:args=book.advance(day)
    plan=prepare_store_plan(args);plan_ref={'path':plan['plan_path'],'sha256':plan['plan_sha256']}
    native_result=json.loads(old('decision','result'))
    journal=DailyJournal(str(root),DAY)
    with journal.locked():
        portfolio=freeze_portfolio_state(journal=journal,store_plan_ref=plan_ref,as_of=native_result['as_of'])
    release=doc('fixture-release.json',{'synthetic':True,'original_factor_producer':'3c46','new_eod_admission':False})
    store=StoreCloseAdapter(arguments=args,trade_date=DAY,plan_ref=plan_ref,release_ref=release)
    store.execute(store.template());store_outputs=store.probe(store.template()).outcome.output_refs
    if future_benchmark_day is not None:
        dates = ['2026-08-19', '2026-08-20', '2026-08-21', *DAYS, future_benchmark_day]
        rows = [dict(date=d, ts_code=symbol, close=1000.0+i+j,
                     source_system='tushare.index_daily', coverage='exact_close', value_date=d)
                for j,d in enumerate(dates) for i,symbol in enumerate(benchmark.REQUIRED_CODES)]
        capture = doc('fixture-future-benchmark-capture.json', {'synthetic': True, 'day': future_benchmark_day})
        generation = benchmark.publish_generation(root/'data/parquet/cn/benchmarks', rows=rows,
            generation_id='benchmark-future-tail-synthetic', captured_at=future_benchmark_day+'T10:00:00Z',
            expected_pointer_sha256=book.benchmark_pointer, acquisition_receipt_ref=capture)
        (root/'portfolio_dashboard/inputs/cn_index_benchmark.csv').write_bytes(
            benchmark.compatibility_csv_bytes(generation['rows']))
    legacy={name:json.loads(old('top100',name)) for name in ('factor_research_rank.json','manifest.json','selected_symbols.json','publish_receipt.json')}
    rank=legacy['factor_research_rank.json']
    observations=[json.loads(old(node,alias)) for node,alias in (('low_observation','LOW'),('w80_observation','W80'))]
    docs=tabular_documents(legacy,bindings=observation_bindings(rank,observations,approved_theme_policy_v2()),generated_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),parquet=encode_top100(rank))
    top={name:put('imported-top100/'+name,raw) for name,raw in docs.items()}
    factor={'generation':put('imported-factor.json',old('factor','generation'))}
    theme={'artifact':put('imported-theme.json',old('theme','artifact')),'capture':put('imported-theme-capture.json',old('theme','capture'))}
    result_ref=doc('imported-decision-result.json',native_result)
    physical={domain:[result_ref] for domain in DOMAINS}
    physical['portfolio']=[portfolio['state_ref'],*portfolio['state']['payload']['source_refs']]
    report=build_decision_report(result=native_result,result_ref=result_ref,portfolio_state=portfolio['state'],portfolio_state_ref=portfolio['state_ref'],domain_physical_refs=physical,freshness_reports={'fundamental':None,'macro':None},created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    decision={'result':result_ref,'decision.v2.json':doc('imported-decision.v2.json',report)}
    refs={}
    with journal.locked():
        for name,outputs in [('factor',factor),('top100',top),('theme',theme),('decision',decision),('store',store_outputs)]:
            inputs={}
            if name in ('theme','decision'):inputs['pool']=top['manifest.json']
            if name=='store':inputs['native_plan']=plan_ref
            request={'schema_version':'cn-daily-node-request.v1','trade_date':DAY,'node_id':name,'graph_sha256':GRAPH_SHA256,'release_ref':release,'adapter_sha256':'a'*64,'policy_refs':{},'input_refs':inputs}
            journal.begin(request);row=journal.finish(request,state=NodeState.SUCCEEDED,output_refs=outputs)
            refs[name]=row['terminal_ref']
        def existing(path):return {'path':path,'sha256':hashlib.sha256((root/path).read_bytes()).hexdigest()}
        dashboard=SealedDashboardAdapter(workspace=str(root),journal=journal,release_ref=release,plan_ref=plan_ref,
            market_ref=existing('data/parquet/cn/_snapshots/synthetic-20260828.json'),
            benchmark_ref=existing('portfolio_dashboard/inputs/cn_index_benchmark.csv'),risk_free_ref=existing('portfolio_dashboard/inputs/cn_govt_bond_yield.csv'),
            authority_terminal_refs=refs,publish_current_dashboard=publish_current_dashboard)
        dashboard.prepare();request=dashboard.template();journal.begin(request)
        dashboard.execute(request);outcome=dashboard.probe(request).outcome
        journal.finish(request,state=outcome.state,output_refs=outcome.output_refs)
    return root,dashboard,request,outcome

if __name__=='__main__':
    root,node,request,outcome=build()
    raw=(root/outcome.output_refs['daily_evidence']['path']).read_bytes();evidence=json.loads(raw)
    assert evidence['payload']['top100_count']==100
    assert sum(evidence['payload']['decision_state_counts'].values())==100
    assert set(outcome.output_refs)=={'capture','v1','v2','daily_evidence'}
    from export_cn_aggressive_dashboard_data import _expected_output_paths
    assert not any(path.exists() for path in _expected_output_paths(root))
    record={'scope':'CURRENT_COLLECTOR_AND_NATIVE_FINANCIAL_CAPTURE','synthetic':True,'root':str(root),'day':DAY,'outputs':outcome.output_refs,'count':100,'decision_state_counts':evidence['payload']['decision_state_counts'],'serving_files_written':False,'full_native_eod_admission':False,'limitations':['Factor/Theme/Decision source artifacts retained from frozen synthetic3c46 run; new journal wrappers are explicit test records','Current Top100 binary conversion, Decision report, Store close, five-source collector and financial Dashboard renderer exercised; no full native EOD claim']}
    Path('.agent/acceptance/phase9-native-draft-evidence.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
