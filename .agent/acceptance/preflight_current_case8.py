"""Mechanical installed preflight on retained physical faults; not Case8 acceptance."""
from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/private/tmp/myquant-canonical-scan-20260916T152008Z');receipt=json.loads((root/'fixture-receipt.json').read_bytes())
program='''import sys,json,hashlib,importlib.util
from pathlib import Path
import quant_investor
root=Path(sys.argv[1]);receipt=json.loads((root/'fixture-receipt.json').read_bytes());assert str(Path(quant_investor.__file__).resolve())==receipt['runtime_verification']['import_origin']
sys.path[:0]=[str(root/'repository'),str(root/'repository/scripts'),str(root/'repository/tests/unit')]
spec=importlib.util.spec_from_file_location('case8_preflight_helper',sys.argv[2]);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
from _verify_five_native_days import readonly_replay_guard
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_contract import EOD_NODE_IDS,ContractError
from scripts.daily_completion_replay import replay_native_completion
w=root/'factor-workspace';p=json.loads((root/'configured-native-proof.json').read_bytes());ref=p['completion_ref'];cases=[]
with readonly_replay_guard():
 baseline=read_daily_status(str(w),'20260827');assert all(baseline['nodes'][n]['state']=='SUCCEEDED' for n in EOD_NODE_IDS)
 inspect_recorded_completion(workspace=str(w),trade_date='20260827',completion_ref=ref)
 for node,key in [('market','market_input_ref'),('pit','market_pit_selection_ref')]:
  output=baseline['nodes'][node]['terminal']['output_refs'][key]
  fault=root/'current-source-mismatch-case8-financial-reader'/(node+'-physical-fault')
  original=h.file_identity(w/output['path'])
  with h.route_physical_fault(w,output['path'],fault) as counts:
   s=read_daily_status(str(w),'20260827');assert s['nodes'][node]['state']=='STALE' and 'DAILY_STATUS_OUTPUT_SHA_MISMATCH' in repr(s['nodes'][node])
   assert all(s['nodes'][n]['state']=='SUCCEEDED' for n in EOD_NODE_IDS if n!=node)
   try:replay_native_completion(workspace=str(w),trade_date='20260827',completion_ref=ref)
   except ContractError as e:assert str(e)=='EOD_READBACK_TERMINAL_CHANGED:'+node
   else:raise AssertionError('corrupt input admitted')
  assert counts['restored'] and counts['target_redirects']>0 and counts['successful_physical_reads']==counts['target_redirects']
  assert h.file_identity(w/output['path'])==original
  cases.append({'node':node,'routing':counts})
print(json.dumps({'state':'PREFLIGHT_PASS','fresh_positive_full_native_replay_performed':False,'case8_acceptance_claimed':False,'cases':cases}))
'''
env=dict(os.environ);env.pop('PYTHONPATH',None)
r=subprocess.run([receipt['python'],'-I','-B','-c',program,str(root),str(Path('.agent/acceptance/current_source_mismatch_reader_case.py').resolve())],cwd=root,env=env,capture_output=True,text=True)
(root/'case8-mechanical-preflight.log').write_text(r.stdout+r.stderr)
print(r.stdout);print(r.stderr[-2400:]);raise SystemExit(r.returncode)
