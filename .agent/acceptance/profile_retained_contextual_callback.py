"""Profile retained synthetic callback inputs under the unchanged native worker bound.

Diagnostic only: no validation receipt, activation or completed-test claim.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

frozen=Path('/private/tmp/myquant-future-calendar-scoped-clock-20260916T125157Z/repository')
root=Path('/private/tmp/myquant-contextual-timeout-repro-20260916T144304Z')
output=root/'retained-callback-profile'
output.mkdir(mode=0o700,exist_ok=False)
sys.path.insert(0,str(frozen))
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.store import SystemStore
from quant_investor.system.validation import _run_callback_worker,MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS
from quant_investor.factors.governance.contextual import validate_prospective_contextual_run
import cProfile
import signal

fixture=root/'pytest-retained/test_full_prospective_store_co0'
workspace=fixture/'workspace'
request_path=workspace/'results/system/objects/system.validation_run_request/a4d10ebcc757c4610c603b8a98f7207a75e9cbdf9a457eee11b418b8f54db048.json'
request=parse_canonical_json_bytes(request_path.read_bytes())
store=SystemStore(workspace,source_root=fixture/'source',source_root_id='factor-full-prospective-source-root')
state={'state':'RUNNING','pid':os.getpid(),'scope':'RETAINED_SYNTHETIC_CALLBACK_PROFILING_NOT_ACCEPTANCE',
       'native_limit_seconds':MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS,'timeout_changed':False,
       'started_at':datetime.now(timezone.utc).isoformat(),'profile_overhead_disclosed':True}
(output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
def profiled(**kwargs):
 profiler=cProfile.Profile()
 def save(*args):
  profiler.disable();profiler.dump_stats(str(output/'callback.prof'));profiler.enable()
 signal.signal(signal.SIGALRM,save)
 signal.setitimer(signal.ITIMER_REAL,90,60)
 profiler.enable()
 try:return validate_prospective_contextual_run(**kwargs)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  profiler.disable();profiler.dump_stats(str(output/'callback.prof'))
try:
 result,peak=_run_callback_worker(profiled,store=store,validation_request=request,
   trusted_at=request['created_at'],maximum_seconds=MAXIMUM_PROSPECTIVE_VALIDATION_SECONDS)
 state.update(state='CALLBACK_RETURNED',peak_rss=peak,result_lane=result.get('lane'))
except Exception as exc:
 state.update(state='DIAGNOSTIC_TERMINAL_ERROR',error_type=type(exc).__name__,error=str(exc))
state['finished_at']=datetime.now(timezone.utc).isoformat()
(output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
