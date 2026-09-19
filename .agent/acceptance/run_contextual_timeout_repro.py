"""One diagnostic replay of the exact failed frozen test; no timeout relaxation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path('/Users/maxwell/mySpace/myQuant')
frozen = Path('/private/tmp/myquant-future-calendar-scoped-clock-20260916T125157Z/repository')
root = Path('/private/tmp/myquant-contextual-timeout-repro-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
root.mkdir(mode=0o700, exist_ok=False)
target = 'tests/unit/test_unified_factor_store_contextual.py::test_full_prospective_store_context_replays_1442_raw_sources_without_activation'
program = '''import faulthandler,os,sys
from pathlib import Path
frozen,root,target=sys.argv[1:]
os.chdir(frozen)
sys.path.insert(0,frozen)
trace=open(Path(root)/"fork-worker-trace.log","a")
def child_trace():
 faulthandler.dump_traceback_later(120,file=trace,repeat=False)
os.register_at_fork(after_in_child=child_trace)
import pytest
raise SystemExit(pytest.main([target,"-q","-ra","-p","no:cacheprovider","--basetemp",str(Path(root)/"pytest-retained")]))
'''
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(frozen))
status = {'state':'RUNNING','root':str(root),'frozen_repository':str(frozen),'target':target,
          'scope':'ISOLATED_DIAGNOSTIC_WITH_120_SECOND_CHILD_STACK_DUMP',
          'native_timeout_seconds':180,'timeout_changed':False,'automatic_retry':False,
          'started_at':datetime.now(timezone.utc).isoformat()}
with (root/'pytest.log').open('w') as output:
 child = subprocess.Popen([str(source/'.venv/bin/python'),'-c',program,str(frozen),str(root),target],cwd=frozen,env=env,stdout=output,stderr=subprocess.STDOUT)
 status['pid']=child.pid
 (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 (source/'.agent/acceptance/phase15-contextual-timeout-repro-location.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status),flush=True)
 code=child.wait()
status.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256((root/'pytest.log').read_bytes()).hexdigest())
(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
raise SystemExit(code)
