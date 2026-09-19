"""One isolated installed missing-prior-Store scenario; preserve any failed stage."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path('/Users/maxwell/mySpace/myQuant')
root = Path(sys.argv[1]).resolve(strict=True)
receipt = json.loads((root / 'fixture-receipt.json').read_bytes())
resume_retained = len(sys.argv) == 3 and sys.argv[2] == 'resume-retained'
if len(sys.argv) > 3 or (len(sys.argv) == 3 and not resume_retained):
    raise ValueError('unknown missing-Store acceptance mode')
prefix = 'missing-store-resume' if resume_retained else 'missing-store'
status_path = root / (prefix + '-driver-status.json')
if status_path.exists() or ((root / 'factor-workspace').exists() and not resume_retained):
    raise ValueError('missing-Store scenario already attempted; inspect original retained state')
if resume_retained:
    previous = json.loads((root / 'missing-store-driver-status.json').read_bytes())
    if previous['state'] != 'FAIL' or previous['helper_drift']:
        raise ValueError('original stopped scenario with unchanged pinned helpers required')
drift = []
for directory in ('quant_investor', 'scripts'):
    for path in (source / directory).rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        frozen = root / 'repository' / path.relative_to(source)
        if not frozen.is_file() or frozen.read_bytes() != path.read_bytes():
            drift.append(str(path.relative_to(source)))
if drift:
    raise ValueError('business runtime differs from installed snapshot: ' + repr(drift))
helpers = root / ('recovery-validation-helpers' if resume_retained else 'validation-helpers')
helpers.mkdir(mode=0o700)
names = ('_native_missing_store_case.py', '_native_configured_daily_scenario.py',
         '_native_full_dag_scenario.py', '_native_daily_maintenance_fixture.py',
         '_native_synthetic_clock.py')
if resume_retained:
    names = ('_native_missing_store_case.py', '_native_synthetic_clock.py')
pins = {}
for name in names:
    raw = (source / 'tests/unit' / name).read_bytes()
    (helpers / name).write_bytes(raw)
    (helpers / name).chmod(0o600)
    pins[name] = hashlib.sha256(raw).hexdigest()
(root / ('recovery-validation-helper-manifest.json' if resume_retained
         else 'validation-helper-manifest.json')).write_text(json.dumps({
    'helper_sha256': pins, 'runtime_commit': receipt['commit'],
    'runtime_differences': drift}, indent=2) + '\n')
program = '''import socket,sys,os,signal,faulthandler,json
from pathlib import Path
import quant_investor
r=Path(sys.argv[1]);resume=sys.argv[3]=='resume-retained'
sys.path[:0]=[str(r/('recovery-validation-helpers' if resume else 'validation-helpers'))]
sys.path.extend([str(r/'repository'),str(r/'repository/scripts'),str(r/'repository/tests/unit'),sys.argv[2]])
def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
socket.socket.connect=forbidden
prefix='missing-store-resume' if resume else 'missing-store'
trace=(r/(prefix+'.stack.txt')).open('a');os.chmod(trace.name,0o600)
faulthandler.register(signal.SIGUSR1,file=trace,all_threads=True)
(r/(prefix+'-trace-ready.json')).write_text(json.dumps({'pid':os.getpid(),'signal':'SIGUSR1','trace':trace.name})+chr(10))
from _native_missing_store_case import dispatch_case
if resume:
    source=json.loads((r/'configured-source-result.json').read_bytes())
    dispatch_case(root=r,workspace=r/'factor-workspace',request_ref=source['request_ref'],
                  config_ref=source['config_ref'],release_install_ref=source['release_install_ref'],
                  resume_retained=True)
else:
    from _native_configured_daily_scenario import run
    run(r,Path(sys.argv[2]),dispatch_case=dispatch_case)
'''
state = {'state': 'RUNNING', 'case': 4, 'root': str(root),
         'producer_commit': receipt['commit'],
         'started_at': datetime.now(timezone.utc).isoformat(), 'production_deployed': False,
         'continued_from_retained_test_failure': resume_retained}
environment = dict(os.environ)
environment.pop('PYTHONPATH', None)
with (root / (prefix + '-native.log')).open('w') as log:
    child = subprocess.Popen([receipt['python'], '-I', '-c', program, str(root),
                              str(source / '.venv/lib/python3.13/site-packages'),
                              'resume-retained' if resume_retained else 'fresh'],
                             cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = child.pid
    status_path.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)
    code = child.wait()
state.update(state='PASS' if code == 0 else 'FAIL', exit_code=code,
             finished_at=datetime.now(timezone.utc).isoformat())
state.pop('child_pid', None)
state['helper_drift'] = [name for name, sha in pins.items()
                         if hashlib.sha256((helpers / name).read_bytes()).hexdigest() != sha]
if state['helper_drift']:
    state['state'] = 'FAIL'
status_path.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state), flush=True)
raise SystemExit(0 if state['state'] == 'PASS' else 1)
