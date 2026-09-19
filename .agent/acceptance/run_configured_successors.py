"""Bounded four-day continuation and native replay of the current installed scenario."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path('/Users/maxwell/mySpace/myQuant')
ROOT = Path(sys.argv[1]).resolve(strict=True)
resume_prepared = len(sys.argv) == 3 and sys.argv[2] == 'resume-prepared'
if len(sys.argv) > 3 or (len(sys.argv) == 3 and not resume_prepared):
    raise ValueError('unknown successor acceptance mode')
receipt = json.loads((ROOT / 'fixture-receipt.json').read_bytes())
initial_status = json.loads((ROOT / 'configured-driver-status.json').read_bytes())
if initial_status['state'] != 'PASS' or initial_status['exit_code'] != 0:
    raise ValueError('the original configured native process must finish successfully first')
if not (ROOT / 'configured-native-proof.json').is_file():
    raise ValueError('the original native completed proof is required')
status_path = ROOT / ('configured-successors-resume-status.json' if resume_prepared else 'configured-successors-status.json')
if status_path.exists():
    raise ValueError('continuation was already started; inspect its exact retained stage')
previous_failure_ref = None
if resume_prepared:
    previous_path = ROOT / 'configured-successors-status.json'
    previous_raw = previous_path.read_bytes()
    previous = json.loads(previous_raw)
    if (previous['state'] != 'FAIL' or previous['helper_drift']
        or [row['stage'] for row in previous['steps']] != ['repeat', '20260828']
        or [row['exit_code'] for row in previous['steps']] != [0, 1]):
        raise ValueError('exact retained preparation-stage stop required')
    previous_failure_ref = {'path': str(previous_path), 'sha256': hashlib.sha256(previous_raw).hexdigest()}
repository = ROOT / 'repository'
runtime = [p for d in ('quant_investor', 'scripts') for p in (SOURCE / d).rglob('*.py') if '__pycache__' not in p.parts]
drift = [str(p.relative_to(SOURCE)) for p in runtime if not (repository / p.relative_to(SOURCE)).is_file() or p.read_bytes() != (repository / p.relative_to(SOURCE)).read_bytes()]
if drift:
    raise ValueError('current runtime differs from the verified installation: ' + repr(drift))
helpers = ROOT / ('successor-recovery-validation-helpers' if resume_prepared else 'successor-validation-helpers')
helpers.mkdir(mode=0o700)
names = ('_native_unresolved_corporate_case.py', '_native_top100_recovery_case.py', '_native_configured_boundary_checks.py', '_native_synthetic_clock.py', '_native_configured_successor.py', '_native_daily_store_fixture.py', '_native_daily_maintenance_fixture.py', '_native_morning_replay_fixture.py', '_verify_five_native_days.py')
pins = {}
for name in names:
    raw = (SOURCE / 'tests/unit' / name).read_bytes()
    (helpers / name).write_bytes(raw)
    (helpers / name).chmod(0o600)
    pins[name] = hashlib.sha256(raw).hexdigest()
(ROOT / ('successor-recovery-validation-helper-manifest.json' if resume_prepared else 'successor-validation-helper-manifest.json')).write_text(json.dumps({'helper_sha256': pins, 'runtime_commit': receipt['commit'], 'runtime_differences': drift, 'separately_pinned_test_helpers': True}, indent=2) + '\n')
program = '''import socket,sys
from pathlib import Path
import quant_investor
original=socket.socket.connect
def guard(sock,address):
    if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
        raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
    return original(sock,address)
socket.socket.connect=guard
sys.path.insert(0,sys.argv[1])
root=Path(sys.argv[2]); dependencies=Path(sys.argv[3]);stage=sys.argv[4];resume=sys.argv[5]=='resume-prepared'
import faulthandler,signal,os
prefix='configured-successor-resumed-' if resume else 'configured-successor-'
trace=(root/(prefix+stage+'.stack.txt')).open('a')
os.chmod(trace.name,0o600)
faulthandler.register(signal.SIGUSR1,file=trace,all_threads=True)
(root/(prefix+stage+'-trace-ready.json')).write_text(__import__('json').dumps({'pid':os.getpid(),'signal':'SIGUSR1','trace':trace.name})+chr(10))
sys.path.extend([str(root/'repository'),str(root/'repository/scripts'),str(root/'repository/tests/unit'),str(dependencies)])
import json,hashlib
if stage in ('repeat','closed'):
    from _native_configured_boundary_checks import run
    print(json.dumps(run(root,dependencies,stage),indent=2))
elif stage=='top100-recovery':
    from _native_configured_successor import run
    print(json.dumps(run(root,dependencies,'20260903',interrupt_top100=True,unresolved_corporate=True),indent=2))
elif stage.startswith('2026'):
    if resume and stage=='20260828':
        from _native_configured_successor import run_prepared
        print(json.dumps(run_prepared(root,dependencies,stage),indent=2))
    else:
        from _native_configured_successor import run
        print(json.dumps(run(root,dependencies,stage),indent=2))
else:
    from _native_configured_successor import SEQUENCE,proof_path
    refs={d:json.loads(proof_path(root,d).read_bytes())['completion_ref'] for d in SEQUENCE}
    if stage=='five-day-replay':
        from _verify_five_native_days import run
        from quant_investor.operations.native_bridge import verified_native_context
        raw=(root/'release-input.json').read_bytes()
        with verified_native_context(release_input_raw=raw,expected_sha256=hashlib.sha256(raw).hexdigest(),repository_root=str(root/'repository')) as operations:
            proof=run(root,replay_native_completion=operations['completion_replay'],v2=True,completion_refs=refs)
        configs=[json.loads(proof_path(root,d).read_bytes())['config_ref'] for d in SEQUENCE]
        assert all(c==configs[0] for c in configs)
        proof.update(unchanged_configured_entry=True,config_ref=configs[0],acceptance_harness_interruption_disclosed=resume,continuous_uninterrupted_harness_claimed=not resume)
        (root/'configured-five-native-trading-day-proof.json').write_text(json.dumps(proof,indent=2)+chr(10))
        print('PASS five configured days and native readonly replay',flush=True)
    elif stage=='morning':
        from _native_morning_replay_fixture import run
        run(root,dependencies,trade_date=SEQUENCE[-1],completion_ref=refs[SEQUENCE[-1]])
    else:
        raise ValueError('unknown bounded acceptance stage')
'''
state = {'state': 'RUNNING', 'root': str(ROOT), 'producer_commit': receipt['commit'], 'started_at': datetime.now(timezone.utc).isoformat(), 'steps': [], 'production_deployed': False, 'continued_from_preparation_test_failure': resume_prepared, 'previous_failure_ref': previous_failure_ref}
environment = dict(os.environ)
environment.pop('PYTHONPATH', None)
stages = ('20260828', '20260831', '20260901', '20260902', 'five-day-replay', 'morning', 'top100-recovery', 'closed')
if not resume_prepared:
    stages = ('repeat', *stages)
for stage in stages:
    changed = [n for n, digest in pins.items() if hashlib.sha256((helpers / n).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError('pinned helper changed before stage: ' + repr(changed))
    log = ROOT / (('configured-successor-resumed-' if resume_prepared else 'configured-successor-') + stage + '.log')
    started = time.monotonic()
    with log.open('w') as stream:
        child = subprocess.Popen([receipt['python'], '-I', '-c', program, str(helpers), str(ROOT), str(SOURCE / '.venv/lib/python3.13/site-packages'), stage, 'resume-prepared' if resume_prepared else 'fresh'], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        state.update(active_step=stage, child_pid=child.pid)
        status_path.write_text(json.dumps(state, indent=2) + '\n')
        print('START', stage, 'pid', child.pid, flush=True)
        code = child.wait()
    state['steps'].append({'stage': stage, 'exit_code': code, 'seconds': round(time.monotonic()-started, 2), 'log': str(log)})
    print('EXIT', stage, code, flush=True)
    if code:
        state['state'] = 'FAIL'
        break
else:
    state['state'] = 'PASS'
state.pop('active_step', None)
state.pop('child_pid', None)
state['helper_drift'] = [n for n, digest in pins.items() if hashlib.sha256((helpers / n).read_bytes()).hexdigest() != digest]
if state['helper_drift']:
    state['state'] = 'FAIL'
state['finished_at'] = datetime.now(timezone.utc).isoformat()
status_path.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state, indent=2), flush=True)
raise SystemExit(0 if state['state'] == 'PASS' else 1)
