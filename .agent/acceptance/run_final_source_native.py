"""Run the existing installed native five-day scenario once, retaining every exit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

SOURCE = Path('/Users/maxwell/mySpace/myQuant')
ROOT = Path('/private/tmp') / ('myquant-final-source-native-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
(SOURCE / '.agent/acceptance/native-run-location.json').write_text(json.dumps({'root': str(ROOT), 'status': 'PREPARING'}) + '\n')
sys.path.insert(0, str(SOURCE / 'tests/unit'))
from _native_daily_release_fixture import prepare_synthetic_release

try:
    receipt = prepare_synthetic_release(SOURCE, ROOT)
    dependency = str(SOURCE / '.venv/lib/python3.13/site-packages')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    bootstrap = '''import runpy, socket, sys
_original_connect = socket.socket.connect
def guarded_connect(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
    return _original_connect(sock, address)
socket.socket.connect = guarded_connect
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''
    programs = [('_native_full_dag_scenario.py', [str(ROOT), dependency, 'future-calendar'], '20260827')]
    programs += [('_native_full_successor.py', [str(ROOT), dependency, str(ROOT / 'repository/tests/unit'), day, 'future-calendar'], day)
                 for day in ('20260828', '20260831', '20260901', '20260902')]
    programs += [('_verify_five_native_days.py', [str(ROOT), 'v2', dependency], 'five-day-replay')]
    progress = {'root': str(ROOT), 'producer_commit': receipt['commit'], 'status': 'RUNNING', 'steps': []}
    for filename, args, label in programs:
        progress['active_step'] = label
        (ROOT / 'run-status.json').write_text(json.dumps(progress, indent=2) + '\n')
        print('START', label, str(ROOT), flush=True)
        script = ROOT / 'repository/tests/unit' / filename
        with (ROOT / (label + '.log')).open('w') as stream:
            child = subprocess.Popen([receipt['python'], '-I', '-c', bootstrap, str(script), *args],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            progress['child_pid'] = child.pid
            (ROOT / 'run-status.json').write_text(json.dumps(progress, indent=2) + '\n')
            code = child.wait()
        progress['steps'].append({'step': label, 'exit_code': code})
        if code:
            progress['status'] = 'FAILED'
            (ROOT / 'run-status.json').write_text(json.dumps(progress, indent=2) + '\n')
            raise RuntimeError('native step failed: ' + label)
        print('PASS', label, flush=True)
    progress['status'] = 'COMPLETE'
    progress.pop('active_step', None)
    progress.pop('child_pid', None)
    (ROOT / 'run-status.json').write_text(json.dumps(progress, indent=2) + '\n')
except BaseException:
    traceback.print_exc()
    if ROOT.exists():
        (ROOT / 'driver-exit.json').write_text(json.dumps({'exit_code': 1}) + '\n')
    raise
else:
    (ROOT / 'driver-exit.json').write_text(json.dumps({'exit_code': 0}) + '\n')
