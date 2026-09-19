"""Independent synthetic baseline, using the already verified frozen installation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

original = Path('/private/tmp/myquant-final-source-native-20260909T030031Z')
receipt = json.loads((original / 'fixture-receipt.json').read_bytes())
case = Path('/private/tmp/myquant-public-history-native-20260909T031046Z')
assert not (case / 'factor-workspace').exists()
assert (case / 'repository').resolve(strict=True) == case / 'repository'
assert subprocess.check_output(['git', '-C', str(case / 'repository'), 'rev-parse', 'HEAD'], text=True).strip() == receipt['commit']
metadata = {'root': str(case), 'producer_commit': receipt['commit'], 'installation_root': str(original),
    'canonical_repository': str((case / 'repository').resolve(strict=True)),
    'synthetic': True, 'status': 'BASELINE_RUNNING', 'prior_preparation_failure_ref': str(case / 'preparation-failure.v1.json'),
    'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(case / 'historical-case-status.json').write_text(json.dumps(metadata, indent=2) + '\n')
Path('/Users/maxwell/mySpace/myQuant/.agent/acceptance/historical-run-location.json').write_text(json.dumps(metadata, indent=2) + '\n')
probe = subprocess.run([receipt['python'], '-I', '-c',
    'import json,sys; from pathlib import Path; from quant_investor.system.release_install import verify_running_release_install_input; print(json.dumps(verify_running_release_install_input(Path(sys.argv[1]).read_bytes(), repository_root=Path(sys.argv[2]))))',
    str(case / 'release-input.json'), str(case / 'repository')], cwd=case, check=True, capture_output=True, text=True)
verification = json.loads(probe.stdout)
assert verification['state'] == 'PASS'
(case / 'reused-install-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
program = '''import runpy, socket, sys
original_connect = socket.socket.connect
def connect(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise RuntimeError('SYNTHETIC_ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')
    return original_connect(sock, address)
socket.socket.connect = connect
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''
args = [receipt['python'], '-I', '-c', program,
    str(case / 'repository/tests/unit/_native_full_dag_scenario.py'), str(case),
    '/Users/maxwell/mySpace/myQuant/.venv/lib/python3.13/site-packages', 'future-calendar']
env = dict(os.environ)
env.pop('PYTHONPATH', None)
print('START historical case baseline', case, flush=True)
with (case / 'baseline.log').open('w') as stream:
    process = subprocess.Popen(args, cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT)
    metadata['child_pid'] = process.pid
    (case / 'historical-case-status.json').write_text(json.dumps(metadata, indent=2) + '\n')
    code = process.wait()
metadata.update(status='BASELINE_COMPLETE' if code == 0 else 'BASELINE_FAILED', exit_code=code)
(case / 'historical-case-status.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(metadata['status'], 'exit', code, flush=True)
raise SystemExit(code)
