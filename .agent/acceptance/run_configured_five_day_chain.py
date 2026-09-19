"""Finite installed acceptance: original first day, then existing successor driver."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source = Path('/Users/maxwell/mySpace/myQuant')
root = Path(sys.argv[1]).resolve(strict=True)
status_path = root / 'configured-chain-status.json'
if status_path.exists() or (root / 'configured-driver-status.json').exists():
    raise ValueError('chain already started; inspect retained status instead of restarting')
receipt = json.loads((root / 'fixture-receipt.json').read_bytes())
driver_root = root / 'acceptance-drivers'
driver_root.mkdir(mode=0o700)
names = ('run_configured_native_acceptance.py', 'run_configured_successors.py')
pins = {}
for name in names:
    raw = (source / '.agent/acceptance' / name).read_bytes()
    compile(raw, name, 'exec')
    (driver_root / name).write_bytes(raw)
    (driver_root / name).chmod(0o600)
    pins[name] = hashlib.sha256(raw).hexdigest()
(root / 'acceptance-driver-manifest.json').write_text(json.dumps(pins, indent=2) + '\n')
state = {'state': 'RUNNING', 'root': str(root), 'runtime_commit': receipt['commit'],
         'started_at': datetime.now(timezone.utc).isoformat(), 'steps': [],
         'synthetic': True, 'production_deployed': False, 'automatic_retry': False}
for name in names:
    if hashlib.sha256((driver_root / name).read_bytes()).hexdigest() != pins[name]:
        raise ValueError('pinned acceptance driver changed')
    state['active_driver'] = name
    with (root / (name + '.log')).open('w') as output:
        child = subprocess.Popen([sys.executable, str(driver_root / name), str(root)],
                                 cwd=source, stdout=output, stderr=subprocess.STDOUT)
        state['driver_pid'] = child.pid
        status_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        code = child.wait()
    state['steps'].append({'driver': name, 'exit_code': code})
    if code:
        state['state'] = 'FAIL'
        break
else:
    state['state'] = 'PASS'
state.pop('active_driver', None)
state.pop('driver_pid', None)
state['finished_at'] = datetime.now(timezone.utc).isoformat()
status_path.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state), flush=True)
raise SystemExit(0 if state['state'] == 'PASS' else 1)
