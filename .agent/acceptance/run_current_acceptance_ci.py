"""Freeze current source and run full gates in that isolated repository once."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path('/Users/maxwell/mySpace/myQuant')
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ROOT = Path(sys.argv[1]).resolve(strict=True) if len(sys.argv) > 1 else Path('/private/tmp') / ('myquant-current-acceptance-' + STAMP)
LOCATION = SOURCE / '.agent/acceptance/phase15-current-acceptance-location.json'
LOCATION.write_text(json.dumps({'root': str(ROOT), 'state': 'PREPARING'}) + '\n')
sys.path.insert(0, str(SOURCE / 'tests/unit'))
from _native_daily_release_fixture import prepare_synthetic_release, git

if 'UV_CACHE_DIR' not in os.environ:
 os.environ['UV_CACHE_DIR'] = subprocess.check_output(['uv', 'cache', 'dir'], text=True).strip()

def resume_install():
 from quant_investor.contracts import canonical_json_bytes
 from quant_investor.system.release_install import prepare_operational_release

 if ROOT.parent != Path('/private/tmp') or not ROOT.name.startswith('myquant-current-acceptance-') or (ROOT/'ci-status.json').exists() or (ROOT/'fixture-receipt.json').exists():
  raise ValueError('resume is limited to the known failed pre-CI installation')
 repository = ROOT/'repository'
 commit = git(repository, 'rev-parse', 'HEAD')
 prepared = prepare_operational_release(repository_root=repository, release_root=ROOT/'release', final_commit=commit, final_tree=git(repository,'rev-parse','HEAD^{tree}'), created_at=None)
 raw = canonical_json_bytes({'deployed_release':prepared['release'], 'release_install_evidence':prepared['release_install_evidence']})
 reference = ROOT/'release-input.json'
 reference.write_bytes(raw)
 reference.chmod(0o600)
 python = prepared['release_install_evidence']['payload']['python_executable']
 program = 'import json,sys; from pathlib import Path; from quant_investor.system.release_install import verify_running_release_install_input; print(json.dumps(verify_running_release_install_input(Path(sys.argv[1]).read_bytes(),repository_root=Path(sys.argv[2]))))'
 environment = dict(os.environ)
 environment.pop('PYTHONPATH',None)
 verified = json.loads(subprocess.check_output([python,'-I','-c',program,str(reference),str(repository)],cwd=ROOT,env=environment,text=True))
 if verified['state'] != 'PASS': raise ValueError('installed runtime verification failed')
 receipt = {'synthetic_test_environment':True,'repository':str(repository),'release_root':str(ROOT/'release'),'release_input':str(reference),'python':python,'commit':commit,'runtime_verification':verified,'full_dag_proof':False,'production_deployed':False}
 (ROOT/'fixture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

receipt = resume_install() if len(sys.argv) > 1 else prepare_synthetic_release(SOURCE, ROOT)
repository = ROOT / 'repository'
python = str(SOURCE / '.venv/bin/python')
env = dict(os.environ)
env['PYTHONPATH'] = str(repository)
files = [p for base in ('quant_investor', 'scripts', 'tests') for p in (repository / base).rglob('*') if p.is_file() and p.suffix == '.py']
files += [repository / p for p in ('pyproject.toml', 'uv.lock', '.github/workflows/ci-cd.yml')]
inventory = {p.relative_to(repository).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT / 'ci-source-freeze.json').write_text(json.dumps(inventory, indent=2) + '\n')
scope = ['quant_investor/contracts', 'quant_investor/system', 'quant_investor/factors/governance', 'quant_investor/intelligence', 'quant_investor/mainline', 'quant_investor/cli']
commands = [
 ('unit', [python, '-m', 'pytest', 'tests/unit', '-q', '-ra']),
 ('fatal-lint', [python, '-m', 'flake8', 'quant_investor', '--select=E9,F63,F7,F82']),
 ('stable-lint', [python, '-m', 'flake8', *scope, '--max-complexity=10', '--max-line-length=100']),
 ('stable-black', [python, '-m', 'black', '--check', *scope]),
 ('stable-mypy', [python, '-m', 'mypy', *scope, '--ignore-missing-imports']),
]
state = {'root': str(ROOT), 'repository': str(repository), 'producer_commit': receipt['commit'], 'state': 'RUNNING', 'steps': [], 'production_deployed': False}
status = ROOT / 'ci-status.json'
for label, command in commands:
 state['active_step'] = label
 status.write_text(json.dumps(state, indent=2) + '\n')
 print('START', label, str(ROOT), flush=True)
 before = time.monotonic()
 with (ROOT / (label + '.log')).open('w') as stream:
  child = subprocess.Popen(command, cwd=repository, env=env, stdout=stream, stderr=subprocess.STDOUT)
  state['child_pid'] = child.pid
  status.write_text(json.dumps(state, indent=2) + '\n')
  code = child.wait()
 state['steps'].append({'step': label, 'command': command, 'exit_code': code, 'seconds': round(time.monotonic()-before, 2), 'log': str(ROOT / (label + '.log'))})
 print('EXIT', label, code, flush=True)
state.pop('active_step', None)
state.pop('child_pid', None)
drift = [name for name, digest in inventory.items() if not (repository/name).is_file() or hashlib.sha256((repository/name).read_bytes()).hexdigest() != digest]
state['source_drift'] = drift
state['state'] = 'PASS' if not drift and all(row['exit_code'] == 0 for row in state['steps']) else 'FAIL'
status.write_text(json.dumps(state, indent=2) + '\n')
LOCATION.write_text(json.dumps({'root': str(ROOT), 'state': state['state'], 'status': str(status)}) + '\n')
print(json.dumps(state, indent=2), flush=True)
raise SystemExit(0 if state['state'] == 'PASS' else 1)
