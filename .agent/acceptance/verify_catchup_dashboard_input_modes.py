"""Read retained incident inputs using a separately verified candidate installation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import quant_investor
from quant_investor.operations.catchup_binding import BindingSources, check_template_sources
from quant_investor.operations.daily_preparation import Sources
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.execution_recipe import validate_catchup_template
from quant_investor.system.errors import SystemSecurityError, SystemPreconditionError
from quant_investor.system.release_install import verify_running_release_install_input

candidate, old, diagnosis_path = map(Path, sys.argv[1:4])
receipt = json.loads((candidate / 'fixture-receipt.json').read_bytes())
verified = verify_running_release_install_input(
    (candidate / 'release-input.json').read_bytes(), repository_root=candidate / 'repository')
assert verified['state'] == 'PASS'
assert Path(quant_investor.__file__).resolve() == Path(verified['import_origin']).resolve()
sys.path.extend([str(candidate / 'repository'), str(candidate / 'repository/scripts')])
workspace = old / 'factor-workspace'
diag = json.loads(diagnosis_path.read_bytes())


def snapshot(paths):
    values = {}
    for relative in sorted(paths):
        path = workspace / relative
        s = path.stat()
        values[relative] = (hashlib.sha256(path.read_bytes()).hexdigest(), s.st_dev, s.st_ino,
                            s.st_mode, s.st_uid, s.st_gid, s.st_nlink, s.st_size,
                            s.st_mtime_ns, s.st_ctime_ns)
    return values


def refs(value):
    if type(value) is dict:
        if set(value) == {'path', 'sha256'}:
            yield value['path']
        else:
            for child in value.values():
                yield from refs(child)
    elif type(value) is list:
        for child in value:
            yield from refs(child)


source = BindingSources(str(workspace))
request = source.document(diag['request_ref'])
collection = source.document(request['recipe_ref'])
paths = set(refs(collection)) | {diag['request_ref']['path'], request['recipe_ref']['path']}
before = snapshot(paths)
status_before = read_daily_status(str(workspace), '20260828')
assert all(row['state'] == 'NOT_STARTED' for row in status_before['nodes'].values())
for day, template in collection['recipes'].items():
    validate_catchup_template(template, request={**request, 'action': 'EXECUTE', 'target_trade_date': day})
    check_template_sources(source, template, profile='FRESH')
source.recheck()
(candidate / 'retained-template-reader-checkpoint.json').write_text(json.dumps({
    'state': 'NEW_INSTALLED_EXACT_RETAINED_TEMPLATE_READ_PASS',
    'candidate_commit': receipt['commit'], 'request_ref': diag['request_ref'],
    'source_count': len(paths), 'original_source_snapshot': before,
    'full_execution_claimed': False,
}, indent=2) + '\n')

# Exercise the installed launcher's actual static producer-control preflight too.
# No Calendar/network/producer dispatch or historical EOD replay is invoked here.
from scripts.daily_launch_inspection import _producer_controls
try:
    _producer_controls({'sources': source, 'resolution': {'derived_collection': collection}}, list(collection['recipes']))
except SystemPreconditionError as exc:
    if str(exc) != 'SYSTEM_PRECONDITION_FAILED:current process is not running the installed release':
        raise
    old_release_gate = 'EXPECTED_OLD_REQUEST_CURRENT_INSTALL_ORIGIN_MISMATCH'
else:
    raise AssertionError('old request admitted under a different installed release')

strict_refusals = []
preparation = Sources(str(workspace))
for row in diag['sources']:
    reference = row['ref']
    assert hashlib.sha256(preparation.raw(reference)).hexdigest() == reference['sha256']
    try:
        BindingSources(str(workspace)).raw(reference)
    except SystemSecurityError:
        strict_refusals.append(row['role'])
    else:
        raise AssertionError('governance reader was broadened')
preparation.recheck()
assert snapshot(paths) == before
assert read_daily_status(str(workspace), '20260828') == status_before
assert len(strict_refusals) == 2
report = {
    'observed_at': datetime.now(timezone.utc).isoformat(),
    'state': 'NEW_INSTALLED_EXACT_RETAINED_INPUT_PREFLIGHT_PASS',
    'candidate_root': str(candidate), 'candidate_commit': receipt['commit'],
    'import_origin': verified['import_origin'], 'installed_verification': verified,
    'original_root': str(old), 'original_runtime_commit': diag['runtime_commit'],
    'request_ref': diag['request_ref'], 'collection_ref': request['recipe_ref'],
    'template_dates': list(collection['recipes']), 'read_source_count': len(paths),
    'dashboard_roles': strict_refusals, 'strict_governance_reader_still_refuses_0644': True,
    'original_bytes_identity_modes_times_unchanged': True,
    'target_business_nodes_still_not_started': True,
    'old_release_launch_gate': old_release_gate,
    'new_release_requires_new_uninterrupted_fixture': True,
    'full_automatic_admission_or_eod_replay': False,
    'original_request_resealed': False, 'old_installation_hotpatched': False,
    'production_deployed': False, 'full_goal_complete': False,
}
print(json.dumps(report, indent=2))
