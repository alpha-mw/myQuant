"""Immutable sibling receipt of installed Calendar transport observations.

Readers replay captured native custody and the release wheel; recovery performs
no capture, credential access, recorder activation or network operation.
"""

from datetime import datetime, timezone, timedelta
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from quant_investor.operations.journal_revisions import selected_binding
from quant_investor.system.errors import SystemSecurityError
from ._calendar_production_transport import _MODULE_PATHS
from .next_session_calendar import inspect_next_session_capture, HORIZON_NATURAL_DAYS

SCHEMA = "cn-calendar-production-transport.v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "eod_trade_date",
        "release_install_input_ref",
        "release_ref",
        "execution_ref",
        "success_ref",
        "capture_root_name",
        "recorder_identity",
        "events",
        "event_set_sha256",
        "recorded_at",
        "authority",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "kind",
        "method",
        "scheme",
        "host",
        "port",
        "path",
        "api_name",
        "parameters",
        "expected_fields",
        "request_started_at",
        "response_completed_at",
        "response_bytes",
        "response_sha256",
        "http_status",
        "tls_verified",
        "redirect_count",
    }
)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(code="NEXT_SESSION_TRANSPORT_BINDING_INVALID"):
    raise SystemSecurityError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def transport_evidence_path(journal, execution_ref):
    validate_ref({"path": execution_ref["relative_path"], "sha256": execution_ref["byte_sha256"]})
    return str(
        journal.root
        / "calendar-future/production-transport"
        / (execution_ref["byte_sha256"] + ".json")
    )


def _native_raw(journal, info, ref):
    if type(ref) is not dict or set(ref) != {"relative_path", "byte_sha256"}:
        _fail()
    validate_ref({"path": ref["relative_path"], "sha256": ref["byte_sha256"]})
    parts = Path(ref["relative_path"]).parts
    if len(parts) != 2 or parts[0] != info["execution"]["payload"]["capture_root_name"]:
        _fail()
    stored = journal.storage.read(
        str(journal.root / "calendar-future/captures" / ref["relative_path"])
    )
    if stored is None or stored.byte_sha256 != ref["byte_sha256"]:
        _fail()
    return stored.data


def _recorder_identity(journal, info):
    """Replay exact installed module hashes from the original release's wheel."""
    from quant_investor.system.release_install import _regular_file, _code_file, _manifest_sha
    from .tushare_calendar_authority import _release_install_components

    execution = info["execution"]["payload"]
    raw = _native_raw(journal, info, execution["release_install_input_file_ref"])
    evidence, release, _ = _release_install_components(raw, repository_root="", historical=True)
    payload = evidence["payload"]
    wheel = payload["wheel"]
    wheel_raw, metadata = _regular_file(
        Path(wheel["path"]), label="Calendar recorder release wheel"
    )
    if (
        len(wheel_raw) != wheel["size"]
        or metadata.st_size != wheel["size"]
        or _sha(wheel_raw) != wheel["byte_sha256"]
    ):
        _fail()
    files = []
    try:
        with zipfile.ZipFile(BytesIO(wheel_raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                _fail()
            total = 0
            for name in sorted(names):
                if name.endswith("/") or not _code_file(name):
                    continue
                row = archive.getinfo(name)
                total += row.file_size
                if row.file_size > 64 * 1024 * 1024 or total > 512 * 1024 * 1024:
                    _fail()
                data = archive.read(name)
                files.append({"path": name, "byte_sha256": _sha(data), "size": len(data)})
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise SystemSecurityError("NEXT_SESSION_TRANSPORT_BINDING_INVALID") from exc
    if (
        _manifest_sha(files) != payload["installed_code_manifest_sha256"]
        or _manifest_sha(files) != release["payload"]["code_manifest_sha256"]
        or wheel["byte_sha256"] != release["payload"]["wheel_sha256"]
        or payload["release_ref"] != execution["deployed_release_ref"]
    ):
        _fail()
    modules = [
        {"path": row["path"], "sha256": row["byte_sha256"]}
        for row in files
        if row["path"] in _MODULE_PATHS
    ]
    if {row["path"] for row in modules} != _MODULE_PATHS:
        _fail()
    return {"release_ref": execution["deployed_release_ref"], "module_file_refs": modules}


def _validate_events(journal, info, events):
    from . import tushare_calendar_authority as native

    if type(events) is not list or len(events) != 4:
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    execution = info["execution"]["payload"]
    raw_refs = {Path(row["relative_path"]).name: row for row in info["raw_refs"]}
    start = (native.RUNTIME_START_DATE - timedelta(days=native.CAPTURE_PREHISTORY_DAYS)).strftime(
        "%Y%m%d"
    )
    end = (
        datetime.strptime(journal.trade_date, "%Y%m%d") + timedelta(days=HORIZON_NATURAL_DAYS)
    ).strftime("%Y%m%d")
    previous = utc_stamp(execution["observed_started_at"])
    for ordinal, event in enumerate(events):
        if type(event) is not dict or set(event) != _EVENT_FIELDS:
            _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
        if ordinal == 0:
            binding = {
                "kind": "DOCUMENTATION",
                "method": "GET",
                "host": "tushare.pro",
                "path": "/document/2?doc_id=26",
                "api_name": None,
                "parameters": {},
                "expected_fields": [],
            }
            ref = execution["documentation_raw_file_ref"]
        else:
            exchange = ("SSE", "SZSE", "BSE")[ordinal - 1]
            binding = {
                "kind": "PROVIDER",
                "method": "POST",
                "host": "api.tushare.pro",
                "path": "/",
                "api_name": native.API_NAME,
                "parameters": {"start_date": start, "end_date": end, "exchange": exchange},
                "expected_fields": list(native.EXPECTED_FIELDS),
            }
            ref = raw_refs[f"response-{exchange.lower()}.raw"]
        raw = _native_raw(journal, info, ref)
        binding.update(
            ordinal=ordinal,
            scheme="https",
            port=443,
            http_status=200,
            tls_verified=True,
            redirect_count=0,
            response_bytes=len(raw),
            response_sha256=_sha(raw),
        )
        if (
            any(
                type(event[k]) is not int
                for k in ("ordinal", "port", "http_status", "redirect_count", "response_bytes")
            )
            or event["tls_verified"] is not True
            or any(event[k] != value for k, value in binding.items())
        ):
            _fail()
        began, ended = utc_stamp(event["request_started_at"]), utc_stamp(
            event["response_completed_at"]
        )
        if not previous <= began <= ended <= utc_stamp(execution["observed_completed_at"]):
            _fail("NEXT_SESSION_TRANSPORT_CHRONOLOGY_INVALID")
        previous = ended


def validate_transport_evidence(journal, info, value):
    """Validate only against an already native-inspected capture; no admission flags."""
    if (
        type(value) is not dict
        or set(value) != _FIELDS
        or value["schema_version"] != SCHEMA
        or not _false_authority(value["authority"])
    ):
        _fail("NEXT_SESSION_PRODUCTION_SCHEMA_INVALID")
    validate_ref(value["release_install_input_ref"])
    execution = info["execution"]["payload"]
    install_sha = execution["release_install_input_file_ref"]["byte_sha256"]
    if (
        value["eod_trade_date"] != journal.trade_date
        or value["release_install_input_ref"]["sha256"] != install_sha
        or value["release_ref"] != execution["deployed_release_ref"]
        or value["execution_ref"] != info["execution_ref"]
        or value["success_ref"] != info["success_ref"]
        or value["capture_root_name"] != execution["capture_root_name"]
        or value["capture_root_name"] != "future-" + journal.trade_date + "-" + install_sha[:16]
        or value["recorder_identity"] != _recorder_identity(journal, info)
        or value["event_set_sha256"] != _sha(canonical_json_bytes(value["events"]))
    ):
        _fail()
    _validate_events(journal, info, value["events"])
    if (
        not utc_stamp(info["success"]["payload"]["observed_completed_at"])
        <= utc_stamp(value["recorded_at"])
        <= utc_stamp(_now())
    ):
        _fail("NEXT_SESSION_TRANSPORT_CHRONOLOGY_INVALID")
    return value


def read_transport_evidence(journal, info, ref=None):
    """Deterministic retained-evidence reader; missing receipt never triggers capture."""
    path = transport_evidence_path(journal, info["execution_ref"])
    if ref is not None:
        validate_ref(ref)
        if ref["path"] != path:
            _fail()
    stored = journal.storage.read(path)
    if stored is None:
        raise ContractError("PROVENANCE_UNAVAILABLE")
    if ref is not None and stored.byte_sha256 != ref["sha256"]:
        _fail()
    value = validate_transport_evidence(journal, info, parse_canonical_json_bytes(stored.data))
    again = journal.storage.read(path)
    if again is None or again.data != stored.data:
        _fail()
    return {"path": path, "sha256": stored.byte_sha256}, value


def publish_transport_evidence(*, workspace, trade_date, captured, session):
    """Fresh invocation only; events cannot be supplied by a public caller."""
    from ._calendar_production_transport import _retained_session_events

    events = _retained_session_events(session)
    if session.workspace != Path(workspace).resolve(strict=True) or session.day != trade_date:
        _fail()
    journal = DailyJournal(str(workspace), trade_date)
    with journal.locked():
        if (
            journal.storage.read(str(journal.root / "completion.v1.json")) is not None
            or selected_binding(journal.storage, str(journal.root / "nodes/calendar"))[0]
            is not None
        ):
            raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")
        info = inspect_next_session_capture(
            workspace=str(workspace),
            eod_trade_date=trade_date,
            execution=captured["capture_execution"],
            execution_ref=captured["capture_execution_file_ref"],
            success=captured["capture_success"],
            success_ref=captured["capture_success_file_ref"],
        )
        value = {
            "schema_version": SCHEMA,
            "eod_trade_date": trade_date,
            "release_install_input_ref": session.install_ref,
            "release_ref": session.release_ref,
            "execution_ref": info["execution_ref"],
            "success_ref": info["success_ref"],
            "capture_root_name": info["execution"]["payload"]["capture_root_name"],
            "recorder_identity": session.recorder_identity,
            "events": events,
            "event_set_sha256": _sha(canonical_json_bytes(events)),
            "recorded_at": _now(),
            "authority": FALSE_AUTHORITY,
        }
        path = transport_evidence_path(journal, info["execution_ref"])
        existing = journal.storage.read(path)
        if existing is not None:
            ref, retained = read_transport_evidence(journal, info)
            if any(retained[key] != item for key, item in value.items() if key != "recorded_at"):
                _fail()
            return ref
        validate_transport_evidence(journal, info, value)
        stored = journal.storage.write(path, canonical_json_bytes(value))
        ref = {"path": path, "sha256": stored.byte_sha256}
        read_transport_evidence(journal, info, ref)
        return ref
