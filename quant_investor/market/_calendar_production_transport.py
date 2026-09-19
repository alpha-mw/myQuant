"""Private observation of fixed successful HTTPS boundaries in an installed run.

This is same-process provenance, not an attestation against a malicious owner.
Only a verified production scope records; ordinary transport behavior is unchanged.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
from pathlib import Path
import ssl

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import utc_stamp, validate_ref
from quant_investor.operations.daily_journal import _validate_day
from quant_investor.system.errors import SystemSecurityError

_REAL_DATETIME = datetime
_ACTIVE = ContextVar("production_calendar_transport", default=None)
_PENDING = ContextVar("production_calendar_provider_request", default=None)
_MODULE_PATHS = frozenset(
    {
        "quant_investor/market/_calendar_production_transport.py",
        "quant_investor/market/tushare_transport.py",
        "quant_investor/market/tushare_calendar_authority.py",
    }
)


def _fail(code):
    session = _ACTIVE.get()
    if session is not None:
        session.failed = True
        session.integrity_error = code
    raise SystemSecurityError(code)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _guard_route():
    from . import tushare_calendar_authority as native
    from . import tushare_transport as transport
    from . import next_session_acquisition as acquisition
    from ._calendar_fixture_capability import _ACTIVE as fixture_active

    if (
        fixture_active.get() is not None
        or _now is not _ORIGINAL_NOW
        or _begin_response is not _ORIGINAL_BEGIN
        or _complete_response is not _ORIGINAL_COMPLETE
        or _provider_request_scope is not _ORIGINAL_PROVIDER_SCOPE
        or datetime is not _REAL_DATETIME
        or native.datetime is not _REAL_DATETIME
        or acquisition.datetime is not _REAL_DATETIME
        or http.client.HTTPSConnection is not transport._ORIGINAL_HTTPS_CONNECTION
        or ssl.create_default_context is not transport._ORIGINAL_CREATE_CONTEXT
        or transport._HTTPS_CONNECTION is not transport._ORIGINAL_HTTPS_CONNECTION
        or transport._CREATE_DEFAULT_CONTEXT is not transport._ORIGINAL_CREATE_CONTEXT
        or transport.OfficialTushareHttpsClient is not transport._ORIGINAL_CALENDAR_CLIENT
        or native.OfficialTushareHttpsClient is not transport._ORIGINAL_CALENDAR_CLIENT
        or native._official_documentation_fetch is not native._ORIGINAL_DOCUMENTATION_FETCH
        or native.capture_trusted_provider_calendar_evidence is not native._ORIGINAL_CAPTURE
        or acquisition.capture_trusted_provider_calendar_evidence is not native._ORIGINAL_CAPTURE
        or transport.OfficialTushareHttpsClient.request is not transport._ORIGINAL_REQUEST
        or transport.OfficialTushareHttpsClient._fetch_raw is not transport._ORIGINAL_FETCH_RAW
        or transport.OfficialTushareHttpsClient._prepare_request is not transport._ORIGINAL_PREPARE
        or native._utc_now is not native._ORIGINAL_UTC_NOW
    ):
        _fail("NEXT_SESSION_TRANSPORT_ROUTE_UNAVAILABLE")


@dataclass
class _Session:
    workspace: Path
    day: str
    install_ref: dict
    release_ref: dict
    recorder_identity: dict
    events: list = field(default_factory=list)
    failed: bool = False
    integrity_error: str | None = None
    ticket: object = None


@contextmanager
def _production_transport_scope(*, workspace, day, install_raw, install_ref, repository):
    """Private fresh-only entry; recovery must never enter this context."""
    if _ACTIVE.get() is not None or _PENDING.get() is not None:
        _fail("NEXT_SESSION_TRANSPORT_ROUTE_UNAVAILABLE")
    if _guard_route is not _ORIGINAL_GUARD:
        _fail("NEXT_SESSION_TRANSPORT_ROUTE_UNAVAILABLE")
    _guard_route()
    _validate_day(day)
    validate_ref(install_ref)
    if (
        type(install_raw) is not bytes
        or hashlib.sha256(install_raw).hexdigest() != install_ref["sha256"]
    ):
        _fail("NEXT_SESSION_TRANSPORT_BINDING_INVALID")
    from quant_investor.system.release_install import verify_running_release_install_input
    from quant_investor.system.release import installed_code_manifest

    verified = verify_running_release_install_input(install_raw, repository_root=repository)
    manifest = installed_code_manifest()
    if (
        verified.get("state") != "PASS"
        or hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        != verified["installed_code_manifest_sha256"]
    ):
        _fail("NEXT_SESSION_TRANSPORT_BINDING_INVALID")
    modules = [
        {"path": row["path"], "sha256": row["byte_sha256"]}
        for row in manifest["files"]
        if row["path"] in _MODULE_PATHS
    ]
    if {row["path"] for row in modules} != _MODULE_PATHS or len(modules) != len(_MODULE_PATHS):
        _fail("NEXT_SESSION_TRANSPORT_BINDING_INVALID")
    _guard_route()
    session = _Session(
        Path(workspace).resolve(strict=True),
        day,
        dict(install_ref),
        dict(verified["release_ref"]),
        {"release_ref": dict(verified["release_ref"]), "module_file_refs": modules},
    )
    token = _ACTIVE.set(session)
    try:
        yield session
    except BaseException:
        session.failed = True
        # Native acquisition retains failures but wraps their exception class.
        # Never let that wrapper turn a provenance-integrity failure into an
        # ordinary optional acquisition failure.
        if session.integrity_error is not None:
            raise SystemSecurityError(session.integrity_error) from None
        raise
    finally:
        _ACTIVE.reset(token)


@contextmanager
def _provider_request_scope(*, api_name, params, expected_fields):
    """Stage sanitized identity only; successful evidence comes from _fetch_raw."""
    session = _ACTIVE.get()
    if session is None:
        yield
        return
    _guard_route()
    from . import tushare_calendar_authority as native

    ordinal = len(session.events)
    end = (datetime.strptime(session.day, "%Y%m%d") + timedelta(days=21)).strftime("%Y%m%d")
    start = (native.RUNTIME_START_DATE - timedelta(days=native.CAPTURE_PREHISTORY_DAYS)).strftime(
        "%Y%m%d"
    )
    if (
        session.failed
        or session.ticket is not None
        or _PENDING.get() is not None
        or ordinal not in (1, 2, 3)
        or api_name != native.API_NAME
        or dict(params)
        != {"start_date": start, "end_date": end, "exchange": ("SSE", "SZSE", "BSE")[ordinal - 1]}
        or tuple(expected_fields) != native.EXPECTED_FIELDS
    ):
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    pending = {
        "api_name": api_name,
        "parameters": dict(params),
        "expected_fields": list(expected_fields),
    }
    token = _PENDING.set(pending)
    try:
        yield
        if len(session.events) != ordinal + 1 or session.ticket is not None:
            _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    except BaseException:
        session.failed = True
        raise
    finally:
        _PENDING.reset(token)


def _begin_response(kind):
    session = _ACTIVE.get()
    if session is None:
        return None
    _guard_route()
    ordinal = len(session.events)
    pending = _PENDING.get()
    if session.failed or session.ticket is not None:
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    if kind == "DOCUMENTATION":
        if ordinal != 0 or pending is not None:
            _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
        event = {
            "method": "GET",
            "host": "tushare.pro",
            "path": "/document/2?doc_id=26",
            "api_name": None,
            "parameters": {},
            "expected_fields": [],
        }
    elif kind == "PROVIDER" and ordinal in (1, 2, 3) and pending is not None:
        event = {"method": "POST", "host": "api.tushare.pro", "path": "/", **pending}
    else:
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    event.update(ordinal=ordinal, kind=kind, scheme="https", port=443, request_started_at=_now())
    ticket = (session, event)
    session.ticket = ticket
    return ticket


def _complete_response(ticket, *, raw, tls_context):
    if ticket is None:
        return
    session = _ACTIVE.get()
    if session is None or ticket is not session.ticket or ticket[0] is not session:
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    _guard_route()
    if (
        session.failed
        or type(raw) is not bytes
        or not raw
        or tls_context.verify_mode != ssl.CERT_REQUIRED
        or tls_context.check_hostname is not True
    ):
        _fail("NEXT_SESSION_TRANSPORT_BINDING_INVALID")
    event = dict(ticket[1])
    stamp = _now()
    if utc_stamp(stamp) < utc_stamp(event["request_started_at"]):
        _fail("NEXT_SESSION_TRANSPORT_CHRONOLOGY_INVALID")
    if session.events and utc_stamp(event["request_started_at"]) < utc_stamp(
        session.events[-1]["response_completed_at"]
    ):
        _fail("NEXT_SESSION_TRANSPORT_CHRONOLOGY_INVALID")
    event.update(
        response_completed_at=stamp,
        response_bytes=len(raw),
        response_sha256=hashlib.sha256(raw).hexdigest(),
        http_status=200,
        tls_verified=True,
        redirect_count=0,
    )
    session.events.append(event)
    session.ticket = None


def _retained_session_events(session):
    """Consume only within the same private invocation; never caller-supplied events."""
    if (
        _ACTIVE.get() is not session
        or session.failed
        or session.ticket is not None
        or len(session.events) != 4
    ):
        _fail("NEXT_SESSION_TRANSPORT_EVENT_SET_INVALID")
    _guard_route()
    # A detached canonical copy prevents mutation through the return value.
    from quant_investor.contracts import parse_canonical_json_bytes

    return parse_canonical_json_bytes(canonical_json_bytes(session.events))


_ORIGINAL_NOW = _now
_ORIGINAL_BEGIN = _begin_response
_ORIGINAL_COMPLETE = _complete_response
_ORIGINAL_PROVIDER_SCOPE = _provider_request_scope
_ORIGINAL_GUARD = _guard_route
