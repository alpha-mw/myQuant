"""Private offline-runner capability; never enabled by serialized configuration.

The scope itself installs byte-fixed adapters. Public inputs cannot set its
ContextVar. This is a test/research boundary, not a Python sandbox.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
from pathlib import Path
from types import MappingProxyType

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, utc_stamp
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _validate_day
from quant_investor.operations.journal_revisions import selected_binding
from . import tushare_calendar_authority as native
from .tushare_transport import replay_tushare_response_bytes

_ACTIVE = ContextVar("offline_calendar_fixture", default=None)
EXCHANGES = frozenset({"SSE", "SZSE", "BSE"})


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unavailable():
    raise ContractError("PROVENANCE_UNAVAILABLE")


@dataclass(frozen=True)
class _Capability:
    workspace: Path
    day: str
    install_sha: str
    manifest_raw: bytes
    transport: type
    documentation_fetch: object


def require_fixture_capability(*, workspace, trade_date=None, install_sha=None):
    cap = _ACTIVE.get()
    if (
        cap is None
        or cap.workspace != Path(workspace).resolve(strict=True)
        or (trade_date is not None and cap.day != trade_date)
        or (install_sha is not None and cap.install_sha != install_sha)
        or native.OfficialTushareHttpsClient is not cap.transport
        or native._official_documentation_fetch is not cap.documentation_fetch
    ):
        _unavailable()
    return cap


@contextmanager
def _offline_calendar_fixture(
    *,
    workspace,
    trade_date,
    install_sha,
    fixture_source_sha,
    documentation_raw,
    provider_raw,
    current_provider_raw=None,
):
    """Only the non-public offline runner calls this, with pinned fixture bytes."""
    _validate_day(trade_date)
    if _ACTIVE.get() is not None or set(provider_raw) != EXCHANGES:
        _unavailable()
    if type(documentation_raw) is not bytes or any(
        type(v) is not bytes for v in provider_raw.values()
    ):
        _unavailable()
    for digest in (install_sha, fixture_source_sha):
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            _unavailable()
    if current_provider_raw is not None and (
        set(current_provider_raw) != EXCHANGES
        or any(type(v) is not bytes for v in current_provider_raw.values())
    ):
        _unavailable()
    responses = MappingProxyType(dict(provider_raw))
    current_responses = (
        None if current_provider_raw is None else MappingProxyType(dict(current_provider_raw))
    )
    future_end = (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=21)).strftime("%Y%m%d")
    manifest = canonical_json_bytes(
        {
            "schema_version": "cn-calendar-offline-fixture-manifest.v1",
            "eod_trade_date": trade_date,
            "release_install_input_sha256": install_sha,
            "fixture_source_sha256": fixture_source_sha,
            "documentation_sha256": _sha(documentation_raw),
            "provider_response_sha256": {k: _sha(v) for k, v in responses.items()},
            "authority": FALSE_AUTHORITY,
        }
    )

    class OfflineTransport:
        def __init__(self, **kwargs):
            pass

        def request(self, *, api_name, params, **kwargs):
            if api_name != "trade_cal" or params.get("exchange") not in EXCHANGES:
                _unavailable()
            end = params.get("end_date")
            selected = (
                responses
                if end == future_end
                else (current_responses if end == trade_date else None)
            )
            if selected is None:
                _unavailable()
            return replay_tushare_response_bytes(
                selected[params["exchange"]],
                api_name=api_name,
                expected_fields=native.EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    def docs():
        return documentation_raw, 200, {"content-type": "text/html; charset=utf-8"}, True, []

    cap = _Capability(
        Path(workspace).resolve(strict=True),
        trade_date,
        install_sha,
        manifest,
        OfflineTransport,
        docs,
    )
    original = native.OfficialTushareHttpsClient, native._official_documentation_fetch
    native.OfficialTushareHttpsClient, native._official_documentation_fetch = OfflineTransport, docs
    token = _ACTIVE.set(cap)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        native.OfficialTushareHttpsClient, native._official_documentation_fetch = original


def _inspect(cap, captured):
    from .next_session_calendar import inspect_next_session_capture

    info = inspect_next_session_capture(
        workspace=str(cap.workspace),
        eod_trade_date=cap.day,
        execution=captured["capture_execution"],
        execution_ref=captured["capture_execution_file_ref"],
        success=captured["capture_success"],
        success_ref=captured["capture_success_file_ref"],
    )
    manifest = parse_canonical_json_bytes(cap.manifest_raw)
    payload = info["execution"]["payload"]
    if payload["release_install_input_file_ref"]["byte_sha256"] != cap.install_sha:
        _unavailable()
    if payload["documentation_raw_file_ref"]["byte_sha256"] != manifest["documentation_sha256"]:
        _unavailable()
    actual = {Path(v["relative_path"]).name: v["byte_sha256"] for v in info["raw_refs"]}
    expected = {
        "response-" + k.lower() + ".raw": v for k, v in manifest["provider_response_sha256"].items()
    }
    if actual != expected:
        _unavailable()
    return info


def _identity(cap, info, manifest_ref):
    manifest = parse_canonical_json_bytes(cap.manifest_raw)
    return {
        "schema_version": "cn-calendar-fixture-transport-evidence.v1",
        "eod_trade_date": cap.day,
        "release_install_input_sha256": cap.install_sha,
        "fixture_manifest_ref": manifest_ref,
        "documentation_sha256": manifest["documentation_sha256"],
        "provider_response_sha256": manifest["provider_response_sha256"],
        "execution_ref": info["execution_ref"],
        "success_ref": info["success_ref"],
        "authority": FALSE_AUTHORITY,
    }


def _validate_evidence(value, identity, info):
    if set(value) != {*identity, "recorded_at"} or any(value[k] != v for k, v in identity.items()):
        _unavailable()
    try:
        captured_at = utc_stamp(info["success"]["payload"]["observed_completed_at"])
        recorded_at = utc_stamp(value["recorded_at"])
        if not captured_at <= recorded_at <= datetime.now(timezone.utc):
            _unavailable()
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("PROVENANCE_UNAVAILABLE") from exc


def publish_fixture_evidence(*, workspace, trade_date, install_sha, captured):
    cap = require_fixture_capability(
        workspace=workspace, trade_date=trade_date, install_sha=install_sha
    )
    info = _inspect(cap, captured)
    journal = DailyJournal(str(cap.workspace), cap.day)
    with journal.locked():
        if (
            journal.storage.read(str(journal.root / "completion.v1.json")) is not None
            or selected_binding(journal.storage, str(journal.root / "nodes/calendar"))[0]
            is not None
        ):
            raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")
        base = journal.root / "calendar-future"
        path = str(base / "fixture-manifests" / (_sha(cap.manifest_raw) + ".json"))
        stored = journal.storage.write(path, cap.manifest_raw)
        ref = {"path": path, "sha256": stored.byte_sha256}
        identity = _identity(cap, info, ref)
        path = str(base / "fixture-evidence" / (_sha(canonical_json_bytes(identity)) + ".json"))
        old = journal.storage.read(path)
        if old is None:
            raw = canonical_json_bytes(
                {
                    **identity,
                    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            old = journal.storage.write(path, raw)
        value = parse_canonical_json_bytes(old.data)
        _validate_evidence(value, identity, info)
        return {"path": path, "sha256": old.byte_sha256}


def publish_bound_fixture_proof(*, workspace, trade_date, install_sha, captured, evidence_ref):
    cap = require_fixture_capability(
        workspace=workspace, trade_date=trade_date, install_sha=install_sha
    )
    info = _inspect(cap, captured)
    journal = DailyJournal(str(cap.workspace), cap.day)
    manifest_path = str(
        journal.root / "calendar-future/fixture-manifests" / (_sha(cap.manifest_raw) + ".json")
    )
    manifest_ref = {"path": manifest_path, "sha256": _sha(cap.manifest_raw)}
    identity = _identity(cap, info, manifest_ref)
    expected_path = str(
        journal.root
        / "calendar-future/fixture-evidence"
        / (_sha(canonical_json_bytes(identity)) + ".json")
    )
    if type(evidence_ref) is not dict or evidence_ref.get("path") != expected_path:
        _unavailable()
    stored = journal.storage.read(expected_path)
    manifest = journal.storage.read(manifest_path)
    if (
        stored is None
        or stored.byte_sha256 != evidence_ref.get("sha256")
        or manifest is None
        or manifest.data != cap.manifest_raw
    ):
        _unavailable()
    value = parse_canonical_json_bytes(stored.data)
    _validate_evidence(value, identity, info)
    from .next_session_proof import publish_synthetic_next_session_proof

    return publish_synthetic_next_session_proof(
        workspace=workspace,
        eod_trade_date=trade_date,
        execution=captured["capture_execution"],
        execution_ref=captured["capture_execution_file_ref"],
        success=captured["capture_success"],
        success_ref=captured["capture_success_file_ref"],
    )


def retained_fixture_evidence(*, workspace, trade_date, install_sha, captured):
    """Resolve one deterministic evidence identity without publishing or scanning."""
    cap = require_fixture_capability(
        workspace=workspace, trade_date=trade_date, install_sha=install_sha
    )
    info = _inspect(cap, captured)
    journal = DailyJournal(str(cap.workspace), cap.day)
    manifest_path = str(
        journal.root / "calendar-future/fixture-manifests" / (_sha(cap.manifest_raw) + ".json")
    )
    manifest_ref = {"path": manifest_path, "sha256": _sha(cap.manifest_raw)}
    identity = _identity(cap, info, manifest_ref)
    path = str(
        journal.root
        / "calendar-future/fixture-evidence"
        / (_sha(canonical_json_bytes(identity)) + ".json")
    )
    stored = journal.storage.read(path)
    manifest = journal.storage.read(manifest_path)
    if stored is None or manifest is None or manifest.data != cap.manifest_raw:
        _unavailable()
    value = parse_canonical_json_bytes(stored.data)
    _validate_evidence(value, identity, info)
    return {"path": path, "sha256": stored.byte_sha256}
