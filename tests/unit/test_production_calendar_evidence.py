"""Receipt contract tests with disclosed native-capture/release admission seams."""

from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import production_calendar_evidence as evidence
from quant_investor.market import tushare_calendar_authority as native
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.daily_contract import ContractError
from quant_investor.system.errors import SystemSecurityError
from quant_investor.system.release_install import _manifest_sha


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bound(tmp_path, monkeypatch):
    # Tests exercise exact wheel/raw/hash/receipt bindings. Native capture and
    # installed-release admission are separate explicit seams, not live evidence.
    journal = DailyJournal(str(tmp_path), "20260902")
    install = b"unit-only-install"
    root = "future-20260902-" + digest(install)[:16]
    release_ref = {"kind": "system.release", "byte_sha256": "b" * 64}
    files = {"documentation.raw": b"docs", "release-install-input.json": install}
    files.update({f"response-{ex.lower()}.raw": ex.encode() for ex in ("SSE", "SZSE", "BSE")})
    refs = {
        name: {"relative_path": root + "/" + name, "byte_sha256": digest(raw)}
        for name, raw in files.items()
    }
    with journal.locked():
        for name, raw in files.items():
            journal.storage.write(str(journal.root / "calendar-future/captures" / root / name), raw)
    module_rows = []
    wheel_path = tmp_path / "release.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for name in sorted(evidence._MODULE_PATHS):
            raw = ("# unit wheel module " + name).encode()
            wheel.writestr(name, raw)
            module_rows.append({"path": name, "byte_sha256": digest(raw), "size": len(raw)})
    wheel_path.chmod(0o600)
    wheel_raw = wheel_path.read_bytes()
    manifest_sha = _manifest_sha(module_rows)
    wheel_row = {"path": str(wheel_path), "size": len(wheel_raw), "byte_sha256": digest(wheel_raw)}
    monkeypatch.setattr(
        native,
        "_release_install_components",
        lambda *a, **k: (
            {
                "payload": {
                    "wheel": wheel_row,
                    "installed_code_manifest_sha256": manifest_sha,
                    "release_ref": release_ref,
                }
            },
            {"payload": {"code_manifest_sha256": manifest_sha, "wheel_sha256": digest(wheel_raw)}},
            {},
        ),
    )
    execution_ref = {"relative_path": root + "/capture-execution.json", "byte_sha256": "c" * 64}
    success_ref = {"relative_path": root + "/capture-success.json", "byte_sha256": "d" * 64}
    info = {
        "execution_ref": execution_ref,
        "success_ref": success_ref,
        "execution": {
            "payload": {
                "capture_root_name": root,
                "release_install_input_file_ref": refs["release-install-input.json"],
                "documentation_raw_file_ref": refs["documentation.raw"],
                "deployed_release_ref": release_ref,
                "observed_started_at": "2026-09-02T11:00:00Z",
                "observed_completed_at": "2026-09-02T11:00:10Z",
            }
        },
        "success": {"payload": {"observed_completed_at": "2026-09-02T11:00:11Z"}},
        "raw_refs": [refs[f"response-{ex.lower()}.raw"] for ex in ("SSE", "SZSE", "BSE")],
    }
    events = []
    for ordinal, ex in enumerate((None, "SSE", "SZSE", "BSE")):
        name = "documentation.raw" if ex is None else f"response-{ex.lower()}.raw"
        events.append(
            {
                "ordinal": ordinal,
                "kind": "DOCUMENTATION" if ex is None else "PROVIDER",
                "method": "GET" if ex is None else "POST",
                "scheme": "https",
                "host": "tushare.pro" if ex is None else "api.tushare.pro",
                "port": 443,
                "path": "/document/2?doc_id=26" if ex is None else "/",
                "api_name": None if ex is None else "trade_cal",
                "parameters": (
                    {}
                    if ex is None
                    else {"start_date": "20231201", "end_date": "20260923", "exchange": ex}
                ),
                "expected_fields": [] if ex is None else list(native.EXPECTED_FIELDS),
                "request_started_at": f"2026-09-02T11:00:0{ordinal * 2}Z",
                "response_completed_at": f"2026-09-02T11:00:0{ordinal * 2 + 1}Z",
                "response_bytes": len(files[name]),
                "response_sha256": digest(files[name]),
                "http_status": 200,
                "tls_verified": True,
                "redirect_count": 0,
            }
        )
    value = {
        "schema_version": evidence.SCHEMA,
        "eod_trade_date": journal.trade_date,
        "release_install_input_ref": {"path": "release.json", "sha256": digest(install)},
        "release_ref": release_ref,
        "execution_ref": execution_ref,
        "success_ref": success_ref,
        "capture_root_name": root,
        "recorder_identity": evidence._recorder_identity(journal, info),
        "events": events,
        "event_set_sha256": digest(canonical_json_bytes(events)),
        "recorded_at": "2026-09-02T11:00:12Z",
        "authority": FALSE_AUTHORITY,
    }
    return journal, info, value


def test_valid_retained_receipt_is_read_only_and_exact(bound):
    journal, info, value = bound
    path = evidence.transport_evidence_path(journal, info["execution_ref"])
    with journal.locked():
        stored = journal.storage.write(path, canonical_json_bytes(value))
    ref = {"path": path, "sha256": stored.byte_sha256}
    assert evidence.read_transport_evidence(journal, info, ref) == (ref, value)
    assert journal.storage.read(path).data == canonical_json_bytes(value)
    with pytest.raises(SystemSecurityError):
        evidence.read_transport_evidence(journal, info, {**ref, "sha256": "e" * 64})


def test_absent_evidence_is_not_fabricated(bound):
    journal, info, _ = bound
    with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
        evidence.read_transport_evidence(journal, info)
    assert (
        journal.storage.read(evidence.transport_evidence_path(journal, info["execution_ref"]))
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "length",
        "bool",
        "host",
        "fields",
        "extra",
        "missing",
        "order",
        "parameter",
        "chronology",
        "future",
        "module",
        "release",
        "schema",
        "root",
    ],
)
def test_resealed_wrong_receipts_fail_native_bindings(bound, mutation):
    journal, info, original = bound
    value = deepcopy(original)
    events = value["events"]
    if mutation == "hash":
        events[1]["response_sha256"] = "e" * 64
    elif mutation == "length":
        events[1]["response_bytes"] += 1
    elif mutation == "bool":
        events[0]["ordinal"] = False
    elif mutation == "host":
        events[1]["host"] = "other.invalid"
    elif mutation == "fields":
        events[1]["expected_fields"] = []
    elif mutation == "extra":
        events.append(deepcopy(events[-1]))
    elif mutation == "missing":
        events.pop()
    elif mutation == "order":
        events[1], events[2] = events[2], events[1]
    elif mutation == "parameter":
        events[1]["parameters"]["end_date"] = "20260924"
    elif mutation == "chronology":
        events[1]["request_started_at"] = "2026-09-02T10:00:00Z"
    elif mutation == "future":
        value["recorded_at"] = "2099-01-01T00:00:00Z"
    elif mutation == "module":
        value["recorder_identity"]["module_file_refs"][0]["sha256"] = "e" * 64
    elif mutation == "release":
        value["release_ref"]["byte_sha256"] = "e" * 64
    elif mutation == "schema":
        value["extra"] = True
    elif mutation == "root":
        value["capture_root_name"] = "other"
    value["event_set_sha256"] = digest(canonical_json_bytes(events))
    with pytest.raises(SystemSecurityError):
        evidence.validate_transport_evidence(journal, info, value)


def test_changed_wheel_rejected(bound):
    journal, info, value = bound
    wheel = Path(journal.storage._io.workspace_root) / "release.whl"
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(SystemSecurityError):
        evidence.validate_transport_evidence(journal, info, value)
