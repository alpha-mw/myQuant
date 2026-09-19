"""Production v2 Calendar proof derived from retained installed HTTPS evidence."""

import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, utc_stamp
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from quant_investor.operations.journal_revisions import selected_binding
from .next_session_calendar import inspect_next_session_capture
from .next_session_proof import _read, _write, _core as _synthetic_core
from .production_calendar_evidence import read_transport_evidence

PROVENANCE_SCHEMA = "cn-calendar-acquisition-provenance.v2"
PROOF_SCHEMA = "cn-next-session-calendar-proof.v2"
PUBLICATION_SCHEMA = "cn-next-session-calendar-proof-publication.v2"
_CORE_FIELDS = frozenset(
    {
        "schema_version",
        "eod_trade_date",
        "observed_through_date",
        "next_open_session",
        "capture_root_ref",
        "transaction_ref",
        "execution_ref",
        "success_ref",
        "provider_capture_refs",
        "raw_refs",
        "projection_sha256",
        "policy_ref",
        "capability_ref",
        "source_limitations",
        "release_ref",
        "capture_started_at",
        "capture_completed_at",
        "acquisition_provenance_ref",
        "synthetic",
        "authority",
        "transport_evidence_ref",
        "source_classification",
    }
)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _identity(info, transport_ref, transport):
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "execution_ref": info["execution_ref"],
        "success_ref": info["success_ref"],
        "release_ref": info["execution"]["payload"]["deployed_release_ref"],
        "release_install_input_ref": transport["release_install_input_ref"],
        "transport_evidence_ref": transport_ref,
        "issuer_route": "INSTALLED_OFFICIAL_CALENDAR",
        "transport_mode": "OFFICIAL_HTTPS",
        "observed_network_call_count": len(transport["events"]),
        "authority": FALSE_AUTHORITY,
    }


def _validate_provenance(info, ref, transport, value):
    identity = _identity(info, ref, transport)
    if (
        type(value) is not dict
        or set(value) != {*identity, "recorded_at"}
        or type(value["observed_network_call_count"]) is not int
        or any(value[k] != expected for k, expected in identity.items())
        or not _false_authority(value["authority"])
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_INVALID_OR_ROUTE_UNAVAILABLE")
    if (
        not utc_stamp(transport["recorded_at"])
        <= utc_stamp(value["recorded_at"])
        <= utc_stamp(_now())
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_CHRONOLOGY_INVALID")


def _core(info, provenance_ref, transport_ref):
    return {
        **_synthetic_core(info, provenance_ref),
        "schema_version": PROOF_SCHEMA,
        "synthetic": False,
        "transport_evidence_ref": transport_ref,
        "source_classification": "INSTALLED_NATIVE_HTTPS_OBSERVED",
    }


def publish_production_next_session_proof(
    *,
    workspace,
    eod_trade_date,
    execution,
    execution_ref,
    success,
    success_ref,
    transport_evidence_ref,
):
    """Fresh or recovery publication; no transport or provenance fabrication."""
    journal = DailyJournal(workspace, eod_trade_date)
    with journal.locked():
        if (
            journal.storage.read(str(journal.root / "completion.v1.json")) is not None
            or selected_binding(journal.storage, str(journal.root / "nodes/calendar"))[0]
            is not None
        ):
            raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")
        info = inspect_next_session_capture(
            workspace=workspace,
            eod_trade_date=eod_trade_date,
            execution=execution,
            execution_ref=execution_ref,
            success=success,
            success_ref=success_ref,
        )
        transport_ref, transport = read_transport_evidence(journal, info, transport_evidence_ref)
        identity = _identity(info, transport_ref, transport)
        base = journal.root / "calendar-future"
        path = base / "production-provenance" / (execution_ref["byte_sha256"] + ".json")
        existing = journal.storage.read(str(path))
        if existing is None:
            provenance_ref = _write(journal, path, {**identity, "recorded_at": _now()})
        else:
            provenance_ref = {"path": str(path), "sha256": existing.byte_sha256}
        _validate_provenance(info, transport_ref, transport, _read(journal, provenance_ref))
        core = _core(info, provenance_ref, transport_ref)
        digest = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
        core_ref = _write(journal, base / "proofs" / (digest + ".json"), core)
        publication_path = base / "publications" / (digest + ".json")
        existing = journal.storage.read(str(publication_path))
        if existing is None:
            publication_ref = _write(
                journal,
                publication_path,
                {
                    "schema_version": PUBLICATION_SCHEMA,
                    "proof_ref": core_ref,
                    "proof_sealed_at": _now(),
                    "authority": FALSE_AUTHORITY,
                },
            )
        else:
            publication_ref = {"path": str(publication_path), "sha256": existing.byte_sha256}
        read_production_next_session_proof(
            workspace=workspace, eod_trade_date=eod_trade_date, publication_ref=publication_ref
        )
        return publication_ref


def read_production_next_session_proof(*, workspace, eod_trade_date, publication_ref):
    """Exact versioned native replay; Calendar provenance only, not EOD admission."""
    journal = DailyJournal(workspace, eod_trade_date)
    publication = _read(journal, publication_ref)
    if (
        type(publication) is not dict
        or set(publication) != {"schema_version", "proof_ref", "proof_sealed_at", "authority"}
        or publication["schema_version"] != PUBLICATION_SCHEMA
        or not _false_authority(publication["authority"])
    ):
        raise ContractError("NEXT_SESSION_PRODUCTION_SCHEMA_INVALID")
    core_ref = publication["proof_ref"]
    core = _read(journal, core_ref)
    base = journal.root / "calendar-future"
    if publication_ref["path"] != str(
        base / "publications" / (core_ref["sha256"] + ".json")
    ) or core_ref["path"] != str(base / "proofs" / (core_ref["sha256"] + ".json")):
        raise ContractError("NEXT_SESSION_PUBLICATION_PATH_INVALID")
    if (
        type(core) is not dict
        or set(core) != _CORE_FIELDS
        or core.get("schema_version") != PROOF_SCHEMA
        or core.get("synthetic") is not False
    ):
        raise ContractError("NEXT_SESSION_PRODUCTION_SCHEMA_INVALID")

    def native_document(ref):
        if type(ref) is not dict or set(ref) != {"relative_path", "byte_sha256"}:
            raise ContractError("NEXT_SESSION_NATIVE_REF_INVALID")
        return _read(
            journal,
            {
                "path": str(base / "captures" / PurePosixPath(ref["relative_path"])),
                "sha256": ref["byte_sha256"],
            },
        )

    info = inspect_next_session_capture(
        workspace=workspace,
        eod_trade_date=eod_trade_date,
        execution=native_document(core["execution_ref"]),
        execution_ref=core["execution_ref"],
        success=native_document(core["success_ref"]),
        success_ref=core["success_ref"],
    )
    transport_ref, transport = read_transport_evidence(
        journal, info, core["transport_evidence_ref"]
    )
    provenance_ref = core["acquisition_provenance_ref"]
    provenance = _read(journal, provenance_ref)
    if provenance_ref["path"] != str(
        base / "production-provenance" / (core["execution_ref"]["byte_sha256"] + ".json")
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_PATH_INVALID")
    _validate_provenance(info, transport_ref, transport, provenance)
    if core != _core(info, provenance_ref, transport_ref) or not _false_authority(
        core["authority"]
    ):
        raise ContractError("NEXT_SESSION_PROOF_DOES_NOT_REPLAY")
    times = [
        info["execution"]["payload"]["observed_started_at"],
        info["execution"]["payload"]["observed_completed_at"],
        info["success"]["payload"]["observed_completed_at"],
        transport["recorded_at"],
        provenance["recorded_at"],
        publication["proof_sealed_at"],
        _now(),
    ]
    if any(utc_stamp(a) > utc_stamp(b) for a, b in zip(times, times[1:])):
        raise ContractError("NEXT_SESSION_PROOF_CHRONOLOGY_INVALID")
    if (
        _read(journal, publication_ref) != publication
        or _read(journal, core_ref) != core
        or _read(journal, provenance_ref) != provenance
        or read_transport_evidence(journal, info, transport_ref)[1] != transport
    ):
        raise ContractError("NEXT_SESSION_PROOF_CHANGED_DURING_READ")
    return {
        "publication_ref": dict(publication_ref),
        "proof": core,
        "proof_sealed_at": publication["proof_sealed_at"],
        "projection": info["projection"],
        "calendar_policy": info["calendar_policy"],
        "synthetic": False,
        "live_eligible": True,
        "consumer_admission": False,
    }
