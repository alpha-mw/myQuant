"""Immutable future Calendar proof custody with exact versioned read dispatch.

Live acquisition/provenance must use its separate installed transport route; this
module's synthetic publisher cannot assert LIVE or accept a caller clock.
"""

from datetime import datetime, timezone
from pathlib import PurePosixPath

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from .next_session_calendar import inspect_next_session_capture

PROVENANCE_SCHEMA = "cn-calendar-acquisition-provenance.v1"
PROOF_SCHEMA = "cn-next-session-calendar-proof.v1"
PUBLICATION_SCHEMA = "cn-next-session-calendar-proof-publication.v1"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(journal, ref):
    validate_ref(ref)
    if not ref["path"].startswith(str(journal.root / "calendar-future") + "/"):
        raise ContractError("NEXT_SESSION_PROOF_PATH_INVALID")
    stored = journal.storage.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise ContractError("NEXT_SESSION_PROOF_SHA_MISMATCH")
    return parse_canonical_json_bytes(stored.data)


def _write(journal, path, value):
    journal._require_lock()
    stored = journal.storage.write(str(path), canonical_json_bytes(value))
    return {"path": str(path), "sha256": stored.byte_sha256}


def _provenance_identity(info):
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "execution_ref": info["execution_ref"],
        "success_ref": info["success_ref"],
        "release_ref": info["execution"]["payload"]["deployed_release_ref"],
        "issuer_route": "SYNTHETIC_FIXTURE",
        "transport_mode": "SYNTHETIC_TRANSPORT",
        "observed_network_call_count": info["execution"]["payload"]["network_call_count"],
        "authority": FALSE_AUTHORITY,
    }


def _validate_provenance(info, value):
    identity = _provenance_identity(info)
    if (
        type(value) is not dict
        or set(value) != {*identity, "recorded_at"}
        or not _false_authority(value.get("authority"))
        or type(value.get("observed_network_call_count")) is not int
        or any(value[k] != v for k, v in identity.items())
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_INVALID_OR_ROUTE_UNAVAILABLE")
    if (
        not utc_stamp(info["success"]["payload"]["observed_completed_at"])
        <= utc_stamp(value["recorded_at"])
        <= utc_stamp(_now())
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_CHRONOLOGY_INVALID")


def _core(info, provenance_ref):
    fields = (
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
    )
    return {
        "schema_version": PROOF_SCHEMA,
        **{k: info[k] for k in fields},
        "release_ref": info["execution"]["payload"]["deployed_release_ref"],
        "capture_started_at": info["execution"]["payload"]["observed_started_at"],
        "capture_completed_at": info["success"]["payload"]["observed_completed_at"],
        "acquisition_provenance_ref": provenance_ref,
        "synthetic": True,
        "authority": FALSE_AUTHORITY,
    }


def publish_synthetic_next_session_proof(
    *,
    workspace: str,
    eod_trade_date: str,
    execution: dict,
    execution_ref: dict,
    success: dict,
    success_ref: dict,
) -> dict:
    """Publish a native-validated test/research proof; never a live-mode shortcut."""
    journal = DailyJournal(workspace, eod_trade_date)
    with journal.locked():
        if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
            raise ContractError("NEXT_SESSION_EOD_ALREADY_COMPLETED")
        info = inspect_next_session_capture(
            workspace=workspace,
            eod_trade_date=eod_trade_date,
            execution=execution,
            execution_ref=execution_ref,
            success=success,
            success_ref=success_ref,
        )
        base = journal.root / "calendar-future"
        identity = _provenance_identity(info)
        path = base / "provenance" / (execution_ref["byte_sha256"] + ".json")
        existing = journal.storage.read(str(path))
        if existing is None:
            provenance_ref = _write(journal, path, {**identity, "recorded_at": _now()})
        else:
            provenance_ref = {"path": str(path), "sha256": existing.byte_sha256}
            provenance = _read(journal, provenance_ref)
            if set(provenance) != {*identity, "recorded_at"} or any(
                provenance[k] != v for k, v in identity.items()
            ):
                raise ContractError("NEXT_SESSION_PROVENANCE_CONFLICT")
        _validate_provenance(info, _read(journal, provenance_ref))
        core = _core(info, provenance_ref)
        import hashlib

        digest = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
        core_ref = _write(journal, base / "proofs" / (digest + ".json"), core)
        publication_path = base / "publications" / (digest + ".json")
        existing = journal.storage.read(str(publication_path))
        if existing is None:
            receipt = {
                "schema_version": PUBLICATION_SCHEMA,
                "proof_ref": core_ref,
                "proof_sealed_at": _now(),
                "authority": FALSE_AUTHORITY,
            }
            publication_ref = _write(journal, publication_path, receipt)
        else:
            publication_ref = {"path": str(publication_path), "sha256": existing.byte_sha256}
        read_next_session_proof(
            workspace=workspace, eod_trade_date=eod_trade_date, publication_ref=publication_ref
        )
        return publication_ref


def read_next_session_proof(*, workspace: str, eod_trade_date: str, publication_ref: dict) -> dict:
    """Read exact native custody; caller must still bind the proof into EOD/Morning."""
    journal = DailyJournal(workspace, eod_trade_date)
    publication = _read(journal, publication_ref)
    if type(publication) is dict and publication.get("schema_version") == (
        "cn-next-session-calendar-proof-publication.v2"
    ):
        from .production_calendar_proof import read_production_next_session_proof

        return read_production_next_session_proof(
            workspace=workspace, eod_trade_date=eod_trade_date, publication_ref=publication_ref
        )
    if (
        set(publication) != {"schema_version", "proof_ref", "proof_sealed_at", "authority"}
        or publication["schema_version"] != PUBLICATION_SCHEMA
        or not _false_authority(publication["authority"])
    ):
        raise ContractError("NEXT_SESSION_PUBLICATION_INVALID")
    core_ref = publication["proof_ref"]
    expected_path = str(
        journal.root / "calendar-future/publications" / (core_ref["sha256"] + ".json")
    )
    if publication_ref["path"] != expected_path or core_ref["path"] != str(
        journal.root / "calendar-future/proofs" / (core_ref["sha256"] + ".json")
    ):
        raise ContractError("NEXT_SESSION_PUBLICATION_PATH_INVALID")
    core = _read(journal, core_ref)
    if (
        type(core) is not dict
        or core.get("schema_version") != PROOF_SCHEMA
        or core.get("synthetic") is not True
    ):
        raise ContractError("NEXT_SESSION_PROOF_SCHEMA_MISMATCH")

    def native_document(ref):
        if type(ref) is not dict or set(ref) != {"relative_path", "byte_sha256"}:
            raise ContractError("NEXT_SESSION_NATIVE_REF_INVALID")
        validate_ref({"path": ref["relative_path"], "sha256": ref["byte_sha256"]})
        return _read(
            journal,
            {
                "path": str(
                    journal.root / "calendar-future/captures" / PurePosixPath(ref["relative_path"])
                ),
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
    provenance_ref = core["acquisition_provenance_ref"]
    provenance = _read(journal, provenance_ref)
    if provenance_ref["path"] != str(
        journal.root
        / "calendar-future/provenance"
        / (core["execution_ref"]["byte_sha256"] + ".json")
    ):
        raise ContractError("NEXT_SESSION_PROVENANCE_PATH_INVALID")
    _validate_provenance(info, provenance)
    if (
        core.get("synthetic") is not True
        or not _false_authority(core.get("authority"))
        or core != _core(info, provenance_ref)
    ):
        raise ContractError("NEXT_SESSION_PROOF_DOES_NOT_REPLAY")
    times = [
        info["execution"]["payload"]["observed_started_at"],
        info["execution"]["payload"]["observed_completed_at"],
        info["success"]["payload"]["observed_completed_at"],
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
    ):
        raise ContractError("NEXT_SESSION_PROOF_CHANGED_DURING_READ")
    return {
        "publication_ref": dict(publication_ref),
        "proof": core,
        "proof_sealed_at": publication["proof_sealed_at"],
        "projection": info["projection"],
        "calendar_policy": info["calendar_policy"],
        "synthetic": True,
        "live_eligible": False,
        "consumer_admission": False,
    }
