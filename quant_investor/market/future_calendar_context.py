"""Versioned opt-in future Calendar context and retained recovery state."""

from pathlib import Path
import re

from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import _validate_day

CONTEXT_SCHEMA = "cn-daily-factor-loop.v2"
STATE_SCHEMA = "cn-daily-factor-state.v2"
PRODUCTION_CONTEXT_SCHEMA = "cn-daily-factor-loop.v3"
PRODUCTION_STATE_SCHEMA = "cn-daily-factor-state.v3"
FUTURE_CONTEXT_SCHEMAS = frozenset({CONTEXT_SCHEMA, PRODUCTION_CONTEXT_SCHEMA})
FUTURE_STATE_SCHEMAS = frozenset({STATE_SCHEMA, PRODUCTION_STATE_SCHEMA})
PRODUCTION_MODE = "PRODUCTION_INSTALLED_CAPTURE"
CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "release_install_input_ref",
        "release_repository_root",
        "release_commit",
        "calendar_capture_parent",
        "initial_calendar_receipt_ref",
        "next_session_calendar_mode",
    }
)
STATE_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "trade_date",
        "context_sha256",
        "calendar_receipt_ref",
        "core_checkpoint_ref",
        "core_observation_refs",
        "future_capture_refs",
        "fixture_transport_evidence_ref",
        "next_session_calendar_proof_ref",
        "next_session_calendar_failure_ref",
        "core_handoff_ref",
    }
)
MODES = frozenset({"DISABLED", "SYNTHETIC_FIXTURE_ONLY"})
PRODUCTION_MODES = frozenset({"DISABLED", PRODUCTION_MODE})
PRODUCTION_STATE_FIELDS = (STATE_FIELDS - {"fixture_transport_evidence_ref"}) | {
    "transport_evidence_ref"
}


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _absolute(value):
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        return False
    path = Path(value)
    return path.is_absolute() and str(path) == value and ".." not in path.parts


def validate_loop_context(value):
    """Keep v1 permissive; v2 is exact and never itself proves transport provenance."""
    if type(value) is not dict:
        raise ContractError("DAILY_FACTOR_CONTEXT_SCHEMA_INVALID")
    if value.get("schema_version") == "cn-daily-factor-loop.v1":
        return value
    schema = value.get("schema_version")
    if (
        type(schema) is not str
        or schema not in FUTURE_CONTEXT_SCHEMAS
        or set(value) != CONTEXT_FIELDS
    ):
        raise ContractError("DAILY_FACTOR_CONTEXT_SCHEMA_INVALID")
    validate_ref(value["release_install_input_ref"])
    if value["initial_calendar_receipt_ref"] is not None:
        _state_ref(value["initial_calendar_receipt_ref"], absolute_allowed=True)
    if (
        not _absolute(value["release_repository_root"])
        or not _absolute(value["calendar_capture_parent"])
        or type(value["release_commit"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", value["release_commit"]) is None
        or type(value["next_session_calendar_mode"]) is not str
        or value["next_session_calendar_mode"]
        not in (PRODUCTION_MODES if schema == PRODUCTION_CONTEXT_SCHEMA else MODES)
    ):
        raise ContractError("DAILY_FACTOR_CONTEXT_FIELDS_INVALID")
    return value


def future_mode(value):
    validate_loop_context(value)
    return (
        value["next_session_calendar_mode"]
        if value["schema_version"] in FUTURE_CONTEXT_SCHEMAS
        else "DISABLED"
    )


def _state_ref(value, *, absolute_allowed=False):
    if type(value) is not dict or set(value) != {"path", "sha256"} or not _sha(value["sha256"]):
        raise ContractError("DAILY_FACTOR_STATE_REF_INVALID")
    if absolute_allowed and _absolute(value["path"]):
        return
    validate_ref(value)


def _native_ref(value, leaf):
    if type(value) is not dict or set(value) != {"relative_path", "byte_sha256"}:
        raise ContractError("DAILY_FACTOR_CAPTURE_REF_INVALID")
    _state_ref({"path": value["relative_path"], "sha256": value["byte_sha256"]})
    path = Path(value["relative_path"])
    if len(path.parts) != 2 or path.name != leaf:
        raise ContractError("DAILY_FACTOR_CAPTURE_REF_INVALID")


def validate_future_state(value, *, context_sha256, trade_date, mode, context_schema=None):
    """Validate exact structural state before custody replay or any recovery write."""
    production = type(value) is dict and value.get("schema_version") == PRODUCTION_STATE_SCHEMA
    expected_schema = PRODUCTION_STATE_SCHEMA if production else STATE_SCHEMA
    if (
        type(value) is not dict
        or set(value) != (PRODUCTION_STATE_FIELDS if production else STATE_FIELDS)
        or value.get("schema_version") != expected_schema
        or not _sha(context_sha256)
        or value["context_sha256"] != context_sha256
        or value["trade_date"] != trade_date
        or mode not in (PRODUCTION_MODES if production else MODES)
        or (
            context_schema is not None
            and context_schema != (PRODUCTION_CONTEXT_SCHEMA if production else CONTEXT_SCHEMA)
        )
    ):
        raise ContractError("DAILY_FACTOR_STATE_INVALID")
    _validate_day(trade_date)
    phase = value["phase"]
    phases = {"CORE_READY", "FUTURE_BOUND", "CORE_PUBLISHED"} | (
        {"CAPTURE_SEALED", "TRANSPORT_BOUND"} if production else {"CAPTURE_BOUND"}
    )
    if phase not in phases:
        raise ContractError("DAILY_FACTOR_STATE_PHASE_INVALID")
    for key in ("calendar_receipt_ref", "core_checkpoint_ref"):
        _state_ref(value[key], absolute_allowed=True)
    observations = value["core_observation_refs"]
    if type(observations) is not dict or set(observations) != {"LOW", "W80"}:
        raise ContractError("DAILY_FACTOR_STATE_OBSERVATIONS_INVALID")
    for ref in observations.values():
        _state_ref(ref, absolute_allowed=True)
    keys = (
        "transport_evidence_ref" if production else "fixture_transport_evidence_ref",
        "next_session_calendar_proof_ref",
        "next_session_calendar_failure_ref",
        "core_handoff_ref",
    )
    for key in keys:
        if value[key] is not None:
            _state_ref(value[key])
    capture = value["future_capture_refs"]
    if capture is not None:
        if type(capture) is not dict or set(capture) != {"execution_ref", "success_ref"}:
            raise ContractError("DAILY_FACTOR_CAPTURE_REF_INVALID")
        _native_ref(capture["execution_ref"], "capture-execution.json")
        _native_ref(capture["success_ref"], "capture-success.json")
        if (
            Path(capture["execution_ref"]["relative_path"]).parent
            != Path(capture["success_ref"]["relative_path"]).parent
        ):
            raise ContractError("DAILY_FACTOR_CAPTURE_REF_INVALID")
    evidence, proof, failure, handoff = (value[k] for k in keys)
    if proof is not None and failure is not None:
        raise ContractError("NEXT_SESSION_PROOF_FAILURE_CONFLICT")
    if phase == "CORE_READY":
        valid = all(value[k] is None for k in (*keys, "future_capture_refs"))
    elif phase == "CAPTURE_SEALED":
        valid = (
            production
            and mode == PRODUCTION_MODE
            and capture is not None
            and evidence is None
            and proof is None
            and failure is None
            and handoff is None
        )
    elif phase in {"CAPTURE_BOUND", "TRANSPORT_BOUND"}:
        valid = (
            mode == (PRODUCTION_MODE if production else "SYNTHETIC_FIXTURE_ONLY")
            and capture is not None
            and evidence is not None
            and proof is None
            and failure is None
            and handoff is None
        )
    else:
        valid = (
            (proof is None and failure is None and capture is None and evidence is None)
            if mode == "DISABLED"
            else ((proof is not None) != (failure is not None))
        )
        valid = valid and (proof is None or (capture is not None and evidence is not None))
        valid = valid and ((handoff is not None) == (phase == "CORE_PUBLISHED"))
    if not valid:
        raise ContractError("DAILY_FACTOR_STATE_PHASE_BINDING_INVALID")
    return value
