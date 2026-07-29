"""Bounded, research-only V17 v4 Shadow factor-set control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from importlib import resources
import math
import os
from pathlib import PurePosixPath
import re
from typing import Any, Final

from quant_investor.factors.governance_literature_incubator_v4 import (
    candidate_catalog_artifact_v4,
    candidate_catalog_v4,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource

from .source_storage import (
    EMPTY_SHA256,
    SourceCASMismatch,
    SourceExactOnceConflict,
    SourceStore,
    SourceStorageError,
    SourceStorageSecurityError,
    StoredBytes,
    WriteResult,
)

FACTOR_SET_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set.v1"
POINTER_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set-pointer.v1"
INPUT_BUNDLE_VERSION: Final = "myquant.v17.v4.research-factor-input-bundle.v1"
FACTOR_SET_ROOT: Final = PurePosixPath("data/private/v17_v4_sources/research_factor_sets")
FACTOR_SET_POINTER: Final = FACTOR_SET_ROOT / "_current.json"
FACTOR_SET_LOCK: Final = FACTOR_SET_ROOT / ".current.lock"
MAX_FACTOR_COUNT: Final = 8
GATE_WEIGHTS: Final = {
    "source_review_accepted": 40,
    "runtime_adapter_supported": 20,
    "pit_inputs_current": 20,
    "required_lookback_complete": 10,
}
GATE_ORDER: Final = tuple(GATE_WEIGHTS)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}

_IDENTIFIER_RE: Final = re.compile(
    r"^[a-z0-9][a-z0-9_.:-]{0,127}$",
    re.ASCII,
)
_PATH_ID_RE: Final = re.compile(
    r"^[a-z0-9][a-z0-9_.-]{0,127}$",
    re.ASCII,
)
_FIELD_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$", re.ASCII)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_CUTOFF_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_REF_KEYS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}
_SELECTION_KEYS: Final = {
    "candidate_catalog_sha256",
    "catalog_resource_sha256",
    "definition_sha256",
    "implementation_resource_sha256",
    "implementation_sha256",
    "name",
    "selection_gates",
}
_FACTOR_KEYS: Final = {
    "definition",
    "definition_sha256",
    "direction",
    "family",
    "implementation",
    "implementation_resource_sha256",
    "implementation_sha256",
    "lookback",
    "name",
    "params",
    "required_fields",
    "selection_gates",
    "selection_score",
    "slot",
}
_FACTOR_SET_KEYS: Final = {
    "audit_session",
    "authority",
    "canary_evidence_eligible",
    "candidate_catalog_sha256",
    "catalog_resource_sha256",
    "cutoff",
    "effective_from_session",
    "eligible_distinct_slot_count",
    "eligible_factor_count",
    "factor_set_id",
    "formal_activation_eligible",
    "implementation_resource_sha256",
    "monthly_audit_ref",
    "performance_evidence_eligible",
    "previous_factor_set_ref",
    "protocol_version",
    "selected_at",
    "selected_factors",
    "selection_policy_sha256",
    "semantic_sha256",
    "shadow_only",
    "strategy_id",
    "target_cardinality",
    "version",
}
_POINTER_KEYS: Final = {
    "authority",
    "canary_evidence_eligible",
    "cutoff",
    "effective_from_session",
    "factor_set_ref",
    "formal_activation_eligible",
    "performance_evidence_eligible",
    "pointer_id",
    "previous_pointer_sha256",
    "protocol_version",
    "selected_at",
    "semantic_sha256",
    "shadow_only",
    "strategy_id",
    "version",
}
_SLICE_KEYS: Final = {
    "available_at",
    "field_name",
    "first_session",
    "last_session",
    "row_count",
    "slice_ref",
}
_INPUT_BUNDLE_KEYS: Final = {
    "authority",
    "bundle_id",
    "canary_evidence_eligible",
    "cutoff",
    "decision_session",
    "factor_set_ref",
    "field_slices",
    "formal_activation_eligible",
    "performance_evidence_eligible",
    "protocol_version",
    "required_fields",
    "research_source_locator_ref",
    "run_id",
    "semantic_sha256",
    "shadow_only",
    "strategy_id",
    "version",
}


class ResearchFactorSetError(RuntimeError):
    """A research factor-set artifact or transition failed closed."""

    exit_code = 2


class ResearchFactorSetCrash(ResearchFactorSetError):
    """Testable crash boundary after an already durable write."""


@dataclass(frozen=True)
class ResearchFactorSetPublication:
    factor_set_ref: Mapping[str, str]
    pointer_ref: Mapping[str, str]
    recovered: bool


@dataclass(frozen=True)
class ResearchFactorSetState:
    pointer: Mapping[str, Any]
    factor_set: Mapping[str, Any]
    pointer_ref: Mapping[str, str]
    factor_set_ref: Mapping[str, str]


def _blocked(reason: str) -> ResearchFactorSetError:
    return ResearchFactorSetError(f"V17_V4_RESEARCH_FACTOR_SET_BLOCKED:{reason}")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise _blocked(f"{label}_fields")
    return value


def _identifier(value: Any, *, label: str, path_safe: bool = False) -> str:
    pattern = _PATH_ID_RE if path_safe else _IDENTIFIER_RE
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise _blocked(label)
    return value


def _field(value: Any, *, label: str = "field_name") -> str:
    if type(value) is not str or _FIELD_RE.fullmatch(value) is None:
        raise _blocked(label)
    return value


def _digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise _blocked(label)
    return value


def _cas(value: Any, *, label: str) -> str:
    if value == EMPTY_SHA256:
        return value
    return _digest(value, label=label)


def _day(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise _blocked(label)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _blocked(label) from exc
    if parsed.isoformat() != value:
        raise _blocked(label)
    return value


def _cutoff(value: Any, *, label: str) -> str:
    if type(value) is not str or _CUTOFF_RE.fullmatch(value) is None:
        raise _blocked(label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _blocked(label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise _blocked(label)
    return value


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise _blocked(label)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise _blocked(label)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _blocked(label) from exc
    return value


def _artifact_ref(
    value: Any,
    *,
    label: str,
    strategy_id: str | None = None,
) -> dict[str, str]:
    ref = _require_keys(value, _REF_KEYS, label=label)
    normalized = {
        "artifact_id": _identifier(
            ref["artifact_id"],
            label=f"{label}_artifact_id",
        ),
        "artifact_version": _identifier(
            ref["artifact_version"],
            label=f"{label}_artifact_version",
        ),
        "byte_sha256": _digest(
            ref["byte_sha256"],
            label=f"{label}_byte_sha256",
        ),
        "cutoff": _cutoff(ref["cutoff"], label=f"{label}_cutoff"),
        "relative_path": _canonical_relative_path(
            ref["relative_path"],
            label=f"{label}_relative_path",
        ),
        "semantic_sha256": _digest(
            ref["semantic_sha256"],
            label=f"{label}_semantic_sha256",
        ),
        "strategy_id": _identifier(
            ref["strategy_id"],
            label=f"{label}_strategy_id",
        ),
    }
    if strategy_id is not None and normalized["strategy_id"] != strategy_id:
        raise _blocked(f"{label}_strategy")
    return normalized


def _sealed(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = dict(value)
    declared = _digest(
        result.get("semantic_sha256"),
        label=f"{label}_semantic_sha256",
    )
    unsealed = dict(result)
    unsealed.pop("semantic_sha256")
    expected = seal_semantic(unsealed)
    if expected["semantic_sha256"] != declared:
        raise _blocked(f"{label}_semantic_sha256_mismatch")
    return result


def _decimal(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise _blocked(label)
    try:
        number = float(value)
    except ValueError as exc:
        raise _blocked(label) from exc
    if not math.isfinite(number) or format(number, ".17g") != value:
        raise _blocked(label)
    return value


def _module_resource_sha256() -> str:
    resource = resources.files("quant_investor.factors").joinpath(
        "governance_literature_incubator_v4.py"
    )
    return _sha(resource.read_bytes())


_CATALOG = candidate_catalog_v4()
_CATALOG_BY_NAME: Final = {str(row["name"]): row for row in _CATALOG}
CANDIDATE_CATALOG_SHA256: Final = _sha(canonical_bytes(_CATALOG))
CATALOG_RESOURCE_SHA256: Final = _sha(canonical_resource_bytes(candidate_catalog_artifact_v4()))
IMPLEMENTATION_RESOURCE_SHA256: Final = _module_resource_sha256()
_IMPLEMENTATION_SHA_BY_NAME: Final = {
    name: _sha(
        canonical_bytes(
            {
                "implementation": row["implementation"],
                "name": name,
            }
        )
    )
    for name, row in _CATALOG_BY_NAME.items()
}
_SELECTION_POLICY = {
    "eligible_rule": (
        "all_gates_pass_and_exact_catalog_definition_implementation_" "resource_hashes_match"
    ),
    "gate_weights": dict(GATE_WEIGHTS),
    "global_order": ["selection_score_desc", "name_ascii_asc"],
    "max_factor_count": MAX_FACTOR_COUNT,
    "per_slot_order": ["selection_score_desc", "name_ascii_asc"],
    "slot_bonus": 0,
}
SELECTION_POLICY_SHA256: Final = _sha(canonical_bytes(_SELECTION_POLICY))


def research_factor_catalog_bindings() -> dict[str, Any]:
    """Return exact caller bindings without granting selection authority."""

    return {
        "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
        "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "selection_policy_sha256": SELECTION_POLICY_SHA256,
        "factors": {
            name: {
                "definition_sha256": str(row["definition_sha256"]),
                "implementation_sha256": _IMPLEMENTATION_SHA_BY_NAME[name],
                "slot": str(row["slot"]),
            }
            for name, row in sorted(_CATALOG_BY_NAME.items())
        },
    }


def _selection_gates(value: Any, *, label: str) -> list[dict[str, Any]]:
    if type(value) is not dict or set(value) != set(GATE_ORDER):
        raise _blocked(f"{label}_fields")
    if any(type(value[gate_id]) is not bool for gate_id in GATE_ORDER):
        raise _blocked(f"{label}_passed")
    return [
        {
            "gate_id": gate_id,
            "passed": value[gate_id],
            "weight": GATE_WEIGHTS[gate_id],
        }
        for gate_id in GATE_ORDER
    ]


def _validate_output_gates(
    value: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(GATE_ORDER):
        raise _blocked(f"{label}_shape")
    normalized: list[dict[str, Any]] = []
    for index, (gate_id, weight) in enumerate(GATE_WEIGHTS.items()):
        row = _require_keys(
            value[index],
            {"gate_id", "passed", "weight"},
            label=f"{label}_{index}",
        )
        if (
            row["gate_id"] != gate_id
            or type(row["passed"]) is not bool
            or type(row["weight"]) is not int
            or row["weight"] != weight
        ):
            raise _blocked(f"{label}_{index}_value")
        normalized.append(dict(row))
    return normalized


def _normalize_selection_row(value: Any) -> dict[str, Any]:
    row = _require_keys(value, _SELECTION_KEYS, label="selection_row")
    name = _identifier(row["name"], label="selection_name")
    catalog = _CATALOG_BY_NAME.get(name)
    if catalog is None:
        raise _blocked("selection_name_unsupported")
    expected = {
        "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
        "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
        "definition_sha256": str(catalog["definition_sha256"]),
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "implementation_sha256": _IMPLEMENTATION_SHA_BY_NAME[name],
    }
    for field_name, expected_value in expected.items():
        observed = _digest(row[field_name], label=field_name)
        if observed != expected_value:
            raise _blocked(f"selection_{field_name}_mismatch")
    gates = _selection_gates(
        row["selection_gates"],
        label="selection_gates",
    )
    score = sum(gate["weight"] for gate in gates if gate["passed"])
    return {
        "catalog": catalog,
        "eligible": all(gate["passed"] for gate in gates),
        "gates": gates,
        "name": name,
        "score": score,
    }


def _selected_factor(
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = normalized["catalog"]
    return {
        "definition": str(catalog["definition"]),
        "definition_sha256": str(catalog["definition_sha256"]),
        "direction": format(float(catalog["direction"]), ".17g"),
        "family": str(catalog["family"]),
        "implementation": str(catalog["implementation"]),
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "implementation_sha256": _IMPLEMENTATION_SHA_BY_NAME[str(catalog["name"])],
        "lookback": int(catalog["lookback"]),
        "name": str(catalog["name"]),
        "params": dict(catalog["params"]),
        "required_fields": list(catalog["required_fields"]),
        "selection_gates": [dict(gate) for gate in normalized["gates"]],
        "selection_score": int(normalized["score"]),
        "slot": str(catalog["slot"]),
    }


def _effective_session(
    audit_session: str,
    open_sessions: Sequence[str],
) -> str:
    if isinstance(open_sessions, (str, bytes)) or not open_sessions:
        raise _blocked("open_sessions")
    normalized = [_day(value, label="open_session") for value in open_sessions]
    if normalized != sorted(set(normalized)):
        raise _blocked("open_sessions_order")
    try:
        return next(session for session in normalized if session > audit_session)
    except StopIteration as exc:
        raise _blocked("effective_open_session_missing") from exc


def build_research_shadow_factor_set(
    *,
    factor_set_id: str,
    strategy_id: str,
    cutoff: str,
    audit_session: str,
    selected_at: str,
    open_sessions: Sequence[str],
    monthly_audit_ref: Mapping[str, Any],
    previous_factor_set_ref: Mapping[str, Any] | None,
    selection_rows: Sequence[Mapping[str, Any]],
    expected_candidate_catalog_sha256: str,
    expected_catalog_resource_sha256: str,
    expected_implementation_resource_sha256: str,
) -> dict[str, Any]:
    """Select one deterministic, bounded factor set from exact catalog rows."""

    set_id = _identifier(
        factor_set_id,
        label="factor_set_id",
        path_safe=True,
    )
    strategy = _identifier(strategy_id, label="strategy_id")
    cutoff_value = _cutoff(cutoff, label="cutoff")
    selected = _cutoff(selected_at, label="selected_at")
    audit = _day(audit_session, label="audit_session")
    if selected != cutoff_value or selected[:10] < audit:
        raise _blocked("selection_timing")
    audit_ref = _artifact_ref(
        monthly_audit_ref,
        label="monthly_audit_ref",
        strategy_id=strategy,
    )
    if audit_ref["cutoff"] != selected:
        raise _blocked("monthly_audit_ref_cutoff")
    previous = (
        None
        if previous_factor_set_ref is None
        else _artifact_ref(
            previous_factor_set_ref,
            label="previous_factor_set_ref",
            strategy_id=strategy,
        )
    )
    if previous is not None and previous["artifact_version"] != FACTOR_SET_VERSION:
        raise _blocked("previous_factor_set_ref_version")
    expected_hashes = (
        (
            expected_candidate_catalog_sha256,
            CANDIDATE_CATALOG_SHA256,
            "candidate_catalog_sha256",
        ),
        (
            expected_catalog_resource_sha256,
            CATALOG_RESOURCE_SHA256,
            "catalog_resource_sha256",
        ),
        (
            expected_implementation_resource_sha256,
            IMPLEMENTATION_RESOURCE_SHA256,
            "implementation_resource_sha256",
        ),
    )
    for observed, exact, label in expected_hashes:
        if _digest(observed, label=label) != exact:
            raise _blocked(f"{label}_mismatch")
    if (
        isinstance(selection_rows, (str, bytes))
        or not selection_rows
        or len(selection_rows) > len(_CATALOG_BY_NAME)
    ):
        raise _blocked("selection_rows")
    normalized = [_normalize_selection_row(row) for row in selection_rows]
    names = [str(row["name"]) for row in normalized]
    if len(names) != len(set(names)):
        raise _blocked("selection_names_duplicate")
    eligible = [row for row in normalized if row["eligible"]]
    winners: list[Mapping[str, Any]] = []
    by_slot: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible:
        by_slot.setdefault(str(row["catalog"]["slot"]), []).append(row)
    for slot in sorted(by_slot):
        winners.append(
            sorted(
                by_slot[slot],
                key=lambda row: (-int(row["score"]), str(row["name"])),
            )[0]
        )
    target = min(MAX_FACTOR_COUNT, len(by_slot))
    if target < 1:
        raise _blocked("no_eligible_distinct_slot")
    chosen = sorted(
        winners,
        key=lambda row: (-int(row["score"]), str(row["name"])),
    )[:target]
    factor_rows = [_selected_factor(row) for row in chosen]
    document = seal_semantic(
        {
            "audit_session": audit,
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
            "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
            "cutoff": cutoff_value,
            "effective_from_session": _effective_session(
                audit,
                open_sessions,
            ),
            "eligible_distinct_slot_count": len(by_slot),
            "eligible_factor_count": len(eligible),
            "factor_set_id": set_id,
            "formal_activation_eligible": False,
            "implementation_resource_sha256": (IMPLEMENTATION_RESOURCE_SHA256),
            "monthly_audit_ref": audit_ref,
            "performance_evidence_eligible": False,
            "previous_factor_set_ref": previous,
            "protocol_version": PROTOCOL_VERSION,
            "selected_at": selected,
            "selected_factors": factor_rows,
            "selection_policy_sha256": SELECTION_POLICY_SHA256,
            "shadow_only": True,
            "strategy_id": strategy,
            "target_cardinality": target,
            "version": FACTOR_SET_VERSION,
        }
    )
    return validate_research_shadow_factor_set(document)


def _validate_selected_factor(value: Any) -> dict[str, Any]:
    row = _require_keys(value, _FACTOR_KEYS, label="selected_factor")
    name = _identifier(row["name"], label="selected_factor_name")
    catalog = _CATALOG_BY_NAME.get(name)
    if catalog is None:
        raise _blocked("selected_factor_name_unsupported")
    gates = _validate_output_gates(
        row["selection_gates"],
        label=f"selected_factor_{name}_gates",
    )
    expected = {
        "definition": str(catalog["definition"]),
        "definition_sha256": str(catalog["definition_sha256"]),
        "direction": format(float(catalog["direction"]), ".17g"),
        "family": str(catalog["family"]),
        "implementation": str(catalog["implementation"]),
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "implementation_sha256": _IMPLEMENTATION_SHA_BY_NAME[name],
        "lookback": int(catalog["lookback"]),
        "name": name,
        "params": dict(catalog["params"]),
        "required_fields": list(catalog["required_fields"]),
        "selection_gates": gates,
        "selection_score": sum(gate["weight"] for gate in gates if gate["passed"]),
        "slot": str(catalog["slot"]),
    }
    if dict(row) != expected or not all(gate["passed"] for gate in gates):
        raise _blocked(f"selected_factor_{name}_definition")
    return expected


def validate_research_shadow_factor_set(
    value: Any,
) -> dict[str, Any]:
    """Validate a factor set against the exact currently packaged catalog."""

    document = _require_keys(
        value,
        _FACTOR_SET_KEYS,
        label="factor_set",
    )
    if (
        document["version"] != FACTOR_SET_VERSION
        or document["protocol_version"] != PROTOCOL_VERSION
        or document["authority"] != NO_AUTHORITY
        or document["shadow_only"] is not True
        or document["formal_activation_eligible"] is not False
        or document["canary_evidence_eligible"] is not False
        or document["performance_evidence_eligible"] is not False
    ):
        raise _blocked("factor_set_authority")
    _identifier(
        document["factor_set_id"],
        label="factor_set_id",
        path_safe=True,
    )
    strategy = _identifier(document["strategy_id"], label="strategy_id")
    cutoff_value = _cutoff(document["cutoff"], label="cutoff")
    selected_at = _cutoff(document["selected_at"], label="selected_at")
    audit = _day(document["audit_session"], label="audit_session")
    effective = _day(
        document["effective_from_session"],
        label="effective_from_session",
    )
    if cutoff_value != selected_at or selected_at[:10] < audit or effective <= audit:
        raise _blocked("factor_set_timing")
    audit_ref = _artifact_ref(
        document["monthly_audit_ref"],
        label="monthly_audit_ref",
        strategy_id=strategy,
    )
    if audit_ref["cutoff"] != selected_at:
        raise _blocked("monthly_audit_ref_cutoff")
    previous = document["previous_factor_set_ref"]
    if previous is not None:
        previous_ref = _artifact_ref(
            previous,
            label="previous_factor_set_ref",
            strategy_id=strategy,
        )
        if previous_ref["artifact_version"] != FACTOR_SET_VERSION:
            raise _blocked("previous_factor_set_ref_version")
    exact_hashes = {
        "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
        "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "selection_policy_sha256": SELECTION_POLICY_SHA256,
    }
    for field_name, exact in exact_hashes.items():
        if _digest(document[field_name], label=field_name) != exact:
            raise _blocked(f"factor_set_{field_name}_mismatch")
    selected = document["selected_factors"]
    if type(selected) is not list or not 1 <= len(selected) <= MAX_FACTOR_COUNT:
        raise _blocked("selected_factors_count")
    factors = [_validate_selected_factor(row) for row in selected]
    expected_order = sorted(
        factors,
        key=lambda row: (-row["selection_score"], row["name"]),
    )
    if factors != expected_order:
        raise _blocked("selected_factors_order")
    names = [row["name"] for row in factors]
    slots = [row["slot"] for row in factors]
    if len(names) != len(set(names)) or len(slots) != len(set(slots)):
        raise _blocked("selected_factor_identity_duplicate")
    count_fields = (
        "eligible_factor_count",
        "eligible_distinct_slot_count",
        "target_cardinality",
    )
    if any(type(document[field_name]) is not int for field_name in count_fields):
        raise _blocked("factor_set_counts")
    eligible_count = document["eligible_factor_count"]
    distinct_count = document["eligible_distinct_slot_count"]
    target = document["target_cardinality"]
    if (
        eligible_count < distinct_count
        or distinct_count < 1
        or target != min(MAX_FACTOR_COUNT, distinct_count)
        or len(factors) != target
    ):
        raise _blocked("factor_set_cardinality")
    return _sealed(document, label="factor_set")


def _document_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    identity_field: str,
) -> dict[str, str]:
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": _sha(canonical_resource_bytes(document)),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _pointer_document(
    factor_set: Mapping[str, Any],
    factor_set_ref: Mapping[str, str],
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "cutoff": str(factor_set["cutoff"]),
            "effective_from_session": str(factor_set["effective_from_session"]),
            "factor_set_ref": dict(factor_set_ref),
            "formal_activation_eligible": False,
            "performance_evidence_eligible": False,
            "pointer_id": ("research-factor-set:" + str(factor_set["factor_set_id"])),
            "previous_pointer_sha256": expected_pointer_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "selected_at": str(factor_set["selected_at"]),
            "shadow_only": True,
            "strategy_id": str(factor_set["strategy_id"]),
            "version": POINTER_VERSION,
        }
    )


def validate_research_shadow_factor_set_pointer(
    value: Any,
) -> dict[str, Any]:
    document = _require_keys(value, _POINTER_KEYS, label="factor_set_pointer")
    if (
        document["version"] != POINTER_VERSION
        or document["protocol_version"] != PROTOCOL_VERSION
        or document["authority"] != NO_AUTHORITY
        or document["shadow_only"] is not True
        or document["formal_activation_eligible"] is not False
        or document["canary_evidence_eligible"] is not False
        or document["performance_evidence_eligible"] is not False
    ):
        raise _blocked("factor_set_pointer_authority")
    _identifier(document["pointer_id"], label="pointer_id")
    strategy = _identifier(document["strategy_id"], label="strategy_id")
    cutoff_value = _cutoff(document["cutoff"], label="cutoff")
    selected = _cutoff(document["selected_at"], label="selected_at")
    effective = _day(
        document["effective_from_session"],
        label="effective_from_session",
    )
    factor_set_ref = _artifact_ref(
        document["factor_set_ref"],
        label="factor_set_ref",
        strategy_id=strategy,
    )
    if (
        factor_set_ref["artifact_version"] != FACTOR_SET_VERSION
        or factor_set_ref["cutoff"] != cutoff_value
        or selected != cutoff_value
        or effective <= selected[:10]
    ):
        raise _blocked("factor_set_pointer_binding")
    _cas(
        document["previous_pointer_sha256"],
        label="previous_pointer_sha256",
    )
    return _sealed(document, label="factor_set_pointer")


def build_research_factor_input_bundle(
    *,
    bundle_id: str,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    factor_set: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    research_source_locator_ref: Mapping[str, Any],
    field_slices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the exact run-local slices required by one selected factor set."""

    normalized_set = validate_research_shadow_factor_set(factor_set)
    strategy = _identifier(strategy_id, label="strategy_id")
    cutoff_value = _cutoff(cutoff, label="cutoff")
    session = _day(decision_session, label="decision_session")
    run = _identifier(run_id, label="run_id", path_safe=True)
    if (
        normalized_set["strategy_id"] != strategy
        or normalized_set["cutoff"] != cutoff_value
        or normalized_set["effective_from_session"] > session
    ):
        raise _blocked("input_bundle_factor_set_timing")
    expected_set_ref = _artifact_ref(
        factor_set_ref,
        label="factor_set_ref",
        strategy_id=strategy,
    )
    if (
        expected_set_ref["artifact_version"] != FACTOR_SET_VERSION
        or expected_set_ref["byte_sha256"] != _sha(canonical_resource_bytes(normalized_set))
        or expected_set_ref["semantic_sha256"] != normalized_set["semantic_sha256"]
    ):
        raise _blocked("input_bundle_factor_set_ref")
    locator_ref = _artifact_ref(
        research_source_locator_ref,
        label="research_source_locator_ref",
        strategy_id=strategy,
    )
    if locator_ref["cutoff"] != cutoff_value:
        raise _blocked("input_bundle_locator_cutoff")
    required = sorted(
        {
            field_name
            for factor in normalized_set["selected_factors"]
            for field_name in factor["required_fields"]
        }
    )
    if isinstance(field_slices, (str, bytes)):
        raise _blocked("field_slices")
    slices = [
        _normalize_field_slice(
            row,
            run_id=run,
            strategy_id=strategy,
            cutoff=cutoff_value,
            decision_session=session,
        )
        for row in field_slices
    ]
    if [row["field_name"] for row in slices] != required:
        raise _blocked("field_slices_exact_required_fields")
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "bundle_id": _identifier(bundle_id, label="bundle_id"),
            "canary_evidence_eligible": False,
            "cutoff": cutoff_value,
            "decision_session": session,
            "factor_set_ref": expected_set_ref,
            "field_slices": slices,
            "formal_activation_eligible": False,
            "performance_evidence_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "required_fields": required,
            "research_source_locator_ref": locator_ref,
            "run_id": run,
            "shadow_only": True,
            "strategy_id": strategy,
            "version": INPUT_BUNDLE_VERSION,
        }
    )
    return validate_research_factor_input_bundle(
        document,
        factor_set=normalized_set,
    )


def _normalize_field_slice(
    value: Any,
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
) -> dict[str, Any]:
    row = _require_keys(value, _SLICE_KEYS, label="field_slice")
    field_name = _field(row["field_name"])
    available_at = _cutoff(row["available_at"], label="available_at")
    first = _day(row["first_session"], label="first_session")
    last = _day(row["last_session"], label="last_session")
    if (
        first > last
        or last < decision_session
        or available_at > cutoff
        or type(row["row_count"]) is not int
        or row["row_count"] < 1
    ):
        raise _blocked("field_slice_coverage")
    slice_ref = _artifact_ref(
        row["slice_ref"],
        label="slice_ref",
        strategy_id=strategy_id,
    )
    prefix = f"data/private/v17_v4_runs/{run_id}/" "research_factor_inputs/"
    if (
        slice_ref["cutoff"] != cutoff
        or not slice_ref["relative_path"].startswith(prefix)
        or PurePosixPath(slice_ref["relative_path"]).name != f"{field_name}.parquet"
    ):
        raise _blocked("field_slice_ref")
    return {
        "available_at": available_at,
        "field_name": field_name,
        "first_session": first,
        "last_session": last,
        "row_count": row["row_count"],
        "slice_ref": slice_ref,
    }


def validate_research_factor_input_bundle(
    value: Any,
    *,
    factor_set: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    document = _require_keys(
        value,
        _INPUT_BUNDLE_KEYS,
        label="input_bundle",
    )
    if (
        document["version"] != INPUT_BUNDLE_VERSION
        or document["protocol_version"] != PROTOCOL_VERSION
        or document["authority"] != NO_AUTHORITY
        or document["shadow_only"] is not True
        or document["formal_activation_eligible"] is not False
        or document["canary_evidence_eligible"] is not False
        or document["performance_evidence_eligible"] is not False
    ):
        raise _blocked("input_bundle_authority")
    _identifier(document["bundle_id"], label="bundle_id")
    run = _identifier(document["run_id"], label="run_id", path_safe=True)
    strategy = _identifier(document["strategy_id"], label="strategy_id")
    cutoff_value = _cutoff(document["cutoff"], label="cutoff")
    session = _day(document["decision_session"], label="decision_session")
    set_ref = _artifact_ref(
        document["factor_set_ref"],
        label="factor_set_ref",
        strategy_id=strategy,
    )
    locator_ref = _artifact_ref(
        document["research_source_locator_ref"],
        label="research_source_locator_ref",
        strategy_id=strategy,
    )
    if (
        set_ref["artifact_version"] != FACTOR_SET_VERSION
        or set_ref["cutoff"] != cutoff_value
        or locator_ref["cutoff"] != cutoff_value
    ):
        raise _blocked("input_bundle_refs")
    required = document["required_fields"]
    if (
        type(required) is not list
        or not required
        or required != sorted(set(required))
        or any(_field(item) != item for item in required)
    ):
        raise _blocked("required_fields")
    slices_value = document["field_slices"]
    if type(slices_value) is not list:
        raise _blocked("field_slices")
    slices = [
        _normalize_field_slice(
            row,
            run_id=run,
            strategy_id=strategy,
            cutoff=cutoff_value,
            decision_session=session,
        )
        for row in slices_value
    ]
    if [row["field_name"] for row in slices] != required:
        raise _blocked("field_slices_exact_required_fields")
    if factor_set is not None:
        normalized_set = validate_research_shadow_factor_set(factor_set)
        exact_required = sorted(
            {
                field_name
                for factor in normalized_set["selected_factors"]
                for field_name in factor["required_fields"]
            }
        )
        if (
            normalized_set["strategy_id"] != strategy
            or normalized_set["cutoff"] != cutoff_value
            or normalized_set["effective_from_session"] > session
            or set_ref["byte_sha256"] != _sha(canonical_resource_bytes(normalized_set))
            or set_ref["semantic_sha256"] != normalized_set["semantic_sha256"]
            or required != exact_required
        ):
            raise _blocked("input_bundle_factor_set_binding")
    return _sealed(document, label="input_bundle")


class _ResearchFactorSetWriter(SourceStore):
    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = PurePosixPath(value)
        if not (path == FACTOR_SET_ROOT or FACTOR_SET_ROOT in path.parents):
            raise SourceStorageSecurityError("path is outside the research factor-set root")
        return super()._canonical_path(path)

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        raw: bytes,
    ) -> WriteResult:
        path = self._canonical_path(relative_path)
        immutable_parent = FACTOR_SET_ROOT / "sets"
        if path != FACTOR_SET_LOCK and not (
            path.parent == immutable_parent
            and path.suffix == ".json"
            and _PATH_ID_RE.fullmatch(path.stem) is not None
        ):
            raise SourceStorageSecurityError("research factor-set exact-once path is not permitted")
        return super().write_exact_once(path, raw)

    def replace_cas(
        self,
        relative_path: str | PurePosixPath,
        expected_sha256: str,
        raw: bytes,
    ) -> WriteResult:
        path = self._canonical_path(relative_path)
        if path != FACTOR_SET_POINTER:
            raise SourceStorageSecurityError("only the research factor-set pointer is mutable")
        return super().replace_cas(path, expected_sha256, raw)

    def initialize(self) -> None:
        fd = self._open_directory(FACTOR_SET_ROOT.parts, create=True)
        try:
            sets_fd = self._open_directory(
                (FACTOR_SET_ROOT / "sets").parts,
                create=True,
            )
            os.close(sets_fd)
        finally:
            os.close(fd)


class ResearchFactorSetStore:
    """Only writes immutable sets and the research-only current pointer."""

    def __init__(self, workspace_root: str) -> None:
        self._writer = _ResearchFactorSetWriter(workspace_root)

    @staticmethod
    def _set_path(factor_set_id: str) -> PurePosixPath:
        set_id = _identifier(
            factor_set_id,
            label="factor_set_id",
            path_safe=True,
        )
        return FACTOR_SET_ROOT / "sets" / f"{set_id}.json"

    def publish(
        self,
        factor_set: Mapping[str, Any],
        *,
        expected_pointer_sha256: str,
        crash_after: str | None = None,
    ) -> ResearchFactorSetPublication:
        normalized = validate_research_shadow_factor_set(factor_set)
        expected = _cas(
            expected_pointer_sha256,
            label="expected_pointer_sha256",
        )
        if crash_after not in {None, "set", "cas"}:
            raise _blocked("crash_after")
        set_path = self._set_path(str(normalized["factor_set_id"]))
        set_raw = canonical_resource_bytes(normalized)
        set_ref = _document_ref(
            normalized,
            relative_path=str(set_path),
            identity_field="factor_set_id",
        )
        pointer = _pointer_document(normalized, set_ref, expected)
        pointer = validate_research_shadow_factor_set_pointer(pointer)
        pointer_raw = canonical_resource_bytes(pointer)
        proposed = _sha(pointer_raw)
        self._writer.initialize()
        with self._writer.locked(FACTOR_SET_LOCK):
            current = self._writer.read_optional(FACTOR_SET_POINTER)
            observed = EMPTY_SHA256 if current is None else current.byte_sha256
            self._validate_previous_binding(
                normalized,
                current=current,
                expected=expected,
                observed=observed,
                proposed=proposed,
                pointer_raw=pointer_raw,
            )
            try:
                self._writer.write_exact_once(set_path, set_raw)
            except SourceExactOnceConflict as exc:
                raise _blocked("immutable_factor_set_conflict") from exc
            if self._writer.read(set_path, set_ref["byte_sha256"]) != set_raw:
                raise _blocked("immutable_factor_set_readback")
            if crash_after == "set":
                raise ResearchFactorSetCrash("crash after immutable set")
            recovered = False
            if observed == expected:
                try:
                    self._writer.replace_cas(
                        FACTOR_SET_POINTER,
                        expected,
                        pointer_raw,
                    )
                except SourceCASMismatch as exc:
                    raise _blocked("pointer_third_state") from exc
            elif observed == proposed and current is not None:
                if current.data != pointer_raw:
                    raise _blocked("pointer_hash_collision")
                recovered = True
            else:
                raise _blocked("pointer_third_state")
            if crash_after == "cas":
                raise ResearchFactorSetCrash("crash after pointer CAS")
            if self._writer.read(FACTOR_SET_POINTER, proposed) != pointer_raw:
                raise _blocked("pointer_readback")
        state = self.read_current()
        if state.pointer_ref["byte_sha256"] != proposed or state.factor_set_ref != set_ref:
            raise _blocked("publication_reread")
        return ResearchFactorSetPublication(
            factor_set_ref=set_ref,
            pointer_ref=state.pointer_ref,
            recovered=recovered,
        )

    def _validate_previous_binding(
        self,
        factor_set: Mapping[str, Any],
        *,
        current: StoredBytes | None,
        expected: str,
        observed: str,
        proposed: str,
        pointer_raw: bytes,
    ) -> None:
        previous = factor_set["previous_factor_set_ref"]
        if observed == proposed and current is not None:
            if current.data != pointer_raw:
                raise _blocked("pointer_hash_collision")
            return
        if observed != expected:
            raise _blocked("pointer_third_state")
        if current is None:
            if expected != EMPTY_SHA256 or previous is not None:
                raise _blocked("previous_factor_set_ref_initial")
            return
        current_pointer = _load_pointer(current.data)
        if previous != current_pointer["factor_set_ref"]:
            raise _blocked("previous_factor_set_ref_mismatch")

    def read_current(self) -> ResearchFactorSetState:
        first = self._writer.read_optional(FACTOR_SET_POINTER)
        if first is None:
            raise _blocked("pointer_missing")
        pointer = _load_pointer(first.data)
        factor_set_ref = pointer["factor_set_ref"]
        try:
            raw = self._writer.read(
                factor_set_ref["relative_path"],
                factor_set_ref["byte_sha256"],
            )
        except (SourceCASMismatch, SourceStorageError) as exc:
            raise _blocked("factor_set_exact_read") from exc
        factor_set = _load_factor_set(raw)
        exact_ref = _document_ref(
            factor_set,
            relative_path=factor_set_ref["relative_path"],
            identity_field="factor_set_id",
        )
        if exact_ref != factor_set_ref:
            raise _blocked("factor_set_ref_mismatch")
        second = self._writer.read_optional(FACTOR_SET_POINTER)
        if second is None or second.byte_sha256 != first.byte_sha256 or second.data != first.data:
            raise _blocked("pointer_changed_during_reread")
        pointer_ref = _document_ref(
            pointer,
            relative_path=str(FACTOR_SET_POINTER),
            identity_field="pointer_id",
        )
        return ResearchFactorSetState(
            pointer=pointer,
            factor_set=factor_set,
            pointer_ref=pointer_ref,
            factor_set_ref=exact_ref,
        )


def _load_pointer(raw: bytes) -> dict[str, Any]:
    try:
        value = load_canonical_resource(raw, label=POINTER_VERSION)
    except (TypeError, ValueError) as exc:
        raise _blocked("pointer_canonical_readback") from exc
    return validate_research_shadow_factor_set_pointer(value)


def _load_factor_set(raw: bytes) -> dict[str, Any]:
    try:
        value = load_canonical_resource(raw, label=FACTOR_SET_VERSION)
    except (TypeError, ValueError) as exc:
        raise _blocked("factor_set_canonical_readback") from exc
    return validate_research_shadow_factor_set(value)


def assert_research_factor_set_reread(
    workspace_root: str,
    *,
    expected_pointer_byte_sha256: str,
    expected_factor_set_ref: Mapping[str, Any],
    expected_factor_set_byte_sha256: str,
) -> ResearchFactorSetState:
    """Assert an exact current pointer and immutable set for downstream use."""

    expected_pointer = _digest(
        expected_pointer_byte_sha256,
        label="expected_pointer_byte_sha256",
    )
    expected_set_sha = _digest(
        expected_factor_set_byte_sha256,
        label="expected_factor_set_byte_sha256",
    )
    expected_ref = _artifact_ref(
        expected_factor_set_ref,
        label="expected_factor_set_ref",
    )
    if expected_ref["byte_sha256"] != expected_set_sha:
        raise _blocked("expected_factor_set_sha_mismatch")
    state = ResearchFactorSetStore(workspace_root).read_current()
    if (
        state.pointer_ref["byte_sha256"] != expected_pointer
        or state.factor_set_ref != expected_ref
        or state.factor_set_ref["byte_sha256"] != expected_set_sha
    ):
        raise _blocked("expected_reread_mismatch")
    return state


__all__ = [
    "CANDIDATE_CATALOG_SHA256",
    "CATALOG_RESOURCE_SHA256",
    "FACTOR_SET_LOCK",
    "FACTOR_SET_POINTER",
    "FACTOR_SET_ROOT",
    "FACTOR_SET_VERSION",
    "GATE_ORDER",
    "GATE_WEIGHTS",
    "IMPLEMENTATION_RESOURCE_SHA256",
    "INPUT_BUNDLE_VERSION",
    "MAX_FACTOR_COUNT",
    "POINTER_VERSION",
    "ResearchFactorSetCrash",
    "ResearchFactorSetError",
    "ResearchFactorSetPublication",
    "ResearchFactorSetState",
    "ResearchFactorSetStore",
    "SELECTION_POLICY_SHA256",
    "assert_research_factor_set_reread",
    "build_research_factor_input_bundle",
    "build_research_shadow_factor_set",
    "research_factor_catalog_bindings",
    "validate_research_factor_input_bundle",
    "validate_research_shadow_factor_set",
    "validate_research_shadow_factor_set_pointer",
]
