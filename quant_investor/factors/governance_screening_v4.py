"""Pure FactorGovernanceProtocol v4 catalog and screening contracts.

The builders and validators in this module operate only on caller-supplied
JSON-like values.  They do not read or write artifacts, inspect a registry, or
import legacy v2/v3 governance code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

PRIMITIVE_ONTOLOGY_SCHEMA_VERSION = "factor-primitive-ontology.v4"
CANDIDATE_CATALOG_SCHEMA_VERSION = "factor-candidate-catalog.v4"
SCREENING_EVIDENCE_SCHEMA_VERSION = "factor-screening-evidence.v4"

RAW_P_METHOD = "rank_ic_normal_erfc_two_sided.v1"
FDR_METHOD = "benjamini_hochberg_by_ontology_family.v1"
FDR_Q = 0.10

EVALUATED_STATUS = "evaluated"
COMPUTE_FAILED_STATUS = "compute_failed"

SOURCE_BINDING_FIELDS = frozenset(
    {
        "code_sha256",
        "registry_file_sha256",
        "latest_pointer_sha256",
        "manifest_sha256",
        "market_data_input_sha256",
        "pit_sha256",
        "calendar_sha256",
        "fundamental_manifest_sha256",
        "run_config_sha256",
    }
)

_PRIMITIVE_FIELDS = frozenset({"primitive_id", "family"})
_ONTOLOGY_FIELDS = frozenset(
    {"schema_version", "primitives", "semantic_sha256"}
)
_CANDIDATE_INPUT_FIELDS = frozenset(
    {
        "name",
        "implementation",
        "expression",
        "direction",
        "params",
        "lookback",
        "slot",
        "input_fields",
        "primitive_ids",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {*_CANDIDATE_INPUT_FIELDS, "family", "definition_sha256"}
)
_CATALOG_FIELDS = frozenset(
    {"schema_version", "ontology_sha256", "candidates", "semantic_sha256"}
)
_STATISTIC_CONTRACT_FIELDS = frozenset(
    {"raw_p_method", "fdr_method", "q"}
)
_EVALUATION_INPUT_FIELDS = frozenset(
    {"name", "evaluation_status", "raw_p_value", "failure_reason"}
)
_SCREENING_ROW_FIELDS = frozenset(
    {
        *_EVALUATION_INPUT_FIELDS,
        "family",
        "bh_input_p_value",
        "family_hypothesis_count",
        "bh_rank",
        "bh_q_value",
        "bh_pass",
    }
)
_SCREENING_FIELDS = frozenset(
    {
        "schema_version",
        "ontology_sha256",
        "candidate_catalog_sha256",
        "source_bindings",
        "statistic_contract",
        "rows",
        "semantic_sha256",
    }
)


class FactorGovernanceScreeningV4Error(ValueError):
    """Raised when a v4 ontology, catalog, or screening artifact is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical semantic JSON encoding."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceScreeningV4Error(
            f"value is not canonical JSON: {exc}"
        ) from exc


def canonical_semantic_sha256(
    value: Any,
    *,
    exclude_fields: Sequence[str] = (),
) -> str:
    """Hash canonical JSON, optionally excluding exact top-level fields."""

    normalized = copy.deepcopy(value)
    if exclude_fields:
        if not isinstance(normalized, Mapping):
            raise FactorGovernanceScreeningV4Error(
                "exclude_fields requires a top-level object"
            )
        normalized = dict(normalized)
        seen: set[str] = set()
        for field in exclude_fields:
            if type(field) is not str or not field or field in seen:
                raise FactorGovernanceScreeningV4Error(
                    "exclude_fields must contain distinct non-empty strings"
                )
            seen.add(field)
            normalized.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceScreeningV4Error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceScreeningV4Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise FactorGovernanceScreeningV4Error(
            f"{label} fields invalid: {';'.join(details)}"
        )
    return payload


def _sequence(value: Any, label: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        raise FactorGovernanceScreeningV4Error(f"{label} must be a list")
    result = list(value)
    if nonempty and not result:
        raise FactorGovernanceScreeningV4Error(f"{label} must not be empty")
    return result


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceScreeningV4Error(
            f"{label} must be an exact non-empty string"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FactorGovernanceScreeningV4Error(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _json_value(value: Any, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise FactorGovernanceScreeningV4Error(f"{label} must be finite JSON")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise FactorGovernanceScreeningV4Error(
                    f"{label} keys must be strings"
                )
            normalized[key] = _json_value(item, f"{label}.{key}")
        return normalized
    raise FactorGovernanceScreeningV4Error(f"{label} must be exact JSON")


def _params(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FactorGovernanceScreeningV4Error(f"{label} must be an object")
    normalized = _json_value(value, label)
    canonical_json_bytes(normalized)
    return normalized


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorGovernanceScreeningV4Error(
            f"{label} must be a positive integer"
        )
    return value


def _direction(value: Any, label: str, *, canonical: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorGovernanceScreeningV4Error(f"{label} must be numeric +/-1")
    number = float(value)
    if not math.isfinite(number) or number not in (-1.0, 1.0):
        raise FactorGovernanceScreeningV4Error(f"{label} must be numeric +/-1")
    if canonical and type(value) is not float:
        raise FactorGovernanceScreeningV4Error(f"{label} must be canonical float")
    return number


def _sorted_text_list(
    value: Any,
    label: str,
    *,
    require_sorted: bool,
) -> list[str]:
    items = _sequence(value, label)
    result = [_text(item, f"{label}[]") for item in items]
    if len(result) != len(set(result)):
        raise FactorGovernanceScreeningV4Error(f"{label} must be distinct")
    ordered = sorted(result)
    if require_sorted and result != ordered:
        raise FactorGovernanceScreeningV4Error(
            f"{label} must be canonically sorted"
        )
    return ordered


def _artifact_sha(payload: Mapping[str, Any]) -> str:
    return canonical_semantic_sha256(
        payload,
        exclude_fields=("semantic_sha256",),
    )


def _normalize_primitives(
    primitives: Any,
    *,
    require_sorted: bool,
) -> list[dict[str, str]]:
    rows = _sequence(primitives, "primitives")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _exact(raw, _PRIMITIVE_FIELDS, f"primitives[{index}]")
        primitive_id = _text(row["primitive_id"], "primitive_id")
        family = _text(row["family"], "primitive family")
        if family.startswith("composite:"):
            raise FactorGovernanceScreeningV4Error(
                "primitive family uses the reserved composite: prefix"
            )
        if primitive_id in seen:
            raise FactorGovernanceScreeningV4Error(
                "primitive_id values must be distinct"
            )
        seen.add(primitive_id)
        normalized.append({"primitive_id": primitive_id, "family": family})
    ordered = sorted(normalized, key=lambda item: item["primitive_id"])
    if require_sorted and normalized != ordered:
        raise FactorGovernanceScreeningV4Error(
            "primitives must be canonically sorted by primitive_id"
        )
    return ordered


def build_primitive_ontology_v4(
    primitives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an immutable primitive-to-family ontology artifact."""

    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_ONTOLOGY_SCHEMA_VERSION,
        "primitives": _normalize_primitives(primitives, require_sorted=False),
    }
    payload["semantic_sha256"] = _artifact_sha(payload)
    return payload


def validate_primitive_ontology_v4(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact ontology schema, ordering, uniqueness, and self hash."""

    payload = _exact(artifact, _ONTOLOGY_FIELDS, "primitive ontology")
    if payload["schema_version"] != PRIMITIVE_ONTOLOGY_SCHEMA_VERSION:
        raise FactorGovernanceScreeningV4Error(
            "unsupported primitive ontology schema"
        )
    normalized = {
        "schema_version": PRIMITIVE_ONTOLOGY_SCHEMA_VERSION,
        "primitives": _normalize_primitives(
            payload["primitives"], require_sorted=True
        ),
    }
    observed_sha = _sha(payload["semantic_sha256"], "ontology semantic SHA")
    if observed_sha != _artifact_sha(normalized):
        raise FactorGovernanceScreeningV4Error(
            "primitive ontology semantic SHA mismatch"
        )
    normalized["semantic_sha256"] = observed_sha
    return normalized


def _ontology_family_map(ontology: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["primitive_id"]: row["family"]
        for row in ontology["primitives"]
    }


def _derived_family(
    primitive_ids: Sequence[str],
    *,
    ontology_families: Mapping[str, str],
) -> str:
    try:
        families = sorted({ontology_families[item] for item in primitive_ids})
    except KeyError as exc:
        raise FactorGovernanceScreeningV4Error(
            f"unknown primitive_id: {exc.args[0]}"
        ) from exc
    if len(families) == 1:
        return families[0]
    family_list = json.dumps(
        families,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"composite:{family_list}"


def _normalize_candidate_input(
    raw: Any,
    *,
    label: str,
    require_sorted_lists: bool,
    canonical_direction: bool,
) -> dict[str, Any]:
    row = _exact(raw, _CANDIDATE_INPUT_FIELDS, label)
    expression = row["expression"]
    if type(expression) is not str:
        raise FactorGovernanceScreeningV4Error(
            f"{label}.expression must be a string"
        )
    return {
        "name": _text(row["name"], f"{label}.name"),
        "implementation": _text(
            row["implementation"], f"{label}.implementation"
        ),
        "expression": expression,
        "direction": _direction(
            row["direction"],
            f"{label}.direction",
            canonical=canonical_direction,
        ),
        "params": _params(row["params"], f"{label}.params"),
        "lookback": _positive_integer(row["lookback"], f"{label}.lookback"),
        "slot": _text(row["slot"], f"{label}.slot"),
        "input_fields": _sorted_text_list(
            row["input_fields"],
            f"{label}.input_fields",
            require_sorted=require_sorted_lists,
        ),
        "primitive_ids": _sorted_text_list(
            row["primitive_ids"],
            f"{label}.primitive_ids",
            require_sorted=require_sorted_lists,
        ),
    }


def _candidate_definition_sha(candidate: Mapping[str, Any]) -> str:
    return canonical_semantic_sha256(
        candidate,
        exclude_fields=("definition_sha256",),
    )


def _build_candidate(
    definition: Mapping[str, Any],
    *,
    ontology_families: Mapping[str, str],
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(definition))
    candidate["family"] = _derived_family(
        candidate["primitive_ids"], ontology_families=ontology_families
    )
    candidate["definition_sha256"] = _candidate_definition_sha(candidate)
    return candidate


def build_candidate_catalog_v4(
    *,
    ontology: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze a unique ordered catalog with ontology-derived families."""

    normalized_ontology = validate_primitive_ontology_v4(ontology)
    ontology_families = _ontology_family_map(normalized_ontology)
    rows = _sequence(candidates, "candidates")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        definition = _normalize_candidate_input(
            raw,
            label=f"candidates[{index}]",
            require_sorted_lists=False,
            canonical_direction=False,
        )
        if definition["name"] in seen:
            raise FactorGovernanceScreeningV4Error(
                "candidate names must be distinct"
            )
        seen.add(definition["name"])
        normalized.append(
            _build_candidate(
                definition,
                ontology_families=ontology_families,
            )
        )
    normalized.sort(key=lambda item: item["name"])
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_CATALOG_SCHEMA_VERSION,
        "ontology_sha256": normalized_ontology["semantic_sha256"],
        "candidates": normalized,
    }
    payload["semantic_sha256"] = _artifact_sha(payload)
    return payload


def _validate_catalog_candidate(
    raw: Any,
    *,
    index: int,
    ontology_families: Mapping[str, str],
) -> dict[str, Any]:
    row = _exact(raw, _CANDIDATE_FIELDS, f"candidates[{index}]")
    base = _normalize_candidate_input(
        {key: row[key] for key in _CANDIDATE_INPUT_FIELDS},
        label=f"candidates[{index}]",
        require_sorted_lists=True,
        canonical_direction=True,
    )
    expected = _build_candidate(base, ontology_families=ontology_families)
    family = _text(row["family"], f"candidates[{index}].family")
    if family != expected["family"]:
        raise FactorGovernanceScreeningV4Error(
            f"candidate {base['name']} ontology-derived family mismatch"
        )
    definition_sha = _sha(
        row["definition_sha256"],
        f"candidate {base['name']} definition SHA",
    )
    if definition_sha != expected["definition_sha256"]:
        raise FactorGovernanceScreeningV4Error(
            f"candidate {base['name']} definition SHA mismatch"
        )
    return expected


def validate_candidate_catalog_v4(
    catalog: Mapping[str, Any],
    *,
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact catalog membership, definitions, families, and hashes."""

    normalized_ontology = validate_primitive_ontology_v4(ontology)
    payload = _exact(catalog, _CATALOG_FIELDS, "candidate catalog")
    if payload["schema_version"] != CANDIDATE_CATALOG_SCHEMA_VERSION:
        raise FactorGovernanceScreeningV4Error(
            "unsupported candidate catalog schema"
        )
    ontology_sha = _sha(payload["ontology_sha256"], "catalog ontology SHA")
    if ontology_sha != normalized_ontology["semantic_sha256"]:
        raise FactorGovernanceScreeningV4Error("catalog ontology SHA mismatch")
    raw_candidates = _sequence(payload["candidates"], "candidates")
    ontology_families = _ontology_family_map(normalized_ontology)
    candidates = [
        _validate_catalog_candidate(
            raw,
            index=index,
            ontology_families=ontology_families,
        )
        for index, raw in enumerate(raw_candidates)
    ]
    names = [item["name"] for item in candidates]
    if names != sorted(names) or len(names) != len(set(names)):
        raise FactorGovernanceScreeningV4Error(
            "catalog candidates must be sorted by unique name"
        )
    normalized = {
        "schema_version": CANDIDATE_CATALOG_SCHEMA_VERSION,
        "ontology_sha256": ontology_sha,
        "candidates": candidates,
    }
    observed_sha = _sha(payload["semantic_sha256"], "catalog semantic SHA")
    if observed_sha != _artifact_sha(normalized):
        raise FactorGovernanceScreeningV4Error(
            "candidate catalog semantic SHA mismatch"
        )
    normalized["semantic_sha256"] = observed_sha
    return normalized


def _validate_source_bindings(value: Any) -> dict[str, str]:
    payload = _exact(value, SOURCE_BINDING_FIELDS, "source_bindings")
    return {
        key: _sha(payload[key], f"source_bindings.{key}")
        for key in sorted(SOURCE_BINDING_FIELDS)
    }


def _validate_statistic_contract(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        _STATISTIC_CONTRACT_FIELDS,
        "statistic_contract",
    )
    if payload["raw_p_method"] != RAW_P_METHOD:
        raise FactorGovernanceScreeningV4Error(
            "statistic_contract raw_p_method mismatch"
        )
    if payload["fdr_method"] != FDR_METHOD:
        raise FactorGovernanceScreeningV4Error(
            "statistic_contract fdr_method mismatch"
        )
    if type(payload["q"]) is not float or payload["q"] != FDR_Q:
        raise FactorGovernanceScreeningV4Error(
            "statistic_contract q must be canonical 0.1"
        )
    return {
        "raw_p_method": RAW_P_METHOD,
        "fdr_method": FDR_METHOD,
        "q": FDR_Q,
    }


def _normalize_evaluation(
    raw: Any,
    *,
    label: str,
    canonical_raw_p: bool,
) -> dict[str, Any]:
    row = _exact(raw, _EVALUATION_INPUT_FIELDS, label)
    name = _text(row["name"], f"{label}.name")
    status = row["evaluation_status"]
    failure_reason = row["failure_reason"]
    raw_p_value = row["raw_p_value"]
    if status == EVALUATED_STATUS:
        if isinstance(raw_p_value, bool) or not isinstance(
            raw_p_value, (int, float)
        ):
            raise FactorGovernanceScreeningV4Error(
                f"{label}.raw_p_value must be numeric for evaluated rows"
            )
        normalized_p = float(raw_p_value)
        if not math.isfinite(normalized_p) or not 0.0 <= normalized_p <= 1.0:
            raise FactorGovernanceScreeningV4Error(
                f"{label}.raw_p_value must be finite in [0, 1]"
            )
        if canonical_raw_p and type(raw_p_value) is not float:
            raise FactorGovernanceScreeningV4Error(
                f"{label}.raw_p_value must be canonical float"
            )
        if failure_reason is not None:
            raise FactorGovernanceScreeningV4Error(
                f"{label}.failure_reason must be null when evaluated"
            )
        return {
            "name": name,
            "evaluation_status": EVALUATED_STATUS,
            "raw_p_value": normalized_p,
            "failure_reason": None,
        }
    if status == COMPUTE_FAILED_STATUS:
        if raw_p_value is not None:
            raise FactorGovernanceScreeningV4Error(
                f"{label}.raw_p_value must be null when compute_failed"
            )
        return {
            "name": name,
            "evaluation_status": COMPUTE_FAILED_STATUS,
            "raw_p_value": None,
            "failure_reason": _text(
                failure_reason,
                f"{label}.failure_reason",
            ),
        }
    raise FactorGovernanceScreeningV4Error(
        f"{label}.evaluation_status is unsupported"
    )


def _bh_rows(
    *,
    catalog: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    catalog_by_name = {
        row["name"]: row for row in catalog["candidates"]
    }
    evaluation_by_name = {row["name"]: row for row in evaluations}
    working: dict[str, dict[str, Any]] = {}
    for name, candidate in catalog_by_name.items():
        evaluation = evaluation_by_name[name]
        bh_input = (
            evaluation["raw_p_value"]
            if evaluation["evaluation_status"] == EVALUATED_STATUS
            else 1.0
        )
        working[name] = {
            **copy.deepcopy(dict(evaluation)),
            "family": candidate["family"],
            "bh_input_p_value": float(bh_input),
        }

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in working.values():
        by_family.setdefault(row["family"], []).append(row)
    for family, members in by_family.items():
        ordered = sorted(
            members,
            key=lambda item: (item["bh_input_p_value"], item["name"]),
        )
        count = len(ordered)
        adjusted = [1.0] * count
        running = 1.0
        for position in range(count - 1, -1, -1):
            rank = position + 1
            running = min(
                running,
                ordered[position]["bh_input_p_value"] * count / rank,
            )
            adjusted[position] = min(1.0, running)
        for rank, (row, q_value) in enumerate(zip(ordered, adjusted), start=1):
            if row["family"] != family:
                raise AssertionError("family grouping invariant violated")
            row.update(
                {
                    "family_hypothesis_count": count,
                    "bh_rank": rank,
                    "bh_q_value": float(q_value),
                    "bh_pass": bool(q_value <= FDR_Q),
                }
            )
    return [working[row["name"]] for row in catalog["candidates"]]


def _normalize_evaluation_set(
    evaluations: Any,
    *,
    catalog: Mapping[str, Any],
    canonical_raw_p: bool,
) -> list[dict[str, Any]]:
    rows = _sequence(evaluations, "evaluations")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _normalize_evaluation(
            raw,
            label=f"evaluations[{index}]",
            canonical_raw_p=canonical_raw_p,
        )
        if row["name"] in seen:
            raise FactorGovernanceScreeningV4Error(
                "evaluation candidate names must be distinct"
            )
        seen.add(row["name"])
        normalized.append(row)
    catalog_names = [row["name"] for row in catalog["candidates"]]
    if set(seen) != set(catalog_names) or len(normalized) != len(catalog_names):
        missing = sorted(set(catalog_names) - seen)
        extra = sorted(seen - set(catalog_names))
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise FactorGovernanceScreeningV4Error(
            "screening must contain exactly one row per catalog candidate"
            + (f": {';'.join(details)}" if details else "")
        )
    by_name = {row["name"]: row for row in normalized}
    return [by_name[name] for name in catalog_names]


def build_screening_evidence_v4(
    *,
    ontology: Mapping[str, Any],
    catalog: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
    statistic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete family-BH evidence from the frozen catalog universe."""

    normalized_ontology = validate_primitive_ontology_v4(ontology)
    normalized_catalog = validate_candidate_catalog_v4(
        catalog,
        ontology=normalized_ontology,
    )
    normalized_evaluations = _normalize_evaluation_set(
        evaluations,
        catalog=normalized_catalog,
        canonical_raw_p=False,
    )
    payload: dict[str, Any] = {
        "schema_version": SCREENING_EVIDENCE_SCHEMA_VERSION,
        "ontology_sha256": normalized_ontology["semantic_sha256"],
        "candidate_catalog_sha256": normalized_catalog["semantic_sha256"],
        "source_bindings": _validate_source_bindings(source_bindings),
        "statistic_contract": _validate_statistic_contract(statistic_contract),
        "rows": _bh_rows(
            catalog=normalized_catalog,
            evaluations=normalized_evaluations,
        ),
    }
    payload["semantic_sha256"] = _artifact_sha(payload)
    return payload


def _screening_row_evaluation(raw: Any, *, index: int) -> dict[str, Any]:
    row = _exact(raw, _SCREENING_ROW_FIELDS, f"rows[{index}]")
    evaluation = _normalize_evaluation(
        {key: row[key] for key in _EVALUATION_INPUT_FIELDS},
        label=f"rows[{index}]",
        canonical_raw_p=True,
    )
    _text(row["family"], f"rows[{index}].family")
    bh_input = row["bh_input_p_value"]
    if type(bh_input) is not float or not math.isfinite(bh_input):
        raise FactorGovernanceScreeningV4Error(
            f"rows[{index}].bh_input_p_value must be canonical finite float"
        )
    if not 0.0 <= bh_input <= 1.0:
        raise FactorGovernanceScreeningV4Error(
            f"rows[{index}].bh_input_p_value must be in [0, 1]"
        )
    _positive_integer(
        row["family_hypothesis_count"],
        f"rows[{index}].family_hypothesis_count",
    )
    _positive_integer(row["bh_rank"], f"rows[{index}].bh_rank")
    q_value = row["bh_q_value"]
    if type(q_value) is not float or not math.isfinite(q_value):
        raise FactorGovernanceScreeningV4Error(
            f"rows[{index}].bh_q_value must be canonical finite float"
        )
    if not 0.0 <= q_value <= 1.0:
        raise FactorGovernanceScreeningV4Error(
            f"rows[{index}].bh_q_value must be in [0, 1]"
        )
    if type(row["bh_pass"]) is not bool:
        raise FactorGovernanceScreeningV4Error(
            f"rows[{index}].bh_pass must be boolean"
        )
    return evaluation


def validate_screening_evidence_v4(
    evidence: Mapping[str, Any],
    *,
    ontology: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute full-catalog family BH and reject every derived-field drift."""

    normalized_ontology = validate_primitive_ontology_v4(ontology)
    normalized_catalog = validate_candidate_catalog_v4(
        catalog,
        ontology=normalized_ontology,
    )
    payload = _exact(evidence, _SCREENING_FIELDS, "screening evidence")
    if payload["schema_version"] != SCREENING_EVIDENCE_SCHEMA_VERSION:
        raise FactorGovernanceScreeningV4Error(
            "unsupported screening evidence schema"
        )
    ontology_sha = _sha(
        payload["ontology_sha256"],
        "screening ontology SHA",
    )
    if ontology_sha != normalized_ontology["semantic_sha256"]:
        raise FactorGovernanceScreeningV4Error(
            "screening ontology SHA mismatch"
        )
    catalog_sha = _sha(
        payload["candidate_catalog_sha256"],
        "screening candidate catalog SHA",
    )
    if catalog_sha != normalized_catalog["semantic_sha256"]:
        raise FactorGovernanceScreeningV4Error(
            "screening candidate catalog SHA mismatch"
        )
    source_bindings = _validate_source_bindings(payload["source_bindings"])
    statistic_contract = _validate_statistic_contract(
        payload["statistic_contract"]
    )
    raw_rows = _sequence(payload["rows"], "rows")
    evaluations: list[dict[str, Any]] = []
    names: list[str] = []
    for index, raw in enumerate(raw_rows):
        evaluation = _screening_row_evaluation(raw, index=index)
        evaluations.append(evaluation)
        names.append(evaluation["name"])
    catalog_names = [row["name"] for row in normalized_catalog["candidates"]]
    if names != catalog_names or len(names) != len(set(names)):
        raise FactorGovernanceScreeningV4Error(
            "screening rows must match catalog order and membership exactly"
        )
    expected_rows = _bh_rows(
        catalog=normalized_catalog,
        evaluations=evaluations,
    )
    if canonical_json_bytes(raw_rows) != canonical_json_bytes(expected_rows):
        raise FactorGovernanceScreeningV4Error(
            "screening BH or ontology-derived fields drifted from recomputation"
        )
    normalized = {
        "schema_version": SCREENING_EVIDENCE_SCHEMA_VERSION,
        "ontology_sha256": ontology_sha,
        "candidate_catalog_sha256": catalog_sha,
        "source_bindings": source_bindings,
        "statistic_contract": statistic_contract,
        "rows": expected_rows,
    }
    observed_sha = _sha(
        payload["semantic_sha256"],
        "screening evidence semantic SHA",
    )
    if observed_sha != _artifact_sha(normalized):
        raise FactorGovernanceScreeningV4Error(
            "screening evidence semantic SHA mismatch"
        )
    normalized["semantic_sha256"] = observed_sha
    return normalized


__all__ = [
    "CANDIDATE_CATALOG_SCHEMA_VERSION",
    "COMPUTE_FAILED_STATUS",
    "EVALUATED_STATUS",
    "FDR_METHOD",
    "FDR_Q",
    "FactorGovernanceScreeningV4Error",
    "PRIMITIVE_ONTOLOGY_SCHEMA_VERSION",
    "RAW_P_METHOD",
    "SCREENING_EVIDENCE_SCHEMA_VERSION",
    "SOURCE_BINDING_FIELDS",
    "build_candidate_catalog_v4",
    "build_primitive_ontology_v4",
    "build_screening_evidence_v4",
    "canonical_json_bytes",
    "canonical_semantic_sha256",
    "validate_candidate_catalog_v4",
    "validate_primitive_ontology_v4",
    "validate_screening_evidence_v4",
]
