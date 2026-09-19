"""Code-owned dependency diagnoses; no I/O, execution or retry authority."""

from copy import deepcopy
from types import MappingProxyType

from .daily_contract import ContractError, FAILURES, failure

REASONS = MappingProxyType(
    {
        "CORE_OBSERVATION_SCHEMA_MISMATCH": "SCHEMA_MISMATCH",
        "CORE_OBSERVATION_DATE_MISMATCH": "DATE_MISMATCH",
        "CORE_OBSERVATION_FACTOR_BINDING_MISMATCH": "POINTER_MISMATCH",
        "CORE_OBSERVATION_MARKET_BINDING_MISMATCH": "POINTER_MISMATCH",
        "CORE_OBSERVATION_PIT_BINDING_MISMATCH": "POINTER_MISMATCH",
        "CORE_OBSERVATION_CALENDAR_BINDING_MISMATCH": "CALENDAR_MISMATCH",
        "CORE_OBSERVATION_SIGNAL_MISMATCH": "POINTER_MISMATCH",
        "CORE_SOURCE_SHA_MISMATCH": "SHA_MISMATCH",
        "CORE_SOURCE_MISSING": "INPUT_MISSING",
        "EXECUTION_CONTEXT_RELEASE_COMMIT_MISSING": "SCHEMA_MISMATCH",
        "EXECUTION_CONTEXT_RELEASE_COMMIT_INVALID": "SCHEMA_MISMATCH",
        "EXECUTION_CONTEXT_RELEASE_COMMIT_MISMATCH": "POINTER_MISMATCH",
        "EXECUTION_CONTEXT_CALENDAR_PARENT_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_REQUEST_DOCUMENT_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_REQUEST_WRAPPER_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_SOURCE_JSON_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_SOURCE_SCHEMA_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_COMPANY_SOURCE_SCHEMA_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_RECEIPT_FIELDS_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_SOURCE_TIME_SHAPE_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_SOURCE_TIME_ROLE_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_SOURCE_BUNDLE_SHAPE_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_NATIVE_FIELDS_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_COMPANY_FIELDS_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_AUXILIARY_REFS_INVALID": "SCHEMA_MISMATCH",
        "CUTOFF_FUNDAMENTAL_DESCRIPTOR_INVALID": "SCHEMA_MISMATCH",
        "RESEARCH_SOURCE_MISSING": "INPUT_MISSING",
        "RESEARCH_NATIVE_SOURCE_MISSING": "INPUT_MISSING",
        "CUTOFF_RETAINED_REF_MISSING": "INPUT_MISSING",
        "CUTOFF_COMMITTED_OBJECT_MISSING": "INPUT_MISSING",
        "CUTOFF_FOCUS_SOURCE_DECLARATION_REQUIRED": "INPUT_MISSING",
        "CUTOFF_MACRO_ADMISSION_MISSING": "INPUT_MISSING",
        "RESEARCH_REQUEST_SHA_MISMATCH": "SHA_MISMATCH",
        "RESEARCH_REQUEST_PAYLOAD_SHA_MISMATCH": "SHA_MISMATCH",
        "RESEARCH_SOURCE_SHA_MISMATCH": "SHA_MISMATCH",
        "RESEARCH_PREVIEW_REQUEST_SHA_MISMATCH": "SHA_MISMATCH",
        "RESEARCH_NATIVE_SOURCE_SHA_MISMATCH": "SHA_MISMATCH",
        "CUTOFF_RETAINED_REF_SHA_MISMATCH": "SHA_MISMATCH",
        "CUTOFF_COMMITTED_BYTES_SHA_MISMATCH": "SHA_MISMATCH",
        "RESEARCH_SOURCE_DATE_MISMATCH": "DATE_MISMATCH",
        "CUTOFF_RECEIPT_DAY_INVALID": "DATE_MISMATCH",
        "CUTOFF_SOURCE_BUNDLE_DAY_INVALID": "DATE_MISMATCH",
        "CUTOFF_CLOCK_TARGET_DAY_CHANGED": "DATE_MISMATCH",
        "RESEARCH_REQUEST_CUTOFF_BINDING_INVALID": "POINTER_MISMATCH",
        "RESEARCH_SOURCE_POOL_BINDING_INVALID": "POINTER_MISMATCH",
        "RESEARCH_SOURCE_REQUEST_MISMATCH": "POINTER_MISMATCH",
        "FOCUS_RECIPE_CONTEXT_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_COMMITMENT_PATH_INVALID": "POINTER_MISMATCH",
        "CUTOFF_TIMING_POLICY_MODE_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_HANDOFF_MODE_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_RETAINED_HANDOFF_PATH_INVALID": "POINTER_MISMATCH",
        "CUTOFF_CORE_PATH_INVALID": "POINTER_MISMATCH",
        "CUTOFF_PINNED_THEME_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_ACQUIRED_THEME_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_STAGE_ATTEMPT_MISMATCH": "POINTER_MISMATCH",
        "CUTOFF_AUXILIARY_SOURCE_MISMATCH": "POINTER_MISMATCH",
        "EXPOSURE_THEME_UPSTREAM_INCOMPLETE": "UPSTREAM_INCOMPLETE",
        "FOCUS_MEMBERSHIP_UPSTREAM_MISSING": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CORE_TERMINAL_INCOMPLETE": "UPSTREAM_INCOMPLETE",
        "CUTOFF_AUXILIARY_INCOMPLETE": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CRITICAL_SOURCE_INCOMPLETE:industry": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CRITICAL_SOURCE_INCOMPLETE:theme": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CRITICAL_SOURCE_INCOMPLETE:exposure": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CRITICAL_SOURCE_INCOMPLETE:fundamental": "UPSTREAM_INCOMPLETE",
        "CUTOFF_CRITICAL_SOURCE_INCOMPLETE:macro": "UPSTREAM_INCOMPLETE",
        "CUTOFF_SOURCE_BUNDLE_CONFLICT": "IDEMPOTENCY_CONFLICT",
        "CUTOFF_COMMITTED_OBJECT_CONFLICT": "IDEMPOTENCY_CONFLICT",
    }
)


class DependencyInputError(ContractError):
    """Raised only for a fixed diagnosis from an owning input validator."""

    def __init__(self, reason_code: str):
        if type(reason_code) is not str or reason_code not in REASONS:
            raise ContractError("DEPENDENCY_DIAGNOSTIC_REASON_INVALID")
        self.reason_code = reason_code
        self.failure_code = REASONS[reason_code]
        if self.failure_code not in FAILURES:
            raise ContractError("DEPENDENCY_DIAGNOSTIC_TAXONOMY_INVALID")
        super().__init__(reason_code)

    def details(self) -> dict:
        return {"failure_code": self.failure_code, "reason_code": self.reason_code}


def dependency_details(value: dict | None) -> dict | None:
    if value is None:
        return None
    if (
        type(value) is not dict
        or set(value) != {"failure_code", "reason_code"}
        or type(value["reason_code"]) is not str
        or value["reason_code"] not in REASONS
        or value["failure_code"] != REASONS[value["reason_code"]]
    ):
        raise ContractError("DEPENDENCY_DIAGNOSTIC_FIELDS_INVALID")
    return dict(value)


def rejected_probe(previous: dict, *, node: str, error: DependencyInputError) -> dict:
    """Keep durable history distinct from a new read-only probe diagnosis."""
    result = deepcopy(previous)
    result["dependency_error"] = error.details()
    state = previous["state"]
    if state == "NOT_STARTED":
        result.update(state="BLOCKED", failure=failure(error.failure_code, next_node=node))
    elif state == "RUNNING":
        result["failure"] = failure("POST_WRITE_IN_DOUBT", next_node=node)
    elif state == "SUCCEEDED":
        result.update(state="STALE", failure=failure(error.failure_code, next_node=node))
    # Non-success terminals retain their original lifecycle, failure and custody.
    return result


def upstream_blockers(missing, nodes: dict) -> list[dict]:
    """Flatten observed leaf causes without scanning or satisfying dependencies."""
    roots = set()
    for node in missing:
        row = nodes.get(node, {})
        inherited = row.get("upstream_blockers")
        if inherited:
            roots.update(
                (item["node_id"], item["failure_code"], item["reason_code"]) for item in inherited
            )
            continue
        current = dependency_details(row.get("dependency_error"))
        if current is not None:
            roots.add((node, current["failure_code"], current["reason_code"]))
        original = row.get("failure") or row.get("terminal", {}).get("failure")
        if original is not None:
            if current is None or original["code"] != current["failure_code"]:
                roots.add((node, original["code"], None))
        elif current is None:
            roots.add((node, "UPSTREAM_INCOMPLETE", None))
    return [
        {"node_id": node, "failure_code": code, "reason_code": reason}
        for node, code, reason in sorted(roots, key=lambda row: (row[0], row[1], row[2] or ""))
    ]
