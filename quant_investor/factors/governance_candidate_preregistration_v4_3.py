"""Pure definition-only FactorGovernanceProtocol v4.3 preregistration.

The evidence version is v4.3 while the governance protocol remains ``v4``.
This module accepts caller-supplied Git blob bytes and immutable descriptors,
derives exact Python-AST/YAML definition identities, and builds a prospective
zero-weight research contract.  It never imports A_quant, reads outcomes,
computes a signal, measures performance, mutates a registry, or authorizes
production activity.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_next_cycle_state_v4_1,
    byte_sha256 as cycle_state_byte_sha256_v4_1,
    validate_cycle_state_v4_1,
)


SCHEMA_VERSION = "factor-governance-candidate-preregistration.v4.3"
PROTOCOL_VERSION = "v4"
STATE_SCHEMA_VERSION = "factor-governance-cycle-state.v4.1"
SOURCE_SET_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-aquant-source-set-receipt.v4.3"
)
OPERATOR_SEMANTICS_SCHEMA_VERSION = (
    "factor-governance-definition-operator-semantics.v4.3"
)
COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-comparison-catalog-receipt.v4.3"
)
SELECTION_SPEC_SCHEMA_VERSION = "factor-governance-selection-spec.v4.3"
SOURCE_ENVELOPE_SCHEMA_VERSION = (
    "factor-governance-future-source-envelope.v4.3"
)
DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION = (
    "factor-governance-definition-identity-collision-audit.v4.3"
)
DISCOVERY_SOURCE_NODE_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-source-node.v4.3"
)
ORCHESTRATION_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-orchestration.v4.3"
)

AQUANT_GIT_TOP = "/Users/maxwell/mySpace"
AQUANT_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"

SOURCE_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "order": 1,
        "git_tree_path": (
            "A_quant/app/factors/event/guidance_revision_signal.py"
        ),
        "blob_oid": "93d7d851aa6a723d9a92a46cdc4f9a4dc90d2f84",
        "raw_sha256": (
            "533aa3379de64d48efb407d933f4191b14afe4c207145367276a9ff7b595b3d3"
        ),
        "mode": "100644",
    },
    {
        "order": 2,
        "git_tree_path": "A_quant/app/factors/event/earnings_event_drift.py",
        "blob_oid": "b12c248929f5390eee7c9610d8c03c3660836b9a",
        "raw_sha256": (
            "3be6b87c6fb69fa034bfc61fecc01e00bed1812c07f59fd957677db858945ca8"
        ),
        "mode": "100644",
    },
    {
        "order": 3,
        "git_tree_path": "A_quant/app/factors/fundamental/roe_delta.py",
        "blob_oid": "44e81d30e2777308d1f23ef4c76f5677366ab59d",
        "raw_sha256": (
            "cd5145bcf79918ffd4b460ba5f3fa2e9ac5388ec3ce51091a944ae63b052c02c"
        ),
        "mode": "100644",
    },
    {
        "order": 4,
        "git_tree_path": "A_quant/scripts/run_factor_batch_screen.py",
        "blob_oid": "6de605a9ebc6c4b1f9cd730c5ffe350d11e8aef9",
        "raw_sha256": (
            "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312"
        ),
        "mode": "100644",
    },
    {
        "order": 5,
        "git_tree_path": (
            "A_quant/app/factors/momentum/industry_relative_momentum.py"
        ),
        "blob_oid": "c3efb0f4df961b95fba1206a42bd07b916df014c",
        "raw_sha256": (
            "fe0c71c3b366ebd159011f3fcc4b14098b56a339fcc025d51bbe65a5ee6eaf64"
        ),
        "mode": "100644",
    },
    {
        "order": 6,
        "git_tree_path": "A_quant/configs/factors.yaml",
        "blob_oid": "61fcaeb2cb74d4d9529479d95ca47847a3bad061",
        "raw_sha256": (
            "6a15d0d64118a190d3cdd47dd6716e9c2dfa9d2d4e5bb4d86e998ed8a9cfd9ff"
        ),
        "mode": "100644",
    },
    {
        "order": 7,
        "git_tree_path": "A_quant/app/data/schemas.py",
        "blob_oid": "2bc56bfea1e0dd6a31a230b72422e0238312f20d",
        "raw_sha256": (
            "848f324ada44b1d6e4c944d7e156fa9901779da797c51d8076e7b56db0a55817"
        ),
        "mode": "100644",
    },
    {
        "order": 8,
        "git_tree_path": "A_quant/app/factor_sandbox/matrix_dataset.py",
        "blob_oid": "ef6f6d0a408176a0e3151d619d097c5190d60ef8",
        "raw_sha256": (
            "eab9ba96576d040622ae170fc36689a4ee62b64f13a91ae0efe9ff9cd8942547"
        ),
        "mode": "100644",
    },
)

EXPECTED_CANDIDATES = (
    "event_guidance_revision_90d",
    "event_earnings_drift_60d",
    "fund_roe_delta_annual",
    "pv_small_float_cap",
    "value_book_to_price",
    "industry_relative_momentum_20d",
)

BLOCKERS = (
    {
        "candidate": "event_guidance_revision_90d",
        "code": "guidance_missing_original_p_change_bounds",
        "detail": "guidance missing original p_change bounds",
    },
    {
        "candidate": "value_book_to_price",
        "code": "book_to_price_proxy_without_pit_equivalence",
        "detail": "book-to-price only 1/pb proxy without PIT equivalence",
    },
    {
        "candidate": "fund_roe_delta_annual",
        "code": "roe_semantic_report_type_unproved",
        "detail": "ROE semantic/report_type unproved",
    },
    {
        "candidate": "event_earnings_drift_60d",
        "code": "earnings_availability_date_equivalence_unproved",
        "detail": "earnings availability-date equivalence unproved",
    },
    {
        "candidate": "industry_relative_momentum_20d",
        "code": "industry_historical_pit_generations_insufficient",
        "detail": "industry historical PIT generations insufficient",
    },
)

AST_RUNTIME_FINGERPRINT = {
    "version_info": [3, 13, 7],
    "version": (
        "3.13.7 (main, Aug 14 2025, 11:12:11) "
        "[Clang 17.0.0 (clang-1700.0.13.3)]"
    ),
    "executable": "/Users/maxwell/mySpace/myQuant/.venv/bin/python",
    "resolved_executable": (
        "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/"
        "Python.framework/Versions/3.13/bin/python3.13"
    ),
    "executable_sha256": (
        "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
    ),
    "source_encoding": "UTF-8 Git blob",
    "ast_parse": {
        "mode": "exec",
        "type_comments": True,
        "feature_version": [3, 13],
        "optimize": -1,
    },
    "ast_dump": {
        "annotate_fields": True,
        "include_attributes": False,
        "indent": None,
        "show_empty": True,
    },
}

TIME_POLICY = {
    "timezone": "Asia/Shanghai",
    "selection_independence": "UNPROVEN",
    "publication_time_authority": "LOCAL_UNVERIFIED",
    "measurement_authorized": False,
    "measurement_anchor_status": (
        "PENDING_INDEPENDENT_POST_PUBLICATION_EVIDENCE"
    ),
    "publication_date_in_measurement_sample": False,
    "first_eligible_session": "STRICTLY_LATER_THAN_PUBLICATION_DATE",
    "embargo_open_sessions": 30,
    "measurement_sample_begins_at_eligible_session": 31,
    "minimum_post_embargo_open_sessions_policy": 240,
    "minimum_distinct_month_ends_policy": 12,
    "horizon_policy_only": True,
}

MEASUREMENT_FLAGS = {
    "runtime_equivalence": "not_run",
    "signal_computability": "not_run",
    "measurement": "not_run",
    "statistics": "not_run",
    "family_bh": "not_run",
    "maturity": "not_run",
    "walk_forward": "not_run",
    "cost": "not_run",
    "neutralization": "not_run",
    "stability": "not_run",
    "structural_dedup": "not_run",
    "formal_dedup": "not_run",
    "high_correlation_dedup": "not_run",
    "verified_v4_replay": "not_run",
    "transaction_plan": "not_run",
    "readiness": "PROSPECTIVE_PREREGISTRATION_ONLY",
    "status": "measurement_not_run",
}

AUTHORITY_FLAGS = {
    "measurement_authorized": False,
    "screening_authorized": False,
    "family_bh_authorized": False,
    "maturity_authorized": False,
    "walk_forward_authorized": False,
    "dedup_authorized": False,
    "replay_authorized": False,
    "candidate_qualified": False,
    "qualification_authorized": False,
    "admission_authorized": False,
    "production_new_risk_authorized": False,
    "production_candidate_authorized": False,
    "registry_write_authorized": False,
    "production_proposal_authorized": False,
    "apply_authorized": False,
}

SIDE_EFFECT_FLAGS = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "provider": False,
    "network": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "transaction": False,
}

OPERATOR_SEMANTICS = {
    "contract": "definition_only_ast_yaml_canonicalization",
    "python_ast_canonicalization": copy.deepcopy(AST_RUNTIME_FINGERPRINT),
    "yaml_selection_canonicalization": "sorted-key canonical JSON",
    "domain_separation": "explicit domain and payload canonical JSON",
    "runtime_equivalence_verified": False,
    "signal_computability_proven": False,
    "measurement_status": "measurement_not_run",
}

EXPECTED_COMPARISON_SOURCE_NAMES = ("base230", "v4_1", "v4_2")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_YAML_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*):(?:\s*(.*))?")


class FactorGovernanceCandidatePreregistrationV4_3Error(ValueError):
    """Raised when v4.3 preregistration evidence fails closed."""


FactorGovernanceCandidatePreregistrationV43Error = (
    FactorGovernanceCandidatePreregistrationV4_3Error
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted finite UTF-8 JSON bytes without a newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceCandidatePreregistrationV4_3Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes(value: Any) -> bytes:
    """Return canonical artifact bytes with exactly one final newline."""

    return canonical_json_bytes(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    """Hash canonical semantic JSON bytes without a final newline."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def byte_sha256(value: Any) -> str:
    """Hash canonical artifact bytes including their final newline."""

    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def domain_separated_sha256_v4_3(domain: str, payload: Any) -> str:
    """Hash canonical JSON under an explicit non-empty domain."""

    if type(domain) is not str or not domain or domain != domain.strip():
        raise FactorGovernanceCandidatePreregistrationV4_3Error(
            "domain must be an exact non-empty string"
        )
    return semantic_sha256({"domain": domain, "payload": payload})


canonical_json_bytes_v4_3 = canonical_json_bytes
canonical_file_bytes_v4_3 = canonical_file_bytes
semantic_sha256_v4_3 = semantic_sha256
byte_sha256_v4_3 = byte_sha256


def _error(message: str) -> FactorGovernanceCandidatePreregistrationV4_3Error:
    return FactorGovernanceCandidatePreregistrationV4_3Error(message)


def _self_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "artifact_semantic_sha256"
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(payload))
    sealed["artifact_semantic_sha256"] = semantic_sha256(_self_payload(sealed))
    return sealed


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise _error(f"{label} fields invalid: {';'.join(details)}")
    canonical_json_bytes(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase git OID")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _exact_bool(value: Any, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise _error(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise _error(f"{label} must be {expected}")
    return value


def _date(value: Any, label: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise _error(f"{label} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be a real ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be a canonical ISO calendar date")
    return value


def _preregistered_at(value: Any) -> tuple[str, str]:
    if type(value) is not str:
        raise _error("preregistered_at must be an ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _error("preregistered_at must be a real ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.microsecond != 0:
        raise _error("preregistered_at must be timezone-aware to exact seconds")
    shanghai = ZoneInfo("Asia/Shanghai")
    normalized = parsed.astimezone(shanghai)
    if normalized.utcoffset() != timedelta(hours=8):
        raise _error("preregistered_at must resolve to Asia/Shanghai")
    canonical = normalized.isoformat(timespec="seconds")
    if value != canonical:
        raise _error("preregistered_at must be canonical Asia/Shanghai ISO time")
    return canonical, normalized.date().isoformat()


def _snapshot_id(value: Any, *, snapshot_date: str) -> str:
    if type(value) is not str or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise _error("snapshot_id must be exact YYYYMMDDTHHMMSSZ")
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("snapshot_id must be a real UTC timestamp") from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != value:
        raise _error("snapshot_id must be a canonical UTC timestamp")
    if parsed.date().isoformat() != snapshot_date:
        raise _error("snapshot_id date must equal snapshot_date")
    return value


def _artifact_semantic(payload: Mapping[str, Any], label: str) -> str:
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"),
        f"{label}.artifact_semantic_sha256",
    )
    if supplied != semantic_sha256(_self_payload(payload)):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return supplied


def _exact_flags(payload: Mapping[str, Any]) -> None:
    if payload.get("measurement") != MEASUREMENT_FLAGS:
        raise _error("measurement flags must be exact not_run values")
    if payload.get("authority") != AUTHORITY_FLAGS:
        raise _error("authority flags must be exact false values")
    if payload.get("side_effects") != SIDE_EFFECT_FLAGS:
        raise _error("side_effect flags must be exact false values")


def runtime_fingerprint_v4_3() -> dict[str, Any]:
    """Read and validate the exact live Python 3.13.7 AST runtime."""

    resolved = Path(sys.executable).resolve(strict=True)
    live = {
        "version_info": list(sys.version_info[:3]),
        "version": sys.version,
        "executable": sys.executable,
        "resolved_executable": str(resolved),
        "executable_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "source_encoding": "UTF-8 Git blob",
        "ast_parse": copy.deepcopy(AST_RUNTIME_FINGERPRINT["ast_parse"]),
        "ast_dump": copy.deepcopy(AST_RUNTIME_FINGERPRINT["ast_dump"]),
    }
    return validate_runtime_fingerprint_v4_3(live)


def validate_runtime_fingerprint_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact AST runtime without consulting ambient state."""

    fields = frozenset(AST_RUNTIME_FINGERPRINT)
    payload = _exact(value, fields, "AST runtime fingerprint")
    if payload != AST_RUNTIME_FINGERPRINT:
        raise _error("AST runtime fingerprint mismatch")
    return copy.deepcopy(payload)


def _decode_source(source_bytes: Any, git_tree_path: str) -> str:
    if type(source_bytes) is not bytes:
        raise _error(f"{git_tree_path} source must be exact bytes")
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error(f"{git_tree_path} must be a UTF-8 Git blob") from exc
    if source.encode("utf-8") != source_bytes:
        raise _error(f"{git_tree_path} UTF-8 round-trip mismatch")
    return source


def _parse_ast(source_bytes: bytes, git_tree_path: str) -> ast.Module:
    source = _decode_source(source_bytes, git_tree_path)
    try:
        return ast.parse(
            source,
            filename=git_tree_path,
            mode="exec",
            type_comments=True,
            feature_version=(3, 13),
            optimize=-1,
        )
    except (SyntaxError, TypeError, ValueError) as exc:
        raise _error(f"{git_tree_path} failed exact Python AST parse") from exc


def _ast_dump(node: ast.AST) -> str:
    return ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
        indent=None,
        show_empty=True,
    )


def _one(nodes: Sequence[ast.AST], selector_id: str) -> ast.AST:
    if len(nodes) != 1:
        raise _error(
            f"selector {selector_id} must match exactly once, got {len(nodes)}"
        )
    return nodes[0]


def _module_assignment(module: ast.Module, name: str, selector_id: str) -> ast.AST:
    matches: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                matches.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                matches.append(node)
    return _one(matches, selector_id)


def _module_class(module: ast.Module, name: str, selector_id: str) -> ast.ClassDef:
    node = _one(
        [item for item in module.body if isinstance(item, ast.ClassDef) and item.name == name],
        selector_id,
    )
    assert isinstance(node, ast.ClassDef)
    return node


def _module_function(
    module: ast.Module, name: str, selector_id: str
) -> ast.FunctionDef:
    node = _one(
        [item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == name],
        selector_id,
    )
    assert isinstance(node, ast.FunctionDef)
    return node


def _class_method(
    module: ast.Module, class_name: str, method_name: str, selector_id: str
) -> ast.FunctionDef:
    class_node = _module_class(module, class_name, selector_id + ".class")
    node = _one(
        [
            item
            for item in class_node.body
            if isinstance(item, ast.FunctionDef) and item.name == method_name
        ],
        selector_id,
    )
    assert isinstance(node, ast.FunctionDef)
    return node


def _candidate_add_call(
    module: ast.Module, candidate_name: str, selector_id: str
) -> ast.Call:
    function = _module_function(
        module, "generate_default_candidates", selector_id + ".function"
    )
    matches: list[ast.AST] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "add":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            if node.args[0].value == candidate_name:
                matches.append(node)
    selected = _one(matches, selector_id)
    assert isinstance(selected, ast.Call)
    return selected


def _candidate_tuple(
    module: ast.Module, candidate_name: str, selector_id: str
) -> ast.Tuple:
    function = _module_function(
        module, "generate_default_candidates", selector_id + ".function"
    )
    containers: list[ast.AST] = []
    for node in function.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "fundamentals"
                for target in node.targets
            ):
                containers.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "fundamentals":
                if node.value is not None:
                    containers.append(node.value)
    container = _one(containers, selector_id + ".fundamentals")
    if not isinstance(container, (ast.List, ast.Tuple)):
        raise _error("fundamentals must be an AST list or tuple")
    matches: list[ast.AST] = []
    for item in container.elts:
        if not isinstance(item, ast.Tuple) or not item.elts:
            continue
        first = item.elts[0]
        if isinstance(first, ast.Constant) and first.value == candidate_name:
            matches.append(item)
    selected = _one(matches, selector_id)
    assert isinstance(selected, ast.Tuple)
    return selected


def _strip_yaml_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    result: list[str] = []
    for character in value:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and quote == '"':
            result.append(character)
            escaped = True
            continue
        if character in {'"', "'"}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            result.append(character)
            continue
        if character == "#" and quote is None:
            break
        result.append(character)
    if quote is not None:
        raise _error("unterminated quoted YAML scalar")
    return "".join(result).strip()


def _yaml_scalar(value: str, label: str) -> Any:
    stripped = _strip_yaml_comment(value)
    if not stripped:
        raise _error(f"{label} must be a scalar")
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", stripped):
        return int(stripped)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]+", stripped):
        return float(stripped)
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", stripped):
        return stripped
    raise _error(f"{label} uses an unsupported YAML scalar")


def _yaml_factor_selection(source_bytes: bytes, factor_name: str) -> dict[str, Any]:
    source = _decode_source(source_bytes, "A_quant/configs/factors.yaml")
    lines = source.splitlines()
    factors_headers: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent]:
            raise _error("YAML indentation must use spaces")
        match = _YAML_KEY_RE.fullmatch(line.strip())
        if match and indent == 0 and match.group(1) == "factors":
            if _strip_yaml_comment(match.group(2) or ""):
                raise _error("factors YAML node must be a mapping")
            factors_headers.append((index, indent))
    if len(factors_headers) != 1:
        raise _error(
            f"YAML selector factors must match exactly once, got {len(factors_headers)}"
        )
    factors_index, factors_indent = factors_headers[0]
    factor_headers: list[tuple[int, int]] = []
    for index in range(factors_index + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= factors_indent:
            break
        match = _YAML_KEY_RE.fullmatch(line.strip())
        if match and indent == factors_indent + 2 and match.group(1) == factor_name:
            if _strip_yaml_comment(match.group(2) or ""):
                raise _error(f"factors.{factor_name} must be a mapping")
            factor_headers.append((index, indent))
    if len(factor_headers) != 1:
        raise _error(
            f"YAML selector factors.{factor_name} must match exactly once, "
            f"got {len(factor_headers)}"
        )
    factor_index, factor_indent = factor_headers[0]
    result: dict[str, Any] = {}
    for index in range(factor_index + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= factor_indent:
            break
        if indent != factor_indent + 2:
            raise _error(f"factors.{factor_name} must contain direct scalar keys")
        match = _YAML_KEY_RE.fullmatch(line.strip())
        if match is None or match.group(2) is None:
            raise _error(f"factors.{factor_name} contains a non-scalar entry")
        key = match.group(1)
        if key in result:
            raise _error(f"factors.{factor_name}.{key} must occur exactly once")
        result[key] = _yaml_scalar(
            match.group(2), f"factors.{factor_name}.{key}"
        )
    if not result:
        raise _error(f"factors.{factor_name} must not be empty")
    canonical_json_bytes(result)
    return result


_SELECTOR_PLAN = (
    (
        "guidance_default_lookback",
        SOURCE_BINDINGS[0]["git_tree_path"],
        "assignment:_DEFAULT_LOOKBACK_DAYS",
    ),
    ("guidance_class", SOURCE_BINDINGS[0]["git_tree_path"], "class:GuidanceRevisionSignal"),
    ("earnings_class", SOURCE_BINDINGS[1]["git_tree_path"], "class:EarningsEventDrift"),
    ("roe_class", SOURCE_BINDINGS[2]["git_tree_path"], "class:RoeDelta"),
    (
        "float_candidate_call",
        SOURCE_BINDINGS[3]["git_tree_path"],
        "call:generate_default_candidates.add:alpha_float_size_small",
    ),
    (
        "book_candidate_tuple",
        SOURCE_BINDINGS[3]["git_tree_path"],
        "tuple:generate_default_candidates.fundamentals:alpha_value_book_to_price",
    ),
    ("industry_class", SOURCE_BINDINGS[4]["git_tree_path"], "class:IndustryRelativeMomentum"),
    ("config_earnings", SOURCE_BINDINGS[5]["git_tree_path"], "yaml:factors.earnings_event_drift"),
    ("config_roe", SOURCE_BINDINGS[5]["git_tree_path"], "yaml:factors.roe_delta"),
    (
        "config_industry",
        SOURCE_BINDINGS[5]["git_tree_path"],
        "yaml:factors.industry_relative_momentum",
    ),
    ("forecast_type_score", SOURCE_BINDINGS[6]["git_tree_path"], "assignment:FORECAST_TYPE_SCORE"),
    ("performance_forecast", SOURCE_BINDINGS[6]["git_tree_path"], "class:PerformanceForecast"),
    ("report_type", SOURCE_BINDINGS[6]["git_tree_path"], "class:ReportType"),
    ("financial_snapshot", SOURCE_BINDINGS[6]["git_tree_path"], "class:FinancialSnapshot"),
    ("matrix_load", SOURCE_BINDINGS[7]["git_tree_path"], "method:FactorMatrixDataset.load"),
    (
        "matrix_valuation",
        SOURCE_BINDINGS[7]["git_tree_path"],
        "method:FactorMatrixDataset._build_valuation_matrices",
    ),
)


def _ast_node_for_selector(
    module: ast.Module, selector_id: str
) -> ast.AST:
    if selector_id == "guidance_default_lookback":
        return _module_assignment(module, "_DEFAULT_LOOKBACK_DAYS", selector_id)
    if selector_id == "guidance_class":
        return _module_class(module, "GuidanceRevisionSignal", selector_id)
    if selector_id == "earnings_class":
        return _module_class(module, "EarningsEventDrift", selector_id)
    if selector_id == "roe_class":
        return _module_class(module, "RoeDelta", selector_id)
    if selector_id == "float_candidate_call":
        return _candidate_add_call(module, "alpha_float_size_small", selector_id)
    if selector_id == "book_candidate_tuple":
        return _candidate_tuple(module, "alpha_value_book_to_price", selector_id)
    if selector_id == "industry_class":
        return _module_class(module, "IndustryRelativeMomentum", selector_id)
    if selector_id == "forecast_type_score":
        return _module_assignment(module, "FORECAST_TYPE_SCORE", selector_id)
    if selector_id == "performance_forecast":
        return _module_class(module, "PerformanceForecast", selector_id)
    if selector_id == "report_type":
        return _module_class(module, "ReportType", selector_id)
    if selector_id == "financial_snapshot":
        return _module_class(module, "FinancialSnapshot", selector_id)
    if selector_id == "matrix_load":
        return _class_method(module, "FactorMatrixDataset", "load", selector_id)
    if selector_id == "matrix_valuation":
        return _class_method(
            module,
            "FactorMatrixDataset",
            "_build_valuation_matrices",
            selector_id,
        )
    raise _error(f"unknown AST selector {selector_id}")


def _validate_selector_intent(
    selector_id: str, selected: ast.AST | Mapping[str, Any]
) -> None:
    if selector_id == "guidance_default_lookback":
        value = selected.value if isinstance(selected, (ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Constant) or value.value != 90:
            raise _error("_DEFAULT_LOOKBACK_DAYS must be exact integer 90")
    elif selector_id == "earnings_class":
        assert isinstance(selected, ast.AST)
        age_defaults = [
            node.args[2].value
            for node in ast.walk(selected)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[2], ast.Constant)
        ]
        if age_defaults.count(60) != 1:
            raise _error("EarningsEventDrift must bind exactly one default 60")
    elif selector_id == "roe_class":
        assert isinstance(selected, ast.AST)
        annual = [
            node
            for node in ast.walk(selected)
            if isinstance(node, ast.Attribute)
            and node.attr == "ANNUAL"
            and isinstance(node.value, ast.Name)
            and node.value.id == "ReportType"
        ]
        _one(annual, "RoeDelta.ReportType.ANNUAL")
    elif selector_id == "float_candidate_call":
        assert isinstance(selected, ast.Call)
        expected = [
            "alpha_float_size_small",
            "cs_rank(-float_market_cap)",
        ]
        actual = [
            item.value if isinstance(item, ast.Constant) else None
            for item in selected.args[:2]
        ]
        if actual != expected:
            raise _error("alpha_float_size_small alias/expression mismatch")
    elif selector_id == "book_candidate_tuple":
        assert isinstance(selected, ast.Tuple)
        expected = ["alpha_value_book_to_price", "cs_rank(book_to_price)"]
        actual = [
            item.value if isinstance(item, ast.Constant) else None
            for item in selected.elts[:2]
        ]
        if actual != expected:
            raise _error("alpha_value_book_to_price alias/expression mismatch")
    elif selector_id == "industry_class":
        assert isinstance(selected, ast.AST)
        defaults = [
            node.args[2].value
            for node in ast.walk(selected)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[2], ast.Constant)
        ]
        if defaults.count(20) != 1:
            raise _error("IndustryRelativeMomentum must bind one default 20")
    elif selector_id == "config_earnings":
        if selected != {"enabled": True, "max_event_age_days": 60}:
            raise _error("earnings_event_drift YAML config mismatch")
    elif selector_id == "config_roe":
        if selected != {"enabled": True}:
            raise _error("roe_delta YAML config mismatch")
    elif selector_id == "config_industry":
        if selected != {"enabled": True, "lookback_days": 20}:
            raise _error("industry_relative_momentum YAML config mismatch")


def _selector_bindings_from_blobs(
    blobs: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    modules: dict[str, ast.Module] = {}
    bindings: list[dict[str, Any]] = []
    yaml_path = SOURCE_BINDINGS[5]["git_tree_path"]
    for selector_id, git_tree_path, selector in _SELECTOR_PLAN:
        if selector_id.startswith("config_"):
            factor_name = {
                "config_earnings": "earnings_event_drift",
                "config_roe": "roe_delta",
                "config_industry": "industry_relative_momentum",
            }[selector_id]
            selected = _yaml_factor_selection(blobs[yaml_path], factor_name)
            _validate_selector_intent(selector_id, selected)
            identity = domain_separated_sha256_v4_3(
                "factor-governance-yaml-selector.v4.3",
                {
                    "git_tree_path": git_tree_path,
                    "selector": selector,
                    "value": selected,
                },
            )
            canonicalization = "sorted-key canonical JSON"
        else:
            module = modules.setdefault(
                git_tree_path, _parse_ast(blobs[git_tree_path], git_tree_path)
            )
            selected_node = _ast_node_for_selector(module, selector_id)
            _validate_selector_intent(selector_id, selected_node)
            identity = domain_separated_sha256_v4_3(
                "factor-governance-python-ast-selector.v4.3",
                {
                    "git_tree_path": git_tree_path,
                    "selector": selector,
                    "ast_dump": _ast_dump(selected_node),
                },
            )
            canonicalization = "ast.dump.v3_13_7"
        bindings.append(
            {
                "selector_id": selector_id,
                "git_tree_path": git_tree_path,
                "selector": selector,
                "canonicalization": canonicalization,
                "canonical_sha256": identity,
            }
        )
    return bindings


_CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "order": 1,
        "name": "event_guidance_revision_90d",
        "source_alias": (
            "GuidanceRevisionSignal + _DEFAULT_LOOKBACK_DAYS=90 + "
            "FORECAST_TYPE_SCORE/PerformanceForecast"
        ),
        "definition_intent": "company guidance revision proxy over 90 calendar days",
        "family": "event_guidance",
        "selector_ids": (
            "guidance_default_lookback",
            "guidance_class",
            "forecast_type_score",
            "performance_forecast",
        ),
    },
    {
        "order": 2,
        "name": "event_earnings_drift_60d",
        "source_alias": "EarningsEventDrift config 60",
        "definition_intent": "earnings availability-date drift over 60 calendar days",
        "family": "event_earnings_drift",
        "selector_ids": (
            "earnings_class",
            "config_earnings",
            "financial_snapshot",
        ),
    },
    {
        "order": 3,
        "name": "fund_roe_delta_annual",
        "source_alias": "RoeDelta + ReportType.ANNUAL",
        "definition_intent": "latest annual ROE minus prior annual ROE",
        "family": "fundamental_profitability",
        "selector_ids": (
            "roe_class",
            "config_roe",
            "report_type",
            "financial_snapshot",
        ),
    },
    {
        "order": 4,
        "name": "pv_small_float_cap",
        "source_alias": (
            "unique generate_default_candidates add alpha_float_size_small "
            "cs_rank(-float_market_cap)"
        ),
        "definition_intent": "cross-sectional small float market capitalization",
        "family": "size",
        "selector_ids": (
            "float_candidate_call",
            "matrix_load",
            "matrix_valuation",
        ),
    },
    {
        "order": 5,
        "name": "value_book_to_price",
        "source_alias": (
            "unique fundamentals tuple alpha_value_book_to_price "
            "cs_rank(book_to_price)"
        ),
        "definition_intent": "cross-sectional book-to-price value proxy",
        "family": "valuation",
        "selector_ids": (
            "book_candidate_tuple",
            "matrix_load",
            "matrix_valuation",
        ),
    },
    {
        "order": 6,
        "name": "industry_relative_momentum_20d",
        "source_alias": "IndustryRelativeMomentum config 20 calendar days",
        "definition_intent": "stock return minus industry return over 20 calendar days",
        "family": "industry_momentum",
        "selector_ids": ("industry_class", "config_industry"),
    },
)


def _candidate_definition_rows(
    selector_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["selector_id"]: dict(row) for row in selector_bindings}
    if len(by_id) != len(_SELECTOR_PLAN):
        raise _error("selector bindings must contain every selector exactly once")
    rows: list[dict[str, Any]] = []
    for spec in _CANDIDATE_SPECS:
        selector_rows = [
            {
                "selector_id": selector_id,
                "canonical_sha256": by_id[selector_id]["canonical_sha256"],
            }
            for selector_id in spec["selector_ids"]
        ]
        identity = domain_separated_sha256_v4_3(
            "factor-governance-candidate-definition-identity.v4.3",
            {
                "name": spec["name"],
                "source_alias": spec["source_alias"],
                "definition_intent": spec["definition_intent"],
                "family": spec["family"],
                "selectors": selector_rows,
            },
        )
        rows.append(
            {
                "order": spec["order"],
                "name": spec["name"],
                "source_alias": spec["source_alias"],
                "definition_intent": spec["definition_intent"],
                "family": spec["family"],
                "selector_bindings": selector_rows,
                "definition_identity_sha256": identity,
            }
        )
    return rows


# Derived once from the exact Git objects with the canonicalizers above.
# Keeping the identities immutable prevents a re-sealed receipt from swapping
# a selector result after the builder has discarded the source bytes.
_EXPECTED_SELECTOR_SHA256 = {
    "guidance_default_lookback": (
        "cd73a1e36d4022e2b10a7169917116e6727d2b8b759bca935d447321eb513bc8"
    ),
    "guidance_class": (
        "e0333fbdd7610f8ce7398017d6cf2bbb207d8dcf14979b52f9d7f6dc839b8637"
    ),
    "earnings_class": (
        "caaa216c3eb7c77136a20955e4a09778d47b4d06f3f2edbe297afec71161ca6a"
    ),
    "roe_class": (
        "af6e42aecc500c068cbe8b7e230fa9cd16fe4b93124d61668f2c833be8fa115d"
    ),
    "float_candidate_call": (
        "8401dbec8a596ae263cb01d2b4748e9c94540d4c2412233957bf9d05e6ace241"
    ),
    "book_candidate_tuple": (
        "8035eb2eb8d0021732ae16931d15533a82fd33dd9d2cd0e96c1a7b77cceb8d59"
    ),
    "industry_class": (
        "a766f9a9c8cee290cfa7385d49bcf0df2fab0137fc97332f110913d622dc5914"
    ),
    "config_earnings": (
        "b2342870e8411fa9539d9764adb57439320598018ed63c5938f8f08256f75e17"
    ),
    "config_roe": (
        "3776da1cd4f5767ab50f21f5d4d5ce03be14755bc822fcb0513393fa59129eb0"
    ),
    "config_industry": (
        "50b6eb3713e1c84116f9ec6ca8292ec79a463a23915c18ba9db30681c21b84d9"
    ),
    "forecast_type_score": (
        "5423073d7acfb6e99c4802c011c45bc5a46f3dd9f48fe121c34cffea0f6dd025"
    ),
    "performance_forecast": (
        "30b217f020eb6b5b9cfa0fd7fac0b3740f9bd1190a039f92fcf69de0d2f411a7"
    ),
    "report_type": (
        "4bff44b2c0fe4363eb231bdbfdc26d2553fcb7b2d83b8cbaf2bb9da153a5d0b1"
    ),
    "financial_snapshot": (
        "6fda2440ed957e17ee8eec78c580c7fd0ab6aa91c9ad870a0687d645c766540c"
    ),
    "matrix_load": (
        "d94205787560674764a2a41a8ce0657302e4872219ac078aec43bc3b48bac854"
    ),
    "matrix_valuation": (
        "4a710a29ad44ab9d92a614a6540fad937db65e80066caacf50c987723f741bdc"
    ),
}
EXPECTED_SELECTOR_BINDINGS = tuple(
    {
        "selector_id": selector_id,
        "git_tree_path": git_tree_path,
        "selector": selector,
        "canonicalization": (
            "sorted-key canonical JSON"
            if selector_id.startswith("config_")
            else "ast.dump.v3_13_7"
        ),
        "canonical_sha256": _EXPECTED_SELECTOR_SHA256[selector_id],
    }
    for selector_id, git_tree_path, selector in _SELECTOR_PLAN
)
EXPECTED_CANDIDATE_DEFINITION_ROWS = tuple(
    _candidate_definition_rows(EXPECTED_SELECTOR_BINDINGS)
)


def _normalize_source_blobs(value: Any) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise _error("aquant_git_objects must be a path-to-bytes mapping")
    expected_by_path = {
        row["git_tree_path"]: row for row in SOURCE_BINDINGS
    }
    if set(value) != set(expected_by_path):
        missing = sorted(set(expected_by_path) - set(value))
        extra = sorted(set(value) - set(expected_by_path))
        raise _error(
            "aquant_git_objects paths mismatch: "
            f"missing={','.join(missing)};extra={','.join(extra)}"
        )
    result: dict[str, bytes] = {}
    for path, item in value.items():
        expected = expected_by_path[path]
        if type(item) is bytes:
            content = item
        elif isinstance(item, Mapping):
            row = dict(item)
            allowed = frozenset(
                {
                    "content",
                    "blob_bytes",
                    "git_tree_path",
                    "blob_oid",
                    "raw_sha256",
                    "mode",
                }
            )
            unknown = set(row) - allowed
            if unknown:
                raise _error(
                    f"aquant_git_objects[{path}] unknown fields: "
                    + ",".join(sorted(unknown))
                )
            content_fields = [
                key for key in ("content", "blob_bytes") if key in row
            ]
            if len(content_fields) != 1 or type(row[content_fields[0]]) is not bytes:
                raise _error(f"aquant_git_objects[{path}] needs one bytes content field")
            content = row[content_fields[0]]
            for field in ("git_tree_path", "blob_oid", "raw_sha256", "mode"):
                if field in row and row[field] != expected[field]:
                    raise _error(f"aquant_git_objects[{path}].{field} mismatch")
        else:
            raise _error(f"aquant_git_objects[{path}] must contain bytes")
        raw_sha = hashlib.sha256(content).hexdigest()
        if raw_sha != expected["raw_sha256"]:
            raise _error(f"{path} raw SHA-256 mismatch")
        _decode_source(content, path)
        result[path] = content
    return result


def validate_aquant_source_set_receipt_v4_3(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact eight-source A_quant definition receipt."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "project",
            "git_top",
            "commit",
            "runtime_fingerprint",
            "source_bindings",
            "selector_bindings",
            "candidate_definitions",
            "source_authenticity_verified",
            "definition_identity_verified",
            "runtime_equivalence_verified",
            "signal_computability_proven",
            "measurement_status",
            "outcome_paths_read",
            "outcomes_used_as_evidence",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "A_quant source-set receipt")
    if payload["schema_version"] != SOURCE_SET_RECEIPT_SCHEMA_VERSION:
        raise _error("A_quant source-set receipt schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["project"] != "A_quant":
        raise _error("A_quant project mismatch")
    if payload["git_top"] != AQUANT_GIT_TOP:
        raise _error("A_quant Git top mismatch")
    if _oid(payload["commit"], "A_quant commit") != AQUANT_COMMIT:
        raise _error("A_quant commit mismatch")
    validate_runtime_fingerprint_v4_3(payload["runtime_fingerprint"])
    if payload["source_bindings"] != list(SOURCE_BINDINGS):
        raise _error("A_quant source bindings/order mismatch")
    if payload["selector_bindings"] != list(EXPECTED_SELECTOR_BINDINGS):
        raise _error("A_quant selector bindings/order mismatch")
    if payload["candidate_definitions"] != list(
        EXPECTED_CANDIDATE_DEFINITION_ROWS
    ):
        raise _error("A_quant candidate definition identities/order mismatch")
    if tuple(
        row["name"] for row in payload["candidate_definitions"]
    ) != EXPECTED_CANDIDATES:
        raise _error("candidate definition allowlist mismatch")
    _exact_bool(
        payload["source_authenticity_verified"],
        "source_authenticity_verified",
        True,
    )
    _exact_bool(
        payload["definition_identity_verified"],
        "definition_identity_verified",
        True,
    )
    _exact_bool(
        payload["runtime_equivalence_verified"],
        "runtime_equivalence_verified",
        False,
    )
    _exact_bool(
        payload["signal_computability_proven"],
        "signal_computability_proven",
        False,
    )
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    if payload["outcome_paths_read"] != []:
        raise _error("outcome_paths_read must be exact empty list")
    _exact_bool(
        payload["outcomes_used_as_evidence"],
        "outcomes_used_as_evidence",
        False,
    )
    _artifact_semantic(payload, "A_quant source-set receipt")
    return copy.deepcopy(payload)


def build_aquant_source_set_receipt_v4_3(
    *,
    aquant_git_objects: Mapping[str, Any] | None = None,
    source_blobs: Mapping[str, Any] | None = None,
    runtime_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source receipt from all eight exact caller-supplied Git blobs."""

    if (aquant_git_objects is None) == (source_blobs is None):
        raise _error("supply exactly one of aquant_git_objects or source_blobs")
    # AST identities are invalid under any ambient runtime other than the
    # explicitly pinned interpreter, even if a caller supplies the right label.
    runtime_fingerprint_v4_3()
    blobs = _normalize_source_blobs(
        aquant_git_objects if aquant_git_objects is not None else source_blobs
    )
    runtime = validate_runtime_fingerprint_v4_3(
        runtime_fingerprint
        if runtime_fingerprint is not None
        else runtime_fingerprint_v4_3()
    )
    selector_bindings = _selector_bindings_from_blobs(blobs)
    candidate_definitions = _candidate_definition_rows(selector_bindings)
    if selector_bindings != list(EXPECTED_SELECTOR_BINDINGS):
        raise _error("derived selector bindings differ from pinned identities")
    if candidate_definitions != list(EXPECTED_CANDIDATE_DEFINITION_ROWS):
        raise _error("derived candidate identities differ from pinned identities")
    return validate_aquant_source_set_receipt_v4_3(
        _seal(
            {
                "schema_version": SOURCE_SET_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "project": "A_quant",
                "git_top": AQUANT_GIT_TOP,
                "commit": AQUANT_COMMIT,
                "runtime_fingerprint": runtime,
                "source_bindings": list(copy.deepcopy(SOURCE_BINDINGS)),
                "selector_bindings": selector_bindings,
                "candidate_definitions": candidate_definitions,
                "source_authenticity_verified": True,
                "definition_identity_verified": True,
                "runtime_equivalence_verified": False,
                "signal_computability_proven": False,
                "measurement_status": "measurement_not_run",
                "outcome_paths_read": [],
                "outcomes_used_as_evidence": False,
            }
        )
    )


_BINDING_FIELDS = frozenset({"name", "byte_sha256", "semantic_sha256"})


def validate_artifact_binding_v4_3(
    value: Mapping[str, Any], *, expected_name: str | None = None
) -> dict[str, str]:
    """Validate one byte+semantic predecessor descriptor."""

    payload = _exact(value, _BINDING_FIELDS, "artifact binding")
    name = payload["name"]
    if type(name) is not str or not name or name != name.strip():
        raise _error("artifact binding name must be an exact non-empty string")
    if expected_name is not None and name != expected_name:
        raise _error(f"artifact binding name must be {expected_name}")
    return {
        "name": name,
        "byte_sha256": _sha256(payload["byte_sha256"], f"{name}.byte_sha256"),
        "semantic_sha256": _sha256(
            payload["semantic_sha256"], f"{name}.semantic_sha256"
        ),
    }


def build_artifact_binding_v4_3(
    *, name: str, artifact: Mapping[str, Any]
) -> dict[str, str]:
    """Bind a self-sealed canonical JSON artifact by both identities."""

    if not isinstance(artifact, Mapping):
        raise _error("artifact must be an object")
    semantic = _artifact_semantic(artifact, name)
    return validate_artifact_binding_v4_3(
        {
            "name": name,
            "byte_sha256": byte_sha256(artifact),
            "semantic_sha256": semantic,
        },
        expected_name=name,
    )


def _cycle_state_binding(
    *, name: str, state: Mapping[str, Any], expected_state: str
) -> dict[str, str]:
    normalized = validate_cycle_state_v4_1(state, expected_state=expected_state)
    return validate_artifact_binding_v4_3(
        {
            "name": name,
            "byte_sha256": cycle_state_byte_sha256_v4_1(normalized),
            "semantic_sha256": normalized["state_semantic_sha256"],
        },
        expected_name=name,
    )


def validate_operator_semantics_v4_3(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact definition-only canonicalization semantics."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "semantics",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "operator semantics")
    if payload["schema_version"] != OPERATOR_SEMANTICS_SCHEMA_VERSION:
        raise _error("operator semantics schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["semantics"] != OPERATOR_SEMANTICS:
        raise _error("operator semantics must be exact")
    _exact_flags(payload)
    _artifact_semantic(payload, "operator semantics")
    return copy.deepcopy(payload)


def build_operator_semantics_v4_3(
    *, runtime_fingerprint: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build the non-executing AST/YAML semantics receipt."""

    runtime = validate_runtime_fingerprint_v4_3(
        runtime_fingerprint
        if runtime_fingerprint is not None
        else runtime_fingerprint_v4_3()
    )
    semantics = copy.deepcopy(OPERATOR_SEMANTICS)
    semantics["python_ast_canonicalization"] = runtime
    return validate_operator_semantics_v4_3(
        _seal(
            {
                "schema_version": OPERATOR_SEMANTICS_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "semantics": semantics,
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        )
    )


def _identity_inventory(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise _error("definition_identity_inventory must be a sequence")
    rows: list[dict[str, str]] = []
    previous_name: str | None = None
    for index, item in enumerate(value):
        row = _exact(
            item,
            frozenset({"name", "definition_identity_sha256"}),
            f"definition_identity_inventory[{index}]",
        )
        name = row["name"]
        if type(name) is not str or not name:
            raise _error("definition identity name must be non-empty")
        if previous_name is not None and name <= previous_name:
            raise _error("definition identity inventory must be sorted by name")
        previous_name = name
        rows.append(
            {
                "name": name,
                "definition_identity_sha256": _sha256(
                    row["definition_identity_sha256"],
                    f"definition_identity_inventory[{index}]",
                ),
            }
        )
    if not rows:
        raise _error("definition identity inventory must not be empty")
    return rows


def _comparison_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise _error("comparison_sources must be a sequence")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = _exact(
            item,
            frozenset(
                {
                    "name",
                    "byte_sha256",
                    "semantic_sha256",
                    "candidate_count",
                }
            ),
            f"comparison_sources[{index}]",
        )
        binding = validate_artifact_binding_v4_3(
            {
                "name": row["name"],
                "byte_sha256": row["byte_sha256"],
                "semantic_sha256": row["semantic_sha256"],
            }
        )
        rows.append(
            {
                **binding,
                "candidate_count": _positive_int(
                    row["candidate_count"],
                    f"comparison_sources[{index}].candidate_count",
                ),
            }
        )
    if tuple(row["name"] for row in rows) != EXPECTED_COMPARISON_SOURCE_NAMES:
        raise _error("comparison sources must be exact base230,v4_1,v4_2 order")
    return rows


def validate_comparison_catalog_receipt_v4_3(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed three-source no-label identity comparison catalog."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "catalog_id",
            "catalog_byte_sha256",
            "catalog_semantic_sha256",
            "comparison_sources",
            "candidate_count",
            "definition_identity_inventory",
            "label_inputs_absent",
            "outcome_fields_absent",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "comparison catalog receipt")
    if payload["schema_version"] != COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION:
        raise _error("comparison catalog schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if type(payload["catalog_id"]) is not str or not payload["catalog_id"]:
        raise _error("catalog_id must be non-empty")
    _sha256(payload["catalog_byte_sha256"], "catalog_byte_sha256")
    _sha256(payload["catalog_semantic_sha256"], "catalog_semantic_sha256")
    sources = _comparison_sources(payload["comparison_sources"])
    inventory = _identity_inventory(payload["definition_identity_inventory"])
    count = _positive_int(payload["candidate_count"], "candidate_count")
    if count != len(inventory):
        raise _error("candidate_count must equal identity inventory length")
    if payload["comparison_sources"] != sources:
        raise _error("comparison source descriptors are not normalized")
    if payload["definition_identity_inventory"] != inventory:
        raise _error("definition identity inventory is not normalized")
    _exact_bool(payload["label_inputs_absent"], "label_inputs_absent", True)
    _exact_bool(payload["outcome_fields_absent"], "outcome_fields_absent", True)
    _artifact_semantic(payload, "comparison catalog receipt")
    return copy.deepcopy(payload)


def build_comparison_catalog_receipt_v4_3(
    *,
    descriptor: Mapping[str, Any] | None = None,
    catalog_id: str | None = None,
    catalog_byte_sha256: str | None = None,
    catalog_semantic_sha256: str | None = None,
    comparison_sources: Sequence[Mapping[str, Any]] | None = None,
    definition_identity_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a comparison receipt from bundle-read identity descriptors."""

    if descriptor is not None:
        if any(
            item is not None
            for item in (
                catalog_id,
                catalog_byte_sha256,
                catalog_semantic_sha256,
                comparison_sources,
                definition_identity_inventory,
            )
        ):
            raise _error("descriptor cannot be combined with explicit fields")
        normalized = _exact(
            descriptor,
            frozenset(
                {
                    "catalog_id",
                    "catalog_byte_sha256",
                    "catalog_semantic_sha256",
                    "comparison_sources",
                    "definition_identity_inventory",
                }
            ),
            "comparison descriptor",
        )
        catalog_id = normalized["catalog_id"]
        catalog_byte_sha256 = normalized["catalog_byte_sha256"]
        catalog_semantic_sha256 = normalized["catalog_semantic_sha256"]
        comparison_sources = normalized["comparison_sources"]
        definition_identity_inventory = normalized[
            "definition_identity_inventory"
        ]
    if (
        catalog_id is None
        or catalog_byte_sha256 is None
        or catalog_semantic_sha256 is None
        or comparison_sources is None
        or definition_identity_inventory is None
    ):
        raise _error("complete comparison descriptor fields are required")
    sources = _comparison_sources(comparison_sources)
    inventory = _identity_inventory(definition_identity_inventory)
    return validate_comparison_catalog_receipt_v4_3(
        _seal(
            {
                "schema_version": COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "catalog_id": catalog_id,
                "catalog_byte_sha256": catalog_byte_sha256,
                "catalog_semantic_sha256": catalog_semantic_sha256,
                "comparison_sources": sources,
                "candidate_count": len(inventory),
                "definition_identity_inventory": inventory,
                "label_inputs_absent": True,
                "outcome_fields_absent": True,
            }
        )
    )


def _selection_candidate_rows(
    source: Mapping[str, Any], operator: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source_binding = build_artifact_binding_v4_3(
        name="aquant_source_set_receipt", artifact=source
    )
    operator_binding = build_artifact_binding_v4_3(
        name="operator_semantics", artifact=operator
    )
    rows: list[dict[str, Any]] = []
    for definition in source["candidate_definitions"]:
        rows.append(
            {
                "order": definition["order"],
                "name": definition["name"],
                "source": "A_quant",
                "source_alias": definition["source_alias"],
                "definition_intent": definition["definition_intent"],
                "family": definition["family"],
                "definition_identity_sha256": definition[
                    "definition_identity_sha256"
                ],
                "source_receipt_semantic_sha256": source_binding[
                    "semantic_sha256"
                ],
                "operator_semantics_sha256": operator_binding[
                    "semantic_sha256"
                ],
                "initial_weight": 0,
                "report_only": True,
            }
        )
    return rows


def _publication_payload(preregistered_at: Any) -> dict[str, Any]:
    timestamp, publication_date = _preregistered_at(preregistered_at)
    return {
        "preregistered_at": timestamp,
        "publication_date": publication_date,
        "timezone": "Asia/Shanghai",
        "publication_time_authority": "LOCAL_UNVERIFIED",
        "measurement_anchor_status": (
            "PENDING_INDEPENDENT_POST_PUBLICATION_EVIDENCE"
        ),
    }


def validate_selection_spec_v4_3(
    value: Mapping[str, Any],
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact ordered six-row definition-only preregistration."""

    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    operator = validate_operator_semantics_v4_3(operator_semantics)
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "predecessor_bindings",
            "candidate_count",
            "candidates",
            "publication",
            "time_policy",
            "claims",
            "blockers",
            "readiness_status",
            "measurement_status",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "selection spec")
    if payload["schema_version"] != SELECTION_SPEC_SCHEMA_VERSION:
        raise _error("selection spec schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    expected_bindings = [
        build_artifact_binding_v4_3(
            name="aquant_source_set_receipt", artifact=source
        ),
        build_artifact_binding_v4_3(
            name="operator_semantics", artifact=operator
        ),
    ]
    if payload["predecessor_bindings"] != expected_bindings:
        raise _error("selection predecessor byte/semantic bindings mismatch")
    if payload["candidate_count"] != len(EXPECTED_CANDIDATES):
        raise _error("candidate_count must be 6")
    expected_rows = _selection_candidate_rows(source, operator)
    if payload["candidates"] != expected_rows:
        raise _error("candidate rows/order/identity mismatch")
    if tuple(row["name"] for row in payload["candidates"]) != EXPECTED_CANDIDATES:
        raise _error("candidate allowlist mismatch")
    for index, row in enumerate(payload["candidates"], start=1):
        if row["order"] != index:
            raise _error("candidate order mismatch")
        if type(row["initial_weight"]) is not int or row["initial_weight"] != 0:
            raise _error("candidate initial_weight must be exact integer 0")
        _exact_bool(row["report_only"], "candidate.report_only", True)
    publication = _exact(
        payload["publication"],
        frozenset(
            {
                "preregistered_at",
                "publication_date",
                "timezone",
                "publication_time_authority",
                "measurement_anchor_status",
            }
        ),
        "publication",
    )
    expected_publication = _publication_payload(publication["preregistered_at"])
    if publication != expected_publication:
        raise _error("publication fields mismatch")
    if payload["time_policy"] != TIME_POLICY:
        raise _error("time policy must be exact")
    expected_claims = {
        "artifact_and_builder_label_inputs_absent": True,
        "outcome_paths_read": [],
        "outcomes_used_as_evidence": False,
        "selection_independence": "UNPROVEN",
        "authoritative_evidence_route": (
            "independent_post_publication_evidence_only"
        ),
        "report_only": True,
    }
    if payload["claims"] != expected_claims:
        raise _error("selection claims must be exact")
    if payload["blockers"] != list(BLOCKERS):
        raise _error("fixed blocker set/order mismatch")
    if payload["readiness_status"] != "PROSPECTIVE_PREREGISTRATION_ONLY":
        raise _error("readiness_status must be prospective only")
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    _exact_flags(payload)
    _artifact_semantic(payload, "selection spec")
    return copy.deepcopy(payload)


def build_selection_spec_v4_3(
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    preregistered_at: str,
) -> dict[str, Any]:
    """Build the exact six-candidate, zero-weight, report-only spec."""

    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    operator = validate_operator_semantics_v4_3(operator_semantics)
    publication = _publication_payload(preregistered_at)
    return validate_selection_spec_v4_3(
        _seal(
            {
                "schema_version": SELECTION_SPEC_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "predecessor_bindings": [
                    build_artifact_binding_v4_3(
                        name="aquant_source_set_receipt", artifact=source
                    ),
                    build_artifact_binding_v4_3(
                        name="operator_semantics", artifact=operator
                    ),
                ],
                "candidate_count": len(EXPECTED_CANDIDATES),
                "candidates": _selection_candidate_rows(source, operator),
                "publication": publication,
                "time_policy": copy.deepcopy(TIME_POLICY),
                "claims": {
                    "artifact_and_builder_label_inputs_absent": True,
                    "outcome_paths_read": [],
                    "outcomes_used_as_evidence": False,
                    "selection_independence": "UNPROVEN",
                    "authoritative_evidence_route": (
                        "independent_post_publication_evidence_only"
                    ),
                    "report_only": True,
                },
                "blockers": list(copy.deepcopy(BLOCKERS)),
                "readiness_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
                "measurement_status": "measurement_not_run",
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )


def validate_candidate_preregistration_v4_3(
    value: Mapping[str, Any],
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for the v4.3 selection validator."""

    return validate_selection_spec_v4_3(
        value,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
    )


def build_candidate_preregistration_v4_3(
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    preregistered_at: str,
) -> dict[str, Any]:
    """Compatibility alias for building the v4.3 selection spec."""

    return build_selection_spec_v4_3(
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
        preregistered_at=preregistered_at,
    )


def _external_binding(
    value: Mapping[str, Any], expected_name: str
) -> dict[str, str]:
    return validate_artifact_binding_v4_3(value, expected_name=expected_name)


def build_cycle_root_predecessor_bindings_v4_3(
    *,
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> list[dict[str, str]]:
    """Return exact selection/strict/code/future bindings for a cycle root."""

    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    future = validate_future_source_envelope_v4_3(
        future_source_envelope,
        selection_spec=selection,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    return [
        build_artifact_binding_v4_3(name="selection_spec", artifact=selection),
        strict,
        code,
        build_artifact_binding_v4_3(
            name="future_source_envelope", artifact=future
        ),
    ]


def validate_future_source_envelope_v4_3(
    value: Mapping[str, Any],
    *,
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Validate the definition-only publication/source envelope."""

    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "analysis_start",
            "cutoff",
            "snapshot_id",
            "snapshot_date",
            "latest_trade_date",
            "latest_complete_trade_date",
            "market",
            "universe",
            "storage_mode",
            "coverage",
            "full_a_scope_sha256",
            "full_a_scope_count",
            "serving_inventory_count",
            "predecessor_bindings",
            "publication",
            "time_policy",
            "blockers",
            "source_authenticity_verified",
            "definition_identity_verified",
            "healthy_source_verified",
            "readiness_status",
            "measurement_status",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "future source envelope")
    if payload["schema_version"] != SOURCE_ENVELOPE_SCHEMA_VERSION:
        raise _error("future source envelope schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    analysis_start = _date(payload["analysis_start"], "analysis_start")
    cutoff = _date(payload["cutoff"], "cutoff")
    snapshot_date = _date(payload["snapshot_date"], "snapshot_date")
    snapshot_id = _snapshot_id(
        payload["snapshot_id"], snapshot_date=snapshot_date
    )
    publication = selection["publication"]
    if date.fromisoformat(cutoff) > date.fromisoformat(
        publication["publication_date"]
    ):
        raise _error("cutoff must not follow the preregistration publication date")
    if date.fromisoformat(analysis_start) > date.fromisoformat(snapshot_date):
        raise _error("analysis_start must not follow snapshot_date")
    if snapshot_date != cutoff:
        raise _error("snapshot_date must equal the strict source cutoff")
    expected_cycle_id = (
        f"cn_full_a_v4_3_{cutoff.replace('-', '')}_{snapshot_id}"
    )
    if payload["cycle_id"] != expected_cycle_id:
        raise _error("cycle_id must bind v4.3 cutoff and snapshot_id")
    if payload["latest_trade_date"] != snapshot_date:
        raise _error("latest_trade_date must equal snapshot_date")
    if payload["latest_complete_trade_date"] != snapshot_date:
        raise _error("latest_complete_trade_date must equal snapshot_date")
    if payload["market"] != "CN" or payload["universe"] != "full_a":
        raise _error("source must be CN full_a")
    if payload["storage_mode"] != "strict_parquet":
        raise _error("storage_mode must be strict_parquet")
    expected_scope_count = _positive_int(
        full_a_scope_count, "full_a_scope_count"
    )
    expected_serving_count = _positive_int(
        serving_inventory_count, "serving_inventory_count"
    )
    expected_scope_sha = _sha256(full_a_scope_sha256, "full_a_scope_sha256")
    if payload["full_a_scope_sha256"] != expected_scope_sha:
        raise _error("full_a_scope_sha256 mismatch")
    if payload["full_a_scope_count"] != expected_scope_count:
        raise _error("full_a_scope_count mismatch")
    if payload["serving_inventory_count"] != expected_serving_count:
        raise _error("serving_inventory_count mismatch")
    expected_coverage = {
        "expected_scope_count": expected_scope_count,
        "complete_count": expected_scope_count,
        "coverage_ratio": 1,
    }
    if payload["coverage"] != expected_coverage:
        raise _error("coverage must be exact complete full-A")
    expected_predecessors = [
        build_artifact_binding_v4_3(
            name="selection_spec", artifact=selection
        ),
        strict,
        code,
    ]
    if payload["predecessor_bindings"] != expected_predecessors:
        raise _error("future envelope predecessor bindings mismatch")
    if payload["publication"] != publication:
        raise _error("publication must copy selection publication exactly")
    if payload["time_policy"] != TIME_POLICY:
        raise _error("time policy must be exact")
    if payload["blockers"] != list(BLOCKERS):
        raise _error("fixed blocker set/order mismatch")
    _exact_bool(
        payload["source_authenticity_verified"],
        "source_authenticity_verified",
        True,
    )
    _exact_bool(
        payload["definition_identity_verified"],
        "definition_identity_verified",
        True,
    )
    _exact_bool(
        payload["healthy_source_verified"],
        "healthy_source_verified",
        False,
    )
    if payload["readiness_status"] != "PROSPECTIVE_PREREGISTRATION_ONLY":
        raise _error("readiness_status must be prospective only")
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    _exact_flags(payload)
    _artifact_semantic(payload, "future source envelope")
    return copy.deepcopy(payload)


def build_future_source_envelope_v4_3(
    *,
    cycle_id: str,
    analysis_start: str,
    cutoff: str,
    snapshot_id: str,
    snapshot_date: str,
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Build a strict-source envelope that authorizes no measurement."""

    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    return validate_future_source_envelope_v4_3(
        _seal(
            {
                "schema_version": SOURCE_ENVELOPE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "cycle_id": cycle_id,
                "analysis_start": analysis_start,
                "cutoff": cutoff,
                "snapshot_id": snapshot_id,
                "snapshot_date": snapshot_date,
                "latest_trade_date": snapshot_date,
                "latest_complete_trade_date": snapshot_date,
                "market": "CN",
                "universe": "full_a",
                "storage_mode": "strict_parquet",
                "coverage": {
                    "expected_scope_count": full_a_scope_count,
                    "complete_count": full_a_scope_count,
                    "coverage_ratio": 1,
                },
                "full_a_scope_sha256": full_a_scope_sha256,
                "full_a_scope_count": full_a_scope_count,
                "serving_inventory_count": serving_inventory_count,
                "predecessor_bindings": [
                    build_artifact_binding_v4_3(
                        name="selection_spec", artifact=selection
                    ),
                    strict,
                    code,
                ],
                "publication": copy.deepcopy(selection["publication"]),
                "time_policy": copy.deepcopy(TIME_POLICY),
                "blockers": list(copy.deepcopy(BLOCKERS)),
                "source_authenticity_verified": True,
                "definition_identity_verified": True,
                "healthy_source_verified": False,
                "readiness_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
                "measurement_status": "measurement_not_run",
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        selection_spec=selection,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def _definition_identity_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row["name"]): str(row["definition_identity_sha256"])
        for row in rows
    }


def _identity_collisions(
    selected: Mapping[str, str], comparison: Mapping[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for selected_name, selected_sha in selected.items():
        for comparison_name, comparison_sha in comparison.items():
            if selected_name == comparison_name or selected_sha == comparison_sha:
                rows.append(
                    {
                        "selected": selected_name,
                        "comparison": comparison_name,
                        "reason": (
                            "name"
                            if selected_name == comparison_name
                            else "definition_identity_sha256"
                        ),
                    }
                )
    return sorted(rows, key=lambda row: (row["selected"], row["comparison"]))


def validate_definition_identity_collision_audit_v4_3(
    value: Mapping[str, Any],
    *,
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate identity equality only; this is not formal dedup evidence."""

    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator_semantics,
    )
    comparison = validate_comparison_catalog_receipt_v4_3(
        comparison_catalog_receipt
    )
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "predecessor_bindings",
            "method",
            "selected_vs_selected",
            "selected_vs_comparison",
            "definition_identity_collision_detected",
            "structural_dedup",
            "formal_dedup",
            "high_correlation_dedup",
            "measurement_status",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "definition identity collision audit")
    if (
        payload["schema_version"]
        != DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
    ):
        raise _error("definition identity collision audit schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    expected_predecessors = [
        build_artifact_binding_v4_3(
            name="aquant_source_set_receipt", artifact=source
        ),
        build_artifact_binding_v4_3(
            name="comparison_catalog_receipt", artifact=comparison
        ),
        build_artifact_binding_v4_3(
            name="selection_spec", artifact=selection
        ),
    ]
    if payload["predecessor_bindings"] != expected_predecessors:
        raise _error("collision audit predecessor bindings mismatch")
    if payload["method"] != "definition_identity_equality_only.v1":
        raise _error("definition identity collision method mismatch")
    selected = _definition_identity_map(selection["candidates"])
    if len(set(selected.values())) != len(selected):
        raise _error("selected definitions collide with each other")
    comparison_map = _definition_identity_map(
        comparison["definition_identity_inventory"]
    )
    collisions = _identity_collisions(selected, comparison_map)
    if collisions:
        raise _error("selected definitions collide with comparison catalog")
    if payload["selected_vs_selected"] != {
        "definition_identities": selected,
        "collisions": [],
    }:
        raise _error("selected-vs-selected identity audit mismatch")
    if payload["selected_vs_comparison"] != {
        "comparison_catalog_id": comparison["catalog_id"],
        "comparison_definition_identities": comparison_map,
        "collisions": [],
    }:
        raise _error("selected-vs-comparison identity audit mismatch")
    _exact_bool(
        payload["definition_identity_collision_detected"],
        "definition_identity_collision_detected",
        False,
    )
    for field in (
        "structural_dedup",
        "formal_dedup",
        "high_correlation_dedup",
    ):
        if payload[field] != "not_run":
            raise _error(f"{field} must be not_run")
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    if payload["authority"] != AUTHORITY_FLAGS:
        raise _error("authority flags must be exact false values")
    if payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("side_effect flags must be exact false values")
    _artifact_semantic(payload, "definition identity collision audit")
    return copy.deepcopy(payload)


def build_definition_identity_collision_audit_v4_3(
    *,
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact identity-equality diagnostic."""

    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator_semantics,
    )
    comparison = validate_comparison_catalog_receipt_v4_3(
        comparison_catalog_receipt
    )
    selected = _definition_identity_map(selection["candidates"])
    comparison_map = _definition_identity_map(
        comparison["definition_identity_inventory"]
    )
    if _identity_collisions(selected, comparison_map):
        raise _error("selected definitions collide with comparison catalog")
    return validate_definition_identity_collision_audit_v4_3(
        _seal(
            {
                "schema_version": (
                    DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
                ),
                "protocol_version": PROTOCOL_VERSION,
                "predecessor_bindings": [
                    build_artifact_binding_v4_3(
                        name="aquant_source_set_receipt", artifact=source
                    ),
                    build_artifact_binding_v4_3(
                        name="comparison_catalog_receipt", artifact=comparison
                    ),
                    build_artifact_binding_v4_3(
                        name="selection_spec", artifact=selection
                    ),
                ],
                "method": "definition_identity_equality_only.v1",
                "selected_vs_selected": {
                    "definition_identities": selected,
                    "collisions": [],
                },
                "selected_vs_comparison": {
                    "comparison_catalog_id": comparison["catalog_id"],
                    "comparison_definition_identities": comparison_map,
                    "collisions": [],
                },
                "definition_identity_collision_detected": False,
                "structural_dedup": "not_run",
                "formal_dedup": "not_run",
                "high_correlation_dedup": "not_run",
                "measurement_status": "measurement_not_run",
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison,
    )


def build_precommit_source_chain_sha256_v4_3(
    future_source_envelope: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
) -> str:
    """Bind future and collision artifacts into the PRECOMMITTED source edge."""

    bindings = [
        build_artifact_binding_v4_3(
            name="future_source_envelope", artifact=future_source_envelope
        ),
        build_artifact_binding_v4_3(
            name="definition_identity_collision_audit",
            artifact=definition_identity_collision_audit,
        ),
    ]
    return domain_separated_sha256_v4_3(
        "factor-governance-precommit-source-chain.v4.3",
        {"ordered_predecessor_bindings": bindings},
    )


def validate_prereg_discovery_source_node_v4_3(
    value: Mapping[str, Any],
    *,
    predecessor_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    cycle_root_binding: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Validate the PRECOMMITTED-to-DISCOVERY definition source node."""

    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    operator = validate_operator_semantics_v4_3(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_3(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    future = validate_future_source_envelope_v4_3(
        future_source_envelope,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    collision = validate_definition_identity_collision_audit_v4_3(
        definition_identity_collision_audit,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
    )
    if predecessor["cycle_id"] != future["cycle_id"]:
        raise _error("predecessor and future envelope cycle_id mismatch")
    cycle_root = _external_binding(cycle_root_binding, "cycle_root")
    if cycle_root["semantic_sha256"] != predecessor["cycle_root_sha256"]:
        raise _error("cycle_root binding semantic SHA mismatch")
    expected_precommit_source_chain = build_precommit_source_chain_sha256_v4_3(
        future, collision
    )
    if predecessor["source_chain_node_sha256"] != expected_precommit_source_chain:
        raise _error("PRECOMMITTED source chain must bind future and collision")
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "state_schema_version",
            "cycle_id",
            "cycle_root_sha256",
            "predecessor_bindings",
            "context_bindings",
            "selection_claims",
            "publication",
            "time_policy",
            "blockers",
            "readiness_status",
            "measurement_status",
            "measurement",
            "authority",
            "side_effects",
            "dual_sha_predecessor_transition_validated",
            "exact_once_publication",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "prereg discovery source node")
    if payload["schema_version"] != DISCOVERY_SOURCE_NODE_SCHEMA_VERSION:
        raise _error("prereg discovery source node schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["state_schema_version"] != STATE_SCHEMA_VERSION:
        raise _error("cycle state schema mismatch")
    if payload["cycle_id"] != predecessor["cycle_id"]:
        raise _error("source node cycle_id mismatch")
    if payload["cycle_root_sha256"] != predecessor["cycle_root_sha256"]:
        raise _error("source node cycle root mismatch")
    expected_predecessors = [
        _cycle_state_binding(
            name="precommitted_state",
            state=predecessor,
            expected_state=PRECOMMITTED,
        ),
        build_artifact_binding_v4_3(
            name="selection_spec", artifact=selection
        ),
        build_artifact_binding_v4_3(
            name="aquant_source_set_receipt", artifact=source
        ),
    ]
    if payload["predecessor_bindings"] != expected_predecessors:
        raise _error("source node predecessor bindings mismatch")
    expected_context = [
        cycle_root,
        build_artifact_binding_v4_3(
            name="operator_semantics", artifact=operator
        ),
        build_artifact_binding_v4_3(
            name="comparison_catalog_receipt", artifact=comparison
        ),
        strict,
        code,
        build_artifact_binding_v4_3(
            name="future_source_envelope", artifact=future
        ),
        build_artifact_binding_v4_3(
            name="definition_identity_collision_audit", artifact=collision
        ),
    ]
    if payload["context_bindings"] != expected_context:
        raise _error("source node context bindings mismatch")
    if payload["selection_claims"] != selection["claims"]:
        raise _error("source node selection claims mismatch")
    if payload["publication"] != selection["publication"]:
        raise _error("source node publication mismatch")
    if payload["time_policy"] != TIME_POLICY:
        raise _error("source node time policy mismatch")
    if payload["blockers"] != list(BLOCKERS):
        raise _error("source node blocker set/order mismatch")
    if payload["readiness_status"] != "PROSPECTIVE_PREREGISTRATION_ONLY":
        raise _error("readiness_status must be prospective only")
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    _exact_flags(payload)
    _exact_bool(
        payload["dual_sha_predecessor_transition_validated"],
        "dual_sha_predecessor_transition_validated",
        True,
    )
    if payload["exact_once_publication"] != "NOT_IMPLEMENTED":
        raise _error("exact_once_publication must be NOT_IMPLEMENTED")
    _artifact_semantic(payload, "prereg discovery source node")
    return copy.deepcopy(payload)


def build_prereg_discovery_source_node_v4_3(
    *,
    predecessor_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    cycle_root_binding: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Build the exact definition-only DISCOVERY source node."""

    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    operator = validate_operator_semantics_v4_3(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_3(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    future = validate_future_source_envelope_v4_3(
        future_source_envelope,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    collision = validate_definition_identity_collision_audit_v4_3(
        definition_identity_collision_audit,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
    )
    cycle_root = _external_binding(cycle_root_binding, "cycle_root")
    return validate_prereg_discovery_source_node_v4_3(
        _seal(
            {
                "schema_version": DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "cycle_id": predecessor["cycle_id"],
                "cycle_root_sha256": predecessor["cycle_root_sha256"],
                "predecessor_bindings": [
                    _cycle_state_binding(
                        name="precommitted_state",
                        state=predecessor,
                        expected_state=PRECOMMITTED,
                    ),
                    build_artifact_binding_v4_3(
                        name="selection_spec", artifact=selection
                    ),
                    build_artifact_binding_v4_3(
                        name="aquant_source_set_receipt", artifact=source
                    ),
                ],
                "context_bindings": [
                    cycle_root,
                    build_artifact_binding_v4_3(
                        name="operator_semantics", artifact=operator
                    ),
                    build_artifact_binding_v4_3(
                        name="comparison_catalog_receipt", artifact=comparison
                    ),
                    strict,
                    code,
                    build_artifact_binding_v4_3(
                        name="future_source_envelope", artifact=future
                    ),
                    build_artifact_binding_v4_3(
                        name="definition_identity_collision_audit",
                        artifact=collision,
                    ),
                ],
                "selection_claims": copy.deepcopy(selection["claims"]),
                "publication": copy.deepcopy(selection["publication"]),
                "time_policy": copy.deepcopy(TIME_POLICY),
                "blockers": list(copy.deepcopy(BLOCKERS)),
                "readiness_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
                "measurement_status": "measurement_not_run",
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
                "dual_sha_predecessor_transition_validated": True,
                "exact_once_publication": "NOT_IMPLEMENTED",
            }
        ),
        predecessor_state=predecessor,
        future_source_envelope=future,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        cycle_root_binding=cycle_root,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def build_preregistration_source_node_v4_3(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the v4.3 DISCOVERY source-node builder."""

    return build_prereg_discovery_source_node_v4_3(**kwargs)


def _build_preregistration_discovery_cycle_payload_v4_3(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    cycle_root_binding: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    actual_predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    if _sha256(predecessor_byte_sha256, "predecessor_byte_sha256") != actual_predecessor_byte:
        raise _error("predecessor byte SHA mismatch")
    if expected_predecessor_byte_sha256 != actual_predecessor_byte:
        raise _error("expected predecessor byte SHA mismatch")
    if expected_predecessor_semantic_sha256 != predecessor["state_semantic_sha256"]:
        raise _error("expected predecessor semantic SHA mismatch")
    source = validate_aquant_source_set_receipt_v4_3(
        aquant_source_set_receipt
    )
    operator = validate_operator_semantics_v4_3(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_3(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_3(
        selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )
    strict = _external_binding(strict_source_binding, "strict_source_binding")
    code = _external_binding(code_binding_set, "code_binding_set")
    future = validate_future_source_envelope_v4_3(
        future_source_envelope,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    collision = validate_definition_identity_collision_audit_v4_3(
        definition_identity_collision_audit,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
    )
    cycle_root = _external_binding(cycle_root_binding, "cycle_root")
    source_node = build_prereg_discovery_source_node_v4_3(
        predecessor_state=predecessor,
        future_source_envelope=future,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        cycle_root_binding=cycle_root,
        strict_source_binding=strict,
        code_binding_set=code,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    discovery = build_next_cycle_state_v4_1(
        predecessor=predecessor,
        predecessor_byte_sha256=actual_predecessor_byte,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        cycle_id=predecessor["cycle_id"],
        cycle_root_sha256=predecessor["cycle_root_sha256"],
        next_state=DISCOVERY,
        source_chain_node_sha256=source_node["artifact_semantic_sha256"],
    )
    graph_bindings = [
        _cycle_state_binding(
            name="precommitted_state",
            state=predecessor,
            expected_state=PRECOMMITTED,
        ),
        cycle_root,
        build_artifact_binding_v4_3(
            name="aquant_source_set_receipt", artifact=source
        ),
        build_artifact_binding_v4_3(
            name="operator_semantics", artifact=operator
        ),
        build_artifact_binding_v4_3(
            name="comparison_catalog_receipt", artifact=comparison
        ),
        build_artifact_binding_v4_3(
            name="selection_spec", artifact=selection
        ),
        strict,
        code,
        build_artifact_binding_v4_3(
            name="future_source_envelope", artifact=future
        ),
        build_artifact_binding_v4_3(
            name="definition_identity_collision_audit", artifact=collision
        ),
        build_artifact_binding_v4_3(
            name="prereg_discovery_source_node", artifact=source_node
        ),
        _cycle_state_binding(
            name="discovery_state", state=discovery, expected_state=DISCOVERY
        ),
    ]
    return _seal(
        {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "predecessor_state": predecessor,
            "source_node": source_node,
            "discovery_state": discovery,
            "graph_bindings": graph_bindings,
            "persisted_state_sequence": [PRECOMMITTED, DISCOVERY],
            "precommitted_state_role": "INTRA_BUNDLE_LINEAGE_ONLY",
            "discovery_state_role": "FINAL_CURRENT",
            "external_state_pointer_mutation": False,
            "selection_claims": copy.deepcopy(selection["claims"]),
            "publication": copy.deepcopy(selection["publication"]),
            "time_policy": copy.deepcopy(TIME_POLICY),
            "blockers": list(copy.deepcopy(BLOCKERS)),
            "readiness_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
            "measurement_status": "measurement_not_run",
            "dual_sha_predecessor_transition_validated": True,
            "exact_once_publication": "NOT_IMPLEMENTED",
            "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
            "authority": copy.deepcopy(AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
        }
    )


def validate_preregistration_discovery_cycle_v4_3(
    value: Mapping[str, Any],
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    cycle_root_binding: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Rebuild and compare the complete v4.3 PRECOMMITTED/DISCOVERY graph."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "state_schema_version",
            "predecessor_state",
            "source_node",
            "discovery_state",
            "graph_bindings",
            "persisted_state_sequence",
            "precommitted_state_role",
            "discovery_state_role",
            "external_state_pointer_mutation",
            "selection_claims",
            "publication",
            "time_policy",
            "blockers",
            "readiness_status",
            "measurement_status",
            "dual_sha_predecessor_transition_validated",
            "exact_once_publication",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "prereg discovery orchestration")
    if payload["schema_version"] != ORCHESTRATION_SCHEMA_VERSION:
        raise _error("prereg discovery orchestration schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["state_schema_version"] != STATE_SCHEMA_VERSION:
        raise _error("cycle state schema mismatch")
    if payload["persisted_state_sequence"] != [PRECOMMITTED, DISCOVERY]:
        raise _error("persisted state sequence mismatch")
    if payload["precommitted_state_role"] != "INTRA_BUNDLE_LINEAGE_ONLY":
        raise _error("PRECOMMITTED role mismatch")
    if payload["discovery_state_role"] != "FINAL_CURRENT":
        raise _error("DISCOVERY role mismatch")
    _exact_bool(
        payload["external_state_pointer_mutation"],
        "external_state_pointer_mutation",
        False,
    )
    _exact_bool(
        payload["dual_sha_predecessor_transition_validated"],
        "dual_sha_predecessor_transition_validated",
        True,
    )
    if payload["exact_once_publication"] != "NOT_IMPLEMENTED":
        raise _error("exact_once_publication must be NOT_IMPLEMENTED")
    if payload["time_policy"] != TIME_POLICY:
        raise _error("orchestration time policy mismatch")
    if payload["blockers"] != list(BLOCKERS):
        raise _error("orchestration blocker set/order mismatch")
    if payload["readiness_status"] != "PROSPECTIVE_PREREGISTRATION_ONLY":
        raise _error("readiness_status must be prospective only")
    if payload["measurement_status"] != "measurement_not_run":
        raise _error("measurement_status must be measurement_not_run")
    _exact_flags(payload)
    _artifact_semantic(payload, "prereg discovery orchestration")
    expected = _build_preregistration_discovery_cycle_payload_v4_3(
        predecessor_state=predecessor_state,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        cycle_root_binding=cycle_root_binding,
        strict_source_binding=strict_source_binding,
        code_binding_set=code_binding_set,
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    if payload != expected:
        raise _error("prereg discovery orchestration graph mismatch")
    return copy.deepcopy(payload)


def build_preregistration_discovery_cycle_v4_3(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    cycle_root_binding: Mapping[str, Any],
    strict_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Build and validate the pure v4.3 DISCOVERY orchestration."""

    kwargs: dict[str, Any] = {
        "predecessor_state": predecessor_state,
        "predecessor_byte_sha256": predecessor_byte_sha256,
        "expected_predecessor_byte_sha256": expected_predecessor_byte_sha256,
        "expected_predecessor_semantic_sha256": (
            expected_predecessor_semantic_sha256
        ),
        "future_source_envelope": future_source_envelope,
        "selection_spec": selection_spec,
        "aquant_source_set_receipt": aquant_source_set_receipt,
        "operator_semantics": operator_semantics,
        "comparison_catalog_receipt": comparison_catalog_receipt,
        "definition_identity_collision_audit": (
            definition_identity_collision_audit
        ),
        "cycle_root_binding": cycle_root_binding,
        "strict_source_binding": strict_source_binding,
        "code_binding_set": code_binding_set,
        "full_a_scope_sha256": full_a_scope_sha256,
        "full_a_scope_count": full_a_scope_count,
        "serving_inventory_count": serving_inventory_count,
    }
    payload = _build_preregistration_discovery_cycle_payload_v4_3(**kwargs)
    return validate_preregistration_discovery_cycle_v4_3(payload, **kwargs)


def build_discovery_cycle_state_v4_3(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias returning the full v4.3 orchestration."""

    return build_preregistration_discovery_cycle_v4_3(**kwargs)


AQUANT_SOURCE_SPECS_V4_3 = SOURCE_BINDINGS
AQUANT_COMMIT_V4_3 = AQUANT_COMMIT
RUNTIME_FINGERPRINT_V4_3 = AST_RUNTIME_FINGERPRINT
EXPECTED_CANDIDATES_V4_3 = EXPECTED_CANDIDATES
BLOCKERS_V4_3 = BLOCKERS
TIME_POLICY_V4_3 = TIME_POLICY
MEASUREMENT_FLAGS_V4_3 = MEASUREMENT_FLAGS
AUTHORITY_FLAGS_V4_3 = AUTHORITY_FLAGS
SIDE_EFFECT_FLAGS_V4_3 = SIDE_EFFECT_FLAGS


__all__ = [
    "AQUANT_COMMIT",
    "AQUANT_COMMIT_V4_3",
    "AQUANT_GIT_TOP",
    "AQUANT_SOURCE_SPECS_V4_3",
    "AST_RUNTIME_FINGERPRINT",
    "AUTHORITY_FLAGS",
    "AUTHORITY_FLAGS_V4_3",
    "BLOCKERS",
    "BLOCKERS_V4_3",
    "COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION",
    "DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION",
    "DISCOVERY_SOURCE_NODE_SCHEMA_VERSION",
    "EXPECTED_CANDIDATES",
    "EXPECTED_CANDIDATES_V4_3",
    "EXPECTED_CANDIDATE_DEFINITION_ROWS",
    "EXPECTED_SELECTOR_BINDINGS",
    "MEASUREMENT_FLAGS",
    "MEASUREMENT_FLAGS_V4_3",
    "OPERATOR_SEMANTICS",
    "OPERATOR_SEMANTICS_SCHEMA_VERSION",
    "ORCHESTRATION_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "RUNTIME_FINGERPRINT_V4_3",
    "SCHEMA_VERSION",
    "SELECTION_SPEC_SCHEMA_VERSION",
    "SIDE_EFFECT_FLAGS",
    "SIDE_EFFECT_FLAGS_V4_3",
    "SOURCE_BINDINGS",
    "SOURCE_ENVELOPE_SCHEMA_VERSION",
    "SOURCE_SET_RECEIPT_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "TIME_POLICY",
    "TIME_POLICY_V4_3",
    "FactorGovernanceCandidatePreregistrationV4_3Error",
    "FactorGovernanceCandidatePreregistrationV43Error",
    "build_aquant_source_set_receipt_v4_3",
    "build_artifact_binding_v4_3",
    "build_candidate_preregistration_v4_3",
    "build_comparison_catalog_receipt_v4_3",
    "build_cycle_root_predecessor_bindings_v4_3",
    "build_definition_identity_collision_audit_v4_3",
    "build_discovery_cycle_state_v4_3",
    "build_future_source_envelope_v4_3",
    "build_operator_semantics_v4_3",
    "build_precommit_source_chain_sha256_v4_3",
    "build_prereg_discovery_source_node_v4_3",
    "build_preregistration_discovery_cycle_v4_3",
    "build_preregistration_source_node_v4_3",
    "build_selection_spec_v4_3",
    "byte_sha256",
    "byte_sha256_v4_3",
    "canonical_file_bytes",
    "canonical_file_bytes_v4_3",
    "canonical_json_bytes",
    "canonical_json_bytes_v4_3",
    "domain_separated_sha256_v4_3",
    "runtime_fingerprint_v4_3",
    "semantic_sha256",
    "semantic_sha256_v4_3",
    "validate_aquant_source_set_receipt_v4_3",
    "validate_artifact_binding_v4_3",
    "validate_candidate_preregistration_v4_3",
    "validate_comparison_catalog_receipt_v4_3",
    "validate_definition_identity_collision_audit_v4_3",
    "validate_future_source_envelope_v4_3",
    "validate_operator_semantics_v4_3",
    "validate_prereg_discovery_source_node_v4_3",
    "validate_preregistration_discovery_cycle_v4_3",
    "validate_runtime_fingerprint_v4_3",
    "validate_selection_spec_v4_3",
]
