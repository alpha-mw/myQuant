"""Canonical JSON contracts for the V17-only mainline.

The package deliberately validates a small frozen authority carrier.  It does
not publish production artifacts and it never treats Shadow or run-forward
artifacts as authority.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any, Final

from .constants import (
    ACTIVE_POINTER_SCHEMA_ID,
    AUTHORITY_SOURCE,
    FORMAL_OUTPUT_SCHEMA_ID,
    MAINLINE_RUN_SCHEMA_ID,
    PORTFOLIO_OUTPUT_SCHEMA_ID,
    PROTOCOL,
    SOURCE_CLOSURE_SCHEMA_ID,
    SUPPORTED_CAPABILITY,
    SUPPORTED_MARKET,
)

_ID: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA: Final = re.compile(r"^[0-9a-f]{64}$")
_UTC: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SYMBOL: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_DECIMAL: Final = re.compile(r"^(?:0(?:\.\d{1,18})?|1(?:\.0{1,18})?)$")
_REF_FIELDS: Final = frozenset({"schema_id", "relative_path", "byte_sha256"})


class MainlineContractError(ValueError):
    """Raised when exact mainline contract validation fails."""


def _canonical_payload_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            dict(document),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise MainlineContractError("document is not canonical JSON data") from exc
    return text.encode("utf-8")


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return _canonical_payload_bytes(document) + b"\n"


def byte_sha256(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise MainlineContractError("payload must be exact bytes")
    return hashlib.sha256(raw).hexdigest()


def seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("semantic_sha256", None)
    payload["semantic_sha256"] = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    return payload


def parse_canonical(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise MainlineContractError("artifact must be bytes")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainlineContractError("artifact is not canonical JSON") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise MainlineContractError("artifact bytes are not canonical")
    validate_semantic(value)
    return value


def validate_semantic(document: Mapping[str, Any]) -> None:
    observed = document.get("semantic_sha256")
    if type(observed) is not str or _SHA.fullmatch(observed) is None:
        raise MainlineContractError("semantic SHA-256 is invalid")
    body = dict(document)
    del body["semantic_sha256"]
    expected = hashlib.sha256(_canonical_payload_bytes(body)).hexdigest()
    if observed != expected:
        raise MainlineContractError("semantic SHA-256 mismatch")


def require_identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) > 80 or _ID.fullmatch(value) is None:
        raise MainlineContractError(f"{label} is not a canonical identifier")
    return value


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise MainlineContractError(f"{label} is not a canonical SHA-256")
    return value


def require_timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise MainlineContractError(f"{label} is not a canonical UTC timestamp")
    return value


def validate_ref(
    value: Any,
    *,
    label: str,
    expected_schema_id: str | None = None,
    required_prefix: str | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise MainlineContractError(f"{label} is not an exact artifact ref")
    schema_id = value.get("schema_id")
    path = value.get("relative_path")
    digest = value.get("byte_sha256")
    if type(schema_id) is not str or not schema_id.startswith("myquant."):
        raise MainlineContractError(f"{label}.schema_id is invalid")
    if expected_schema_id is not None and schema_id != expected_schema_id:
        raise MainlineContractError(f"{label}.schema_id mismatch")
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or (required_prefix is not None and not path.startswith(required_prefix + "/"))
    ):
        raise MainlineContractError(f"{label}.relative_path is invalid")
    return {
        "schema_id": schema_id,
        "relative_path": path,
        "byte_sha256": require_sha256(digest, label=f"{label}.byte_sha256"),
    }


def _exact_fields(document: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    if set(document) != fields:
        raise MainlineContractError(f"{label} fields are not exact")


def validate_mainline_run(document: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        document,
        {
            "schema_id",
            "protocol",
            "canonical_strategy_id",
            "run_id",
            "created_at",
            "market",
            "capabilities",
            "authority_source",
            "formal_output_ref",
            "portfolio_output_ref",
            "source_closure_ref",
            "semantic_sha256",
        },
        label="mainline run",
    )
    validate_semantic(document)
    if document.get("schema_id") != MAINLINE_RUN_SCHEMA_ID or document.get("protocol") != PROTOCOL:
        raise MainlineContractError("mainline run schema/protocol mismatch")
    strategy = require_identifier(
        document.get("canonical_strategy_id"), label="canonical_strategy_id"
    )
    run_id = require_identifier(document.get("run_id"), label="run_id")
    require_timestamp(document.get("created_at"), label="created_at")
    capabilities = document.get("capabilities")
    if type(capabilities) is not list or capabilities != [SUPPORTED_CAPABILITY]:
        raise MainlineContractError("unsupported mainline capability")
    if document.get("market") != SUPPORTED_MARKET:
        raise MainlineContractError("unsupported mainline market")
    if document.get("authority_source") != AUTHORITY_SOURCE:
        raise MainlineContractError("mainline authority is not formal V17 v4")
    return {
        **dict(document),
        "canonical_strategy_id": strategy,
        "run_id": run_id,
        "formal_output_ref": validate_ref(
            document.get("formal_output_ref"),
            label="formal_output_ref",
            expected_schema_id=FORMAL_OUTPUT_SCHEMA_ID,
            required_prefix="results/v17_v4_formal_research",
        ),
        "portfolio_output_ref": validate_ref(
            document.get("portfolio_output_ref"),
            label="portfolio_output_ref",
            expected_schema_id=PORTFOLIO_OUTPUT_SCHEMA_ID,
            required_prefix="results/v17_v4_formal_research",
        ),
        "source_closure_ref": validate_ref(
            document.get("source_closure_ref"),
            label="source_closure_ref",
            expected_schema_id=SOURCE_CLOSURE_SCHEMA_ID,
            required_prefix="data/private/v17_v4_sources",
        ),
    }


def validate_active_pointer(document: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        document,
        {
            "schema_id",
            "protocol",
            "canonical_strategy_id",
            "run_id",
            "updated_at",
            "run_ref",
            "semantic_sha256",
        },
        label="active pointer",
    )
    validate_semantic(document)
    if (
        document.get("schema_id") != ACTIVE_POINTER_SCHEMA_ID
        or document.get("protocol") != PROTOCOL
    ):
        raise MainlineContractError("active pointer schema/protocol mismatch")
    strategy = require_identifier(
        document.get("canonical_strategy_id"), label="canonical_strategy_id"
    )
    run_id = require_identifier(document.get("run_id"), label="run_id")
    require_timestamp(document.get("updated_at"), label="updated_at")
    run_ref = validate_ref(
        document.get("run_ref"),
        label="run_ref",
        expected_schema_id=MAINLINE_RUN_SCHEMA_ID,
        required_prefix=f"results/v17_mainline/strategies/{strategy}/runs",
    )
    expected_path = f"results/v17_mainline/strategies/{strategy}/runs/{run_id}/run.json"
    if run_ref["relative_path"] != expected_path:
        raise MainlineContractError("active pointer run path is not canonical")
    return {**dict(document), "run_ref": run_ref}


def validate_formal_output(document: Mapping[str, Any], *, strategy_id: str) -> dict[str, Any]:
    validate_semantic(document)
    authority = document.get("authority")
    if (
        document.get("version") != FORMAL_OUTPUT_SCHEMA_ID
        or document.get("protocol_version") != PROTOCOL
        or document.get("strategy_id") != strategy_id
        or document.get("terminal_state") != "PUBLISHED_RESEARCH_ONLY"
        or type(authority) is not dict
        or authority.get("formal_research_publication") is not True
        or authority.get("broker") is not False
        or authority.get("execution") is not False
        or authority.get("order") is not False
        or authority.get("trade") is not False
        or document.get("shadow_only") is True
        or document.get("run_state") == "FORWARD_EVIDENCE_ACTIVE"
    ):
        raise MainlineContractError("formal output authority is invalid")
    return dict(document)


def _decimal_string(value: Any, *, label: str) -> str:
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise MainlineContractError(f"{label} must be a canonical decimal string in [0,1]")
    return value


def validate_portfolio_output(
    document: Mapping[str, Any],
    *,
    strategy_id: str,
    run_id: str,
) -> dict[str, Any]:
    validate_semantic(document)
    targets = document.get("targets")
    if (
        document.get("version") != PORTFOLIO_OUTPUT_SCHEMA_ID
        or document.get("protocol_version") != PROTOCOL
        or document.get("strategy_id") != strategy_id
        or document.get("run_id") != run_id
        or document.get("status") != "COMPLETE"
        or type(targets) is not list
        or not targets
        or document.get("shadow_only") is True
        or document.get("run_state") == "FORWARD_EVIDENCE_ACTIVE"
    ):
        raise MainlineContractError("portfolio output is invalid")
    normalized_targets: list[dict[str, str]] = []
    previous = ""
    for row in targets:
        if type(row) is not dict or set(row) != {
            "symbol",
            "current_target",
            "final_target",
            "lane",
        }:
            raise MainlineContractError("portfolio target fields are invalid")
        symbol = row.get("symbol")
        if type(symbol) is not str or _SYMBOL.fullmatch(symbol) is None or symbol <= previous:
            raise MainlineContractError("portfolio targets are not strictly ordered")
        lane = row.get("lane")
        if lane not in {"SELECTION_POOL", "REVIEW_ONLY_HOLDING"}:
            raise MainlineContractError("portfolio target lane is invalid")
        normalized_targets.append(
            {
                "symbol": symbol,
                "current_target": _decimal_string(
                    row.get("current_target"), label="current_target"
                ),
                "final_target": _decimal_string(row.get("final_target"), label="final_target"),
                "lane": lane,
            }
        )
        previous = symbol
    result = dict(document)
    result["cash_weight"] = _decimal_string(document.get("cash_weight"), label="cash_weight")
    result["gross_weight"] = _decimal_string(document.get("gross_weight"), label="gross_weight")
    result["targets"] = normalized_targets
    return result


def validate_source_closure(document: Mapping[str, Any], *, strategy_id: str) -> dict[str, Any]:
    validate_semantic(document)
    if (
        document.get("version") != SOURCE_CLOSURE_SCHEMA_ID
        or document.get("protocol_version") != PROTOCOL
        or document.get("strategy_id") != strategy_id
        or require_sha256(document.get("source_closure_sha256"), label="source_closure_sha256")
        != document.get("source_closure_sha256")
        or document.get("shadow_only") is True
        or document.get("run_state") == "FORWARD_EVIDENCE_ACTIVE"
    ):
        raise MainlineContractError("source closure is invalid")
    return dict(document)


def build_ref(*, schema_id: str, relative_path: str, raw: bytes) -> dict[str, str]:
    return validate_ref(
        {"schema_id": schema_id, "relative_path": relative_path, "byte_sha256": byte_sha256(raw)},
        label="artifact_ref",
        expected_schema_id=schema_id,
    )


def build_mainline_run(
    *,
    canonical_strategy_id: str,
    run_id: str,
    created_at: str,
    formal_output_ref: Mapping[str, Any],
    portfolio_output_ref: Mapping[str, Any],
    source_closure_ref: Mapping[str, Any],
) -> dict[str, Any]:
    document = seal_document(
        {
            "schema_id": MAINLINE_RUN_SCHEMA_ID,
            "protocol": PROTOCOL,
            "canonical_strategy_id": canonical_strategy_id,
            "run_id": run_id,
            "created_at": created_at,
            "market": SUPPORTED_MARKET,
            "capabilities": [SUPPORTED_CAPABILITY],
            "authority_source": AUTHORITY_SOURCE,
            "formal_output_ref": dict(formal_output_ref),
            "portfolio_output_ref": dict(portfolio_output_ref),
            "source_closure_ref": dict(source_closure_ref),
        }
    )
    return validate_mainline_run(document)


def build_active_pointer(
    *,
    canonical_strategy_id: str,
    run_id: str,
    updated_at: str,
    run_ref: Mapping[str, Any],
) -> dict[str, Any]:
    document = seal_document(
        {
            "schema_id": ACTIVE_POINTER_SCHEMA_ID,
            "protocol": PROTOCOL,
            "canonical_strategy_id": canonical_strategy_id,
            "run_id": run_id,
            "updated_at": updated_at,
            "run_ref": dict(run_ref),
        }
    )
    return validate_active_pointer(document)


def contains_forbidden_authority(value: Any) -> bool:
    if type(value) is dict:
        for key, item in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {"shadow_only", "run_forward_authority"} and item is True:
                return True
            if (
                lowered_key in {"authority_source", "authority_kind", "source_lane", "run_state"}
                and type(item) is str
            ):
                upper = item.upper()
                if "SHADOW" in upper or "FORWARD_EVIDENCE" in upper or "RUN_FORWARD" in upper:
                    return True
            if contains_forbidden_authority(item):
                return True
        return False
    if type(value) is list:
        return any(contains_forbidden_authority(item) for item in value)
    return False


__all__ = [
    "MainlineContractError",
    "build_active_pointer",
    "build_mainline_run",
    "build_ref",
    "byte_sha256",
    "canonical_bytes",
    "contains_forbidden_authority",
    "parse_canonical",
    "require_identifier",
    "require_sha256",
    "seal_document",
    "validate_active_pointer",
    "validate_formal_output",
    "validate_mainline_run",
    "validate_portfolio_output",
    "validate_source_closure",
]
