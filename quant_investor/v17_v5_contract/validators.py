"""Semantic validators for V17 v5 artifacts."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
from typing import Any, Final, Mapping

from .canonical import CanonicalContractError, canonical_bytes, validate_semantic_sha
from .identities import (
    IdentityContractError,
    require_git_commit,
    require_identifier,
    require_relative_path,
    require_sha256,
)

NO_AUTHORITY: Final = {
    "broker": False,
    "canary": False,
    "execution": False,
    "factor_governance_write": False,
    "formal_activation": False,
    "formal_research_publication": False,
    "llm": False,
    "order": False,
    "portfolio": False,
    "promotion": False,
    "provider": False,
    "research_runtime_default": False,
    "selector": False,
    "trade": False,
}
PREDECESSOR_BINDING_V1_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v1"
PREDECESSOR_BINDING_V2_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v2"
PREDECESSOR_BINDING_V3_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v3"
PREDECESSOR_BINDING_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v4"
FACTOR_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-diagnostic.v1"
FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-lifecycle-diagnostic.v1"
FACTOR_REGIME_ORIGIN_INVENTORY_V1_VERSION: Final = (
    "myquant.v17.v5.factor-regime-origin-inventory.v1"
)
FACTOR_REGIME_ORIGIN_INVENTORY_V2_VERSION: Final = (
    "myquant.v17.v5.factor-regime-origin-inventory.v2"
)
FACTOR_REGIME_ORIGIN_INVENTORY_VERSION: Final = "myquant.v17.v5.factor-regime-origin-inventory.v3"
REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V1_VERSION: Final = (
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v1"
)
REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V2_VERSION: Final = (
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v2"
)
REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION: Final = (
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v3"
)
REGIME_CONDITIONING_STATES: Final = frozenset({"趋势上涨", "震荡低波", "震荡高波", "趋势下跌"})
FACTOR_DIAGNOSTIC_POLICY_ID: Final = "v17.v5.factor.diagnostic.policy.sprint1a"
FACTOR_DIAGNOSTIC_POLICY_VERSION: Final = "myquant.v17.v5.factor-diagnostic-policy.v1"
FACTOR_DIAGNOSTIC_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/factor_diagnostic_policy.v1.json"
)
FACTOR_DIAGNOSTIC_POLICY_BYTE_SHA256: Final = (
    "6b0af1537108e99d496e26b314da3d6fe6eb19b862805d0c7cda0c682b057128"
)
FACTOR_DIAGNOSTIC_POLICY_SEMANTIC_SHA256: Final = (
    "f2c763f68d64381fbdbc11db0e57a25075bb9da4eb6968d168261318fabe68c4"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_ID: Final = "v17.v5.factor.regime.diagnostic.policy.sprint1b"
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_VERSION: Final = (
    "myquant.v17.v5.factor-regime-diagnostic-policy.v1"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/" "factor_regime_diagnostic_policy.v1.json"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_BYTE_SHA256: Final = (
    "4884be00e368cbd92a7edb7f947ce98b4fe04bee9555e721da330ff6a31cd607"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_SEMANTIC_SHA256: Final = (
    "6e406c5c7b66be9f85550c9a5931b866452a73762fc5b46b7505ba8473082ca6"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_ID: Final = "v17.v5.factor.regime.diagnostic.policy.sprint1d"
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION: Final = (
    "myquant.v17.v5.factor-regime-diagnostic-policy.v2"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/" "factor_regime_diagnostic_policy.v2.json"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_BYTE_SHA256: Final = (
    "10d87fe085caa69f9ecac80fd5a069e449b5f59509a088bb24727ad473e797c1"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_SEMANTIC_SHA256: Final = (
    "ac2e13a79f06e7e172b3263b0271ff7de179e9ac2b32f96d08cca39dd1d11d60"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_ID: Final = "v17.v5.factor.regime.diagnostic.policy.sprint1e0b"
FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION: Final = "myquant.v17.v5.factor-regime-diagnostic-policy.v3"
FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/" "factor_regime_diagnostic_policy.v3.json"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256: Final = (
    "8e78febb36c40e751851bf494061d8ee6baf96519a1baad872425a196064ab03"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256: Final = (
    "0da838ccf64afc1a7f5c71683b7c85d022a10cae2745f9d5590d987386ec3d50"
)
V4_COMPATIBILITY_POLICY_V1_ID: Final = "v17.v4.compatibility.policy.sprint1a"
V4_COMPATIBILITY_POLICY_V1_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v1"
V4_COMPATIBILITY_POLICY_V1_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v1.json"
)
V4_COMPATIBILITY_POLICY_V1_BYTE_SHA256: Final = (
    "bfb29a67fcee1e440ebc70d9d7299b28636cbcf7d38b6a88d0a5d720ec8a95ca"
)
V4_COMPATIBILITY_POLICY_V1_SEMANTIC_SHA256: Final = (
    "73439952d7844949694df4c1259db70dd46b0ed870700c98ec9aee088db47c53"
)
V4_COMPATIBILITY_POLICY_V2_ID: Final = "v17.v4.compatibility.policy.sprint1d"
V4_COMPATIBILITY_POLICY_V2_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v2"
V4_COMPATIBILITY_POLICY_V2_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v2.json"
)
V4_COMPATIBILITY_POLICY_V2_BYTE_SHA256: Final = (
    "0c0c4ccd5030e54f4e8cabc2742b7510be294c5b536a582cd59ab57635118190"
)
V4_COMPATIBILITY_POLICY_V2_SEMANTIC_SHA256: Final = (
    "0581fbd43bb77d63362a60b12734e3acd43e5aeac00e072a4be9b0681a077995"
)
V4_COMPATIBILITY_POLICY_V3_ID: Final = "v17.v4.compatibility.policy.sprint1e0b"
V4_COMPATIBILITY_POLICY_V3_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v3"
V4_COMPATIBILITY_POLICY_V3_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v3.json"
)
V4_COMPATIBILITY_POLICY_V3_BYTE_SHA256: Final = (
    "c61b3bc188d3dc8b23f531855a0399b5523cade4eea62d12034cb0ae68f7637f"
)
V4_COMPATIBILITY_POLICY_V3_SEMANTIC_SHA256: Final = (
    "bd8b77337eb90e9310792bdd4dbd28f6c8d0623a804c76ec74fd50084efca966"
)
V4_COMPATIBILITY_POLICY_ID: Final = "v17.v4.compatibility.policy.release-rc-1"
V4_COMPATIBILITY_POLICY_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v4"
V4_COMPATIBILITY_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v4.json"
)
V4_COMPATIBILITY_POLICY_BYTE_SHA256: Final = (
    "39b506d3950f2b8f36b422752f5317373fb7a221ae1950bb971d92e10b7342ca"
)
V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256: Final = (
    "46835a7582e0ecd44622cef4487955365ac5723dba37fb14c756ed88558cfb40"
)
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_ID: Final = "v17.v5.v4.factor.evidence.adapter.policy.sprint1a"
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_VERSION: Final = (
    "myquant.v17.v5.v4-factor-evidence-adapter-policy.v1"
)
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/" "v4_factor_evidence_adapter_policy.v1.json"
)
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256: Final = (
    "cf6d01b9db09a4ba7924c85675cb41ba0a8720689f212f13e34986c4ad9c3188"
)
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_SEMANTIC_SHA256: Final = (
    "463cd280d80ebb7914bca720d1b585380638b59863a48bec7a4f5615cdf8e225"
)
V4_SOURCE_GIT_COMMIT: Final = "6a2fa23dec68d87eb686464a86d8ba8997416310"
V4_PACKAGE_MANIFEST_SHA256: Final = (
    "a603b5f3e5f012548e3c3a224ba32ffc62b072d6555849887369f48f45012449"
)
V4_RUNTIME_MANIFEST_SHA256: Final = (
    "9f3e6ebc2bc9283b5d81113630d2dad68eef6bec0eddd0fcd28077a5153edfbe"
)
V4_V3_SOURCE_GIT_COMMIT: Final = "73c5b6eea6c60d9a31865e176646687ffeee9d6a"
V4_V3_PACKAGE_MANIFEST_SHA256: Final = (
    "270c863fdcc2b092265444db9cc2fac9e3e19e1ef5fb2a36ddde6b47e443a1ff"
)
V4_V3_RUNTIME_MANIFEST_SHA256: Final = (
    "7c7dc183a419623542fb1d8b95d092283c948c46a804eedd8424f931645f3a28"
)
V4_REGIME_EVIDENCE_V3_SCHEMA_SHA256: Final = (
    "429c9ed6f664ae70f0a34d92e0a94bc10293291217d58eb22f2fb2e36e83ab80"
)
V4_REGIME_INFERENCE_POLICY_V2_SHA256: Final = (
    "46733a14377476c43ed230f9167dd786795c9b01159755cf91f358d07d44a3c1"
)
V4_REGIME_EVIDENCE_V3_RUNTIME_SHA256: Final = (
    "b9819326d32df1f094ecc5954f3664c36f060d9e5e3044adaaf17c4abb8b4180"
)
V4_V2_PUBLICATION_BLOCK_CLI_SHA256: Final = (
    "fc185ce2a6cd214e5ef1f2e9c8e8fc19e17a8d8cebd9c175a85a132001e8980f"
)
V4_V3_PUBLICATION_BLOCK_CLI_SHA256: Final = (
    "015f0a05e03ae3864d8f8935f7260a42aa01531f9dd133bef1527d69a5adadc3"
)


class ArtifactContractError(ValueError):
    """Raised when a schema-valid V17 v5 artifact violates semantics."""

    exit_code = 2


def validate_v3_excluded_regime_origin_row(row: Mapping[str, Any]) -> None:
    """Require every excluded V3 origin to carry its exact exclusion facts."""

    continuity = row.get("regime_continuity_kind")
    regime_state = row.get("regime_state")
    observed_codes = row.get("row_limitation_codes")
    expected_codes: list[str] = []
    if continuity == "GENESIS":
        expected_codes.append("REGIME_CONTINUITY_GENESIS")
    elif continuity == "RECOVERY":
        expected_codes.append("REGIME_CONTINUITY_RECOVERY")
    elif continuity not in {"CONTIGUOUS", "ROLLOVER"}:
        raise ArtifactContractError("factor regime v3 excluded continuity mismatch")
    if regime_state == "未知":
        expected_codes.append("REGIME_HARD_STATE_UNKNOWN")
    elif regime_state not in REGIME_CONDITIONING_STATES:
        raise ArtifactContractError("factor regime v3 excluded state mismatch")
    if (
        row.get("regime_finalized") is not True
        or not expected_codes
        or observed_codes != sorted(expected_codes)
    ):
        raise ArtifactContractError("factor regime v3 excluded origin mismatch")


def _fixed_decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str:
        raise ArtifactContractError(f"{label} must be a fixed decimal string")
    number = Decimal(value)
    if (
        not number.is_finite()
        or number < Decimal("-1")
        or number > Decimal("1")
        or (number.is_zero() and value.startswith("-"))
        or format(number, ".12f") != value
    ):
        raise ArtifactContractError(f"{label} is not a canonical fixed decimal")
    return number


def _render_decimal(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _rank_ic_statistics(values: list[Decimal]) -> dict[str, str] | None:
    if not values:
        return None
    ordered = sorted(values)
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        count = Decimal(len(values))
        mean = sum(values, Decimal(0)) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        if len(ordered) % 2:
            median = ordered[len(ordered) // 2]
        else:
            middle = len(ordered) // 2
            median = (ordered[middle - 1] + ordered[middle]) / Decimal(2)
        stddev = variance.sqrt()
    return {
        "rank_ic_max": _render_decimal(max(ordered)),
        "rank_ic_mean": _render_decimal(mean),
        "rank_ic_median": _render_decimal(median),
        "rank_ic_min": _render_decimal(min(ordered)),
        "rank_ic_population_stddev": _render_decimal(stddev),
    }


def _validate_timestamp(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{label} is not a valid UTC timestamp") from exc
    return parsed


def _factor_regime_policy_ref(*, version: str) -> dict[str, str]:
    if version in {
        FACTOR_REGIME_ORIGIN_INVENTORY_V1_VERSION,
        REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V1_VERSION,
    }:
        return {
            "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_ID,
            "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_BYTE_SHA256,
            "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH,
            "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_SEMANTIC_SHA256,
            "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_VERSION,
        }
    if version in {
        FACTOR_REGIME_ORIGIN_INVENTORY_V2_VERSION,
        REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V2_VERSION,
    }:
        return {
            "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_ID,
            "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_BYTE_SHA256,
            "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH,
            "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_SEMANTIC_SHA256,
            "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION,
        }
    return {
        "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    }


def _validate_factor_regime_origin_inventory(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(document["inventory_id"], label="inventory_id")
        require_identifier(document["strategy_id"], label="strategy_id")
        require_identifier(document["factor_name"], label="factor_name")
        require_sha256(
            document["factor_implementation_sha256"],
            label="factor_implementation_sha256",
        )
        cutoff = _validate_timestamp(document["cutoff"], label="cutoff")
        created_at = _validate_timestamp(document["created_at"], label="created_at")
    except (
        CanonicalContractError,
        IdentityContractError,
        KeyError,
        TypeError,
    ) as exc:
        raise ArtifactContractError("factor regime origin inventory is invalid") from exc
    if (
        document["authority"] != NO_AUTHORITY
        or document["protocol_version"] != "myquant.v17.v5"
        or document["horizon_sessions"] != 20
        or document["policy_ref"] != _factor_regime_policy_ref(version=document["version"])
        or created_at < cutoff
    ):
        raise ArtifactContractError("factor regime origin inventory contract mismatch")
    rows = document["origin_rows"]
    is_v2 = document["version"] == FACTOR_REGIME_ORIGIN_INVENTORY_V2_VERSION
    is_v3 = document["version"] == FACTOR_REGIME_ORIGIN_INVENTORY_VERSION
    is_modern = is_v2 or is_v3
    if document["origin_count"] != len(rows):
        raise ArtifactContractError("factor regime origin count mismatch")
    excluded_rows = document.get("excluded_origin_rows", [])
    if is_modern and (document.get("excluded_origin_count") != len(excluded_rows)):
        raise ArtifactContractError("factor regime excluded origin count mismatch")
    if is_v2 and document.get("limitation_codes") != (
        ["REGIME_HARD_STATE_UNKNOWN"] if excluded_rows else []
    ):
        raise ArtifactContractError("factor regime v2 limitation code mismatch")
    if is_v3 and document.get("limitation_codes") != sorted(
        {code for row in excluded_rows for code in row.get("row_limitation_codes", [])}
    ):
        raise ArtifactContractError("factor regime v3 limitation code mismatch")
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["factor_observation_ref"]["artifact_id"],
            row["matured_label_ref"]["artifact_id"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    seen: set[tuple[str, str]] = set()
    seen_origin_ids: set[str] = set()
    counts: dict[str, int] = {}
    for row in [*rows, *excluded_rows]:
        key = (row["decision_session"], row["factor_name"])
        if key in seen or row["origin_id"] in seen_origin_ids:
            raise ArtifactContractError("duplicate factor regime origin")
        seen.add(key)
        seen_origin_ids.add(row["origin_id"])
        expected_row_identity = dict(row)
        observed_row_identity = expected_row_identity.pop("row_identity_sha256")
        if (
            hashlib.sha256(canonical_bytes(expected_row_identity)).hexdigest()
            != observed_row_identity
        ):
            raise ArtifactContractError("factor regime origin row identity mismatch")
        if row in excluded_rows:
            if is_v2 and (
                row["regime_state"] != "未知"
                or row["row_limitation_codes"] != ["REGIME_HARD_STATE_UNKNOWN"]
            ):
                raise ArtifactContractError("factor regime excluded origin mismatch")
            if is_v3:
                validate_v3_excluded_regime_origin_row(row)
            continue
        eligible_count = row["eligible_symbol_count"]
        comparable_count = row["comparable_symbol_count"]
        coverage = Decimal(row["coverage"])
        if (eligible_count == 0 and (comparable_count != 0 or coverage != 0)) or (
            eligible_count > 0
            and (Decimal(comparable_count) / Decimal(eligible_count)).quantize(
                Decimal("0.000000000001"),
                rounding=ROUND_HALF_EVEN,
            )
            != coverage
        ):
            raise ArtifactContractError("factor regime origin coverage mismatch")
        source_refs = (
            row["factor_evidence_ref"],
            row["factor_observation_ref"],
            row["matured_label_ref"],
            row["observation_run_ref"],
            row["request_ref"],
            row["source_locator_ref"],
            row["regime_evidence_ref"],
        )
        if any(ref["strategy_id"] != document["strategy_id"] for ref in source_refs):
            raise ArtifactContractError("factor regime origin ref strategy mismatch")
        if row["regime_source_version"] != row["regime_evidence_ref"]["version"]:
            raise ArtifactContractError("factor regime source version mismatch")
        probabilities = row["state_probabilities"]
        if probabilities is not None and row["regime_state"] not in {
            probability["regime_state"] for probability in probabilities
        }:
            raise ArtifactContractError("factor regime posterior omits the sealed hard state")
        if (
            row["factor_name"] != document["factor_name"]
            or row["label_horizon_sessions"] != 20
            or _validate_timestamp(
                row["regime_available_at"],
                label="regime_available_at",
            )
            > _validate_timestamp(row["origin_cutoff"], label="origin_cutoff")
            or _validate_timestamp(
                row["regime_published_at"],
                label="regime_published_at",
            )
            > _validate_timestamp(row["origin_cutoff"], label="origin_cutoff")
            or (
                row["regime_decision_session"] is not None
                and row["regime_decision_session"] > row["decision_session"]
            )
            or (
                row["regime_effective_session"] is not None
                and row["regime_effective_session"] > row["decision_session"]
            )
        ):
            raise ArtifactContractError("factor regime origin causality mismatch")
        if is_v2 and (
            row["regime_decision_session"] != row["decision_session"]
            or row["regime_effective_session"] != row["decision_session"]
            or row["regime_observed_through_session"] >= row["decision_session"]
            or row["regime_publication_phase"] != "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
            or row["regime_inference_kind"] != "FILTERED_CAUSAL"
            or row["regime_smoothing_used"] is not False
            or row["regime_hard_state_derivation"] != "SEALED_ARGMAX_POLICY_V1"
            or row["regime_scope_kind"] != "FULL_MARKET"
            or row["regime_no_retroactive_causal_backfill"] is not True
            or row["regime_source_commit"] != V4_SOURCE_GIT_COMMIT
            or row["regime_source_version"] != "myquant.v17.v4.regime-evidence.v2"
        ):
            raise ArtifactContractError("factor regime v2 origin binding mismatch")
        if is_v3 and (
            row["regime_decision_session"] != row["decision_session"]
            or row["regime_effective_session"] != row["decision_session"]
            or row["regime_observed_through_session"] >= row["decision_session"]
            or row["regime_publication_phase"] != "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
            or row["regime_inference_kind"] != "FILTERED_CAUSAL"
            or row["regime_smoothing_used"] is not False
            or row["regime_hard_state_derivation"] != "SEALED_ARGMAX_POLICY_V1"
            or row["regime_scope_kind"] != "FULL_MARKET"
            or row["regime_no_retroactive_causal_backfill"] is not True
            or row["regime_source_commit"] != V4_SOURCE_GIT_COMMIT
            or row["regime_source_version"] != "myquant.v17.v4.regime-evidence.v3"
            or row["regime_continuity_kind"] not in {"CONTIGUOUS", "ROLLOVER"}
            or row["regime_finalized"] is not True
        ):
            raise ArtifactContractError("factor regime v3 origin binding mismatch")
        counts[row["regime_state"]] = counts.get(row["regime_state"], 0) + 1
    expected_excluded_order = sorted(
        excluded_rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    if (
        rows != expected_order
        or excluded_rows != expected_excluded_order
        or document["regime_counts"]
        != [
            {
                "origin_count": counts[key],
                "regime_state": key,
            }
            for key in sorted(counts)
        ]
    ):
        raise ArtifactContractError("factor regime origin ordering mismatch")
    identity_material = dict(document)
    identity_material.pop("inventory_id")
    identity_material.pop("semantic_sha256")
    expected_inventory_id = (
        "factor-regime-origin-inventory-"
        f"{hashlib.sha256(canonical_bytes(identity_material)).hexdigest()[:32]}"
    )
    if document["inventory_id"] != expected_inventory_id:
        raise ArtifactContractError("factor regime origin inventory identity mismatch")
    return document


def _validate_regime_conditioned_factor_diagnostic(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(document["diagnostic_id"], label="diagnostic_id")
        require_identifier(document["strategy_id"], label="strategy_id")
        require_identifier(document["factor_name"], label="factor_name")
        cutoff = _validate_timestamp(document["cutoff"], label="cutoff")
        created_at = _validate_timestamp(document["created_at"], label="created_at")
    except (
        CanonicalContractError,
        IdentityContractError,
        KeyError,
        TypeError,
    ) as exc:
        raise ArtifactContractError("regime-conditioned factor diagnostic is invalid") from exc
    if (
        document["authority"] != NO_AUTHORITY
        or document["protocol_version"] != "myquant.v17.v5"
        or document["horizon_sessions"] != 20
        or document["policy_ref"] != _factor_regime_policy_ref(version=document["version"])
        or created_at < cutoff
    ):
        raise ArtifactContractError("regime-conditioned factor diagnostic contract mismatch")
    identity_material = dict(document)
    identity_material.pop("diagnostic_id")
    identity_material.pop("semantic_sha256")
    expected_diagnostic_id = (
        "regime-conditioned-factor-diagnostic-"
        f"{hashlib.sha256(canonical_bytes(identity_material)).hexdigest()[:32]}"
    )
    if document["diagnostic_id"] != expected_diagnostic_id:
        raise ArtifactContractError("regime-conditioned factor diagnostic identity mismatch")
    status = document["status"]
    occupancy = document["regime_occupancy"]
    if occupancy["ambiguous_regime_count"] != 0:
        raise ArtifactContractError("ambiguous regime evidence cannot be diagnosed")
    if status == "UNAVAILABLE":
        if (
            document["factor_evidence_ref"] is not None
            or document["origin_inventory_ref"] is not None
            or document["by_regime"]
            or document["regime_source_refs"]
            or document["unconditional_metrics"] is not None
            or not document["limitation_codes"]
            or occupancy
            != {
                "ambiguous_regime_count": 0,
                "missing_regime_count": 0,
                "posterior_confidence_summary": None,
                "regime_concentration": None,
                "regime_origin_counts": [],
                "total_origin_count": 0,
            }
        ):
            raise ArtifactContractError("UNAVAILABLE regime diagnostic is inconsistent")
    else:
        regime_counts = occupancy["regime_origin_counts"]
        by_regime = document["by_regime"]
        if any(row["regime_state"] not in REGIME_CONDITIONING_STATES for row in regime_counts):
            raise ArtifactContractError("regime occupancy contains an ineligible state")
        if any(row["regime_state"] not in REGIME_CONDITIONING_STATES for row in by_regime):
            raise ArtifactContractError("by-regime diagnostic contains an ineligible state")
        if regime_counts != sorted(regime_counts, key=lambda row: row["regime_state"]):
            raise ArtifactContractError("regime occupancy counts are noncanonical")
        if len({row["regime_state"] for row in regime_counts}) != len(regime_counts):
            raise ArtifactContractError("regime occupancy contains duplicate states")
        if sum(row["origin_count"] for row in regime_counts) != occupancy["total_origin_count"]:
            raise ArtifactContractError("regime occupancy counts do not close")
        try:
            require_sha256(
                document["factor_implementation_sha256"],
                label="factor_implementation_sha256",
            )
        except IdentityContractError as exc:
            raise ArtifactContractError("observed regime diagnostic has no factor SHA") from exc
        if (
            document["factor_evidence_ref"] is None
            or document["origin_inventory_ref"] is None
            or (
                status == "UNOBSERVED"
                and (
                    occupancy["total_origin_count"] != 0
                    or document["unconditional_metrics"] is not None
                    or document["by_regime"]
                )
            )
            or (
                status == "ACCUMULATING"
                and (
                    occupancy["total_origin_count"] < 1
                    or document["unconditional_metrics"] is None
                    or not document["by_regime"]
                )
            )
        ):
            raise ArtifactContractError("observed regime diagnostic is inconsistent")
    return document


def _validate_session(value: Any, *, label: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{label} is not a valid session date") from exc
    if parsed.isoformat() != value:
        raise ArtifactContractError(f"{label} is not canonical")
    return value


def _validate_factor_diagnostic(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(document["diagnostic_id"], label="diagnostic_id")
        require_identifier(document["subject_factor_name"], label="subject_factor_name")
        _validate_timestamp(document["evaluation_cutoff"], label="evaluation_cutoff")
        policy_ref = document["policy_ref"]
        require_identifier(policy_ref["artifact_id"], label="policy artifact_id")
        require_sha256(policy_ref["byte_sha256"], label="policy byte SHA-256")
        require_sha256(policy_ref["semantic_sha256"], label="policy semantic SHA-256")
        require_relative_path(policy_ref["relative_path"], label="policy relative_path")
    except (
        CanonicalContractError,
        IdentityContractError,
        KeyError,
        TypeError,
    ) as exc:
        raise ArtifactContractError("V17 v5 factor diagnostic is invalid") from exc
    if document["authority"] != NO_AUTHORITY:
        raise ArtifactContractError("V17 v5 factor diagnostic grants authority")
    if policy_ref != {
        "artifact_id": FACTOR_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_DIAGNOSTIC_POLICY_VERSION,
    }:
        raise ArtifactContractError("factor diagnostic policy identity mismatch")
    blockers = document["blockers"]
    if blockers != sorted(set(blockers)):
        raise ArtifactContractError("factor diagnostic blockers are noncanonical")
    for blocker in blockers:
        try:
            require_identifier(blocker, label="factor diagnostic blocker")
        except IdentityContractError as exc:
            raise ArtifactContractError("factor diagnostic blocker is invalid") from exc
    status = document["status"]
    stratum = document["stratum"]
    stratum_sha = document["stratum_sha256"]
    rows = document["origin_diagnostics"]
    if status == "UNAVAILABLE":
        if (
            stratum is not None
            or stratum_sha is not None
            or rows
            or document["matured_origin_count"] != 0
            or document["rank_ic_available_origin_count"] != 0
            or document["minimum_comparable_symbol_count"] is not None
            or document["total_comparable_symbol_rows"] != 0
            or document["statistics"] is not None
            or document["descriptive_coverage_minimum_met"]
            or "inference_not_implemented" not in blockers
            or len(blockers) < 2
            or any(
                blocker
                in {
                    "descriptive_coverage_minimum_not_met",
                    "no_naturally_matured_origins",
                }
                for blocker in blockers
            )
        ):
            raise ArtifactContractError("UNAVAILABLE factor diagnostic is inconsistent")
    else:
        if type(stratum) is not dict or type(stratum_sha) is not str:
            raise ArtifactContractError("observed factor diagnostic has no exact stratum")
        try:
            require_identifier(stratum["strategy_id"], label="strategy_id")
            require_identifier(stratum["factor_name"], label="factor_name")
            for field in (
                "adapter_policy_byte_sha256",
                "factor_definition_sha256",
                "factor_implementation_sha256",
                "factor_set_sha256",
                "market_calendar_sha256",
                "quant_policy_sha256",
                "source_lineage_series_sha256",
            ):
                require_sha256(stratum[field], label=field)
            require_sha256(stratum_sha, label="stratum_sha256")
        except (IdentityContractError, KeyError, TypeError) as exc:
            raise ArtifactContractError("factor diagnostic stratum is invalid") from exc
        expected_stratum_sha = hashlib.sha256(canonical_bytes(stratum)).hexdigest()
        if (
            stratum_sha != expected_stratum_sha
            or stratum["horizon_sessions"] != 20
            or stratum["factor_name"] != document["subject_factor_name"]
            or stratum["adapter_policy_byte_sha256"]
            != V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256
        ):
            raise ArtifactContractError("factor diagnostic stratum identity mismatch")
    previous_key: tuple[str, str] | None = None
    origin_ids: set[str] = set()
    decision_sessions: set[str] = set()
    available_values: list[Decimal] = []
    comparable_counts: list[int] = []
    for index, row in enumerate(rows):
        try:
            origin_id = require_identifier(row["origin_id"], label="origin_id")
            require_sha256(
                row["evidence_lineage_sha256"],
                label="evidence_lineage_sha256",
            )
            decision_session = _validate_session(
                row["decision_session"],
                label="decision_session",
            )
            _validate_session(
                row["horizon_end_session"],
                label="horizon_end_session",
            )
            _validate_timestamp(
                row["label_available_at"],
                label="label_available_at",
            )
        except (IdentityContractError, KeyError, TypeError) as exc:
            raise ArtifactContractError(f"factor diagnostic origin row {index} is invalid") from exc
        key = (decision_session, origin_id)
        if (
            (previous_key is not None and key <= previous_key)
            or origin_id in origin_ids
            or decision_session in decision_sessions
        ):
            raise ArtifactContractError("factor diagnostic origins are noncanonical")
        previous_key = key
        origin_ids.add(origin_id)
        decision_sessions.add(decision_session)
        row_blockers = row["blockers"]
        if row_blockers != sorted(set(row_blockers)):
            raise ArtifactContractError("origin blockers are noncanonical")
        comparable_counts.append(row["comparable_symbol_count"])
        if row["rank_ic_status"] == "AVAILABLE":
            if row["rank_ic"] is None or row_blockers:
                raise ArtifactContractError("AVAILABLE origin diagnostic is inconsistent")
            available_values.append(_fixed_decimal(row["rank_ic"], label="rank_ic"))
        elif row["rank_ic"] is not None or row_blockers not in [
            ["constant_factor"],
            ["constant_return"],
            ["insufficient_comparable_symbols"],
        ]:
            raise ArtifactContractError("UNAVAILABLE origin diagnostic is inconsistent")
    if document["matured_origin_count"] != len(rows):
        raise ArtifactContractError("matured origin count mismatch")
    if document["rank_ic_available_origin_count"] != len(available_values):
        raise ArtifactContractError("available RankIC origin count mismatch")
    expected_minimum = min(comparable_counts) if comparable_counts else None
    if document["minimum_comparable_symbol_count"] != expected_minimum:
        raise ArtifactContractError("minimum comparable-symbol count mismatch")
    if document["total_comparable_symbol_rows"] != sum(comparable_counts):
        raise ArtifactContractError("total comparable-symbol row count mismatch")
    expected_coverage = len(available_values) >= 60 and all(
        row["comparable_symbol_count"] >= 100
        for row in rows
        if row["rank_ic_status"] == "AVAILABLE"
    )
    if document["descriptive_coverage_minimum_met"] != expected_coverage:
        raise ArtifactContractError("descriptive coverage minimum mismatch")
    if document["statistics"] != _rank_ic_statistics(available_values):
        raise ArtifactContractError("factor diagnostic statistics mismatch")
    if status == "UNOBSERVED":
        if rows or blockers != ["inference_not_implemented", "no_naturally_matured_origins"]:
            raise ArtifactContractError("UNOBSERVED factor diagnostic is inconsistent")
    elif status == "ACCUMULATING":
        expected_blockers = (
            ["inference_not_implemented"]
            if expected_coverage
            else [
                "descriptive_coverage_minimum_not_met",
                "inference_not_implemented",
            ]
        )
        if not rows or blockers != expected_blockers:
            raise ArtifactContractError("ACCUMULATING factor diagnostic is inconsistent")
    identity_material = dict(document)
    identity_material.pop("diagnostic_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if document["diagnostic_id"] != f"factor-diagnostic-{identity[:32]}":
        raise ArtifactContractError("factor diagnostic identity mismatch")
    return document


def _validate_factor_lifecycle_diagnostic(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(
            document["lifecycle_diagnostic_id"],
            label="lifecycle_diagnostic_id",
        )
        require_identifier(document["factor_name"], label="factor_name")
        _validate_timestamp(document["evaluation_cutoff"], label="evaluation_cutoff")
    except (
        CanonicalContractError,
        IdentityContractError,
        KeyError,
        TypeError,
    ) as exc:
        raise ArtifactContractError("V17 v5 factor lifecycle diagnostic is invalid") from exc
    if document["authority"] != NO_AUTHORITY:
        raise ArtifactContractError("V17 v5 factor lifecycle diagnostic grants authority")
    for field in (
        "effectiveness_claimed",
        "factor_tier_change_eligible",
        "factor_weight_change_eligible",
        "promotion_eligible",
    ):
        if document[field] is not False:
            raise ArtifactContractError(f"{field} must remain false")
    if document["lifecycle_action"] is not None or document["lifecycle_conclusion"] is not None:
        raise ArtifactContractError("factor lifecycle conclusions are forbidden")
    blockers = document["blockers"]
    if blockers != sorted(set(blockers)):
        raise ArtifactContractError("factor lifecycle blockers are noncanonical")
    for blocker in blockers:
        try:
            require_identifier(blocker, label="factor lifecycle blocker")
        except IdentityContractError as exc:
            raise ArtifactContractError("factor lifecycle blocker is invalid") from exc
    input_shas = document["input_factor_diagnostic_semantic_sha256s"]
    if input_shas != sorted(set(input_shas)):
        raise ArtifactContractError("factor lifecycle input SHA list is noncanonical")
    for value in input_shas:
        try:
            require_sha256(value, label="input factor diagnostic semantic SHA-256")
        except IdentityContractError as exc:
            raise ArtifactContractError("factor lifecycle input SHA is invalid") from exc
    status = document["status"]
    stratum = document["stratum"]
    stratum_sha = document["stratum_sha256"]
    count = document["unique_origin_count"]
    first_session = document["first_decision_session"]
    last_session = document["last_decision_session"]
    if status == "UNAVAILABLE":
        if (
            stratum is not None
            or stratum_sha is not None
            or count != 0
            or first_session is not None
            or last_session is not None
            or document["descriptive_coverage_minimum_met"]
            or not blockers
            or input_shas
            or "lifecycle_inputs_unavailable" not in blockers
        ):
            raise ArtifactContractError("UNAVAILABLE lifecycle diagnostic is inconsistent")
    else:
        if type(stratum) is not dict or type(stratum_sha) is not str:
            raise ArtifactContractError("observed lifecycle diagnostic has no stratum")
        try:
            require_sha256(stratum_sha, label="stratum_sha256")
        except IdentityContractError as exc:
            raise ArtifactContractError("lifecycle stratum SHA is invalid") from exc
        if (
            hashlib.sha256(canonical_bytes(stratum)).hexdigest() != stratum_sha
            or stratum["factor_name"] != document["factor_name"]
            or stratum["adapter_policy_byte_sha256"]
            != V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256
        ):
            raise ArtifactContractError("lifecycle stratum identity mismatch")
        if status == "UNOBSERVED":
            if (
                count != 0
                or first_session is not None
                or last_session is not None
                or input_shas == []
                or document["descriptive_coverage_minimum_met"]
                or "lifecycle_no_observed_origins" not in blockers
            ):
                raise ArtifactContractError("UNOBSERVED lifecycle diagnostic is inconsistent")
        elif status == "ACCUMULATING":
            if (
                count <= 0
                or first_session is None
                or last_session is None
                or "lifecycle_diagnostic_only" not in blockers
            ):
                raise ArtifactContractError("ACCUMULATING lifecycle diagnostic is inconsistent")
            first = _validate_session(first_session, label="first_decision_session")
            last = _validate_session(last_session, label="last_decision_session")
            if first > last or not input_shas:
                raise ArtifactContractError("lifecycle session bounds are invalid")
        else:
            raise ArtifactContractError("unknown factor lifecycle diagnostic status")
    identity_material = dict(document)
    identity_material.pop("lifecycle_diagnostic_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if document["lifecycle_diagnostic_id"] != f"factor-lifecycle-diagnostic-{identity[:32]}":
        raise ArtifactContractError("factor lifecycle diagnostic identity mismatch")
    return document


def _validate_predecessor_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        document = validate_semantic_sha(payload)
        require_identifier(document["binding_id"], label="binding_id")
        require_git_commit(document["source_git_commit"])
        require_sha256(document["source_package_manifest_byte_sha256"])
        require_sha256(document["source_runtime_manifest_byte_sha256"])
        require_relative_path(document["source_package_manifest_relative_path"])
        require_relative_path(document["source_runtime_manifest_relative_path"])
        policy = document["compatibility_policy_ref"]
        require_identifier(policy["artifact_id"], label="compatibility policy artifact_id")
        require_sha256(policy["byte_sha256"], label="compatibility policy byte SHA-256")
        require_sha256(policy["semantic_sha256"], label="compatibility policy semantic SHA-256")
        require_relative_path(policy["relative_path"], label="compatibility policy path")
    except (CanonicalContractError, IdentityContractError, KeyError, TypeError) as exc:
        raise ArtifactContractError("V17 v4 predecessor binding is invalid") from exc
    if document["authority"] != NO_AUTHORITY:
        raise ArtifactContractError("V17 v5 predecessor binding grants authority")
    if document["version"] == PREDECESSOR_BINDING_V1_VERSION:
        expected_policy_ref = {
            "artifact_id": V4_COMPATIBILITY_POLICY_V1_ID,
            "byte_sha256": V4_COMPATIBILITY_POLICY_V1_BYTE_SHA256,
            "relative_path": V4_COMPATIBILITY_POLICY_V1_PATH,
            "semantic_sha256": V4_COMPATIBILITY_POLICY_V1_SEMANTIC_SHA256,
            "version": V4_COMPATIBILITY_POLICY_V1_VERSION,
        }
    elif document["version"] == PREDECESSOR_BINDING_V2_VERSION:
        expected_policy_ref = {
            "artifact_id": V4_COMPATIBILITY_POLICY_V2_ID,
            "byte_sha256": V4_COMPATIBILITY_POLICY_V2_BYTE_SHA256,
            "relative_path": V4_COMPATIBILITY_POLICY_V2_PATH,
            "semantic_sha256": V4_COMPATIBILITY_POLICY_V2_SEMANTIC_SHA256,
            "version": V4_COMPATIBILITY_POLICY_V2_VERSION,
        }
    elif document["version"] == PREDECESSOR_BINDING_V3_VERSION:
        expected_policy_ref = {
            "artifact_id": V4_COMPATIBILITY_POLICY_V3_ID,
            "byte_sha256": V4_COMPATIBILITY_POLICY_V3_BYTE_SHA256,
            "relative_path": V4_COMPATIBILITY_POLICY_V3_PATH,
            "semantic_sha256": V4_COMPATIBILITY_POLICY_V3_SEMANTIC_SHA256,
            "version": V4_COMPATIBILITY_POLICY_V3_VERSION,
        }
    else:
        expected_policy_ref = {
            "artifact_id": V4_COMPATIBILITY_POLICY_ID,
            "byte_sha256": V4_COMPATIBILITY_POLICY_BYTE_SHA256,
            "relative_path": V4_COMPATIBILITY_POLICY_PATH,
            "semantic_sha256": V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256,
            "version": V4_COMPATIBILITY_POLICY_VERSION,
        }
    if policy != expected_policy_ref:
        raise ArtifactContractError("V17 v4 compatibility policy identity mismatch")
    if document["version"] == PREDECESSOR_BINDING_V1_VERSION:
        expected_source = {
            "source_git_commit": "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e",
            "source_package_manifest_byte_sha256": (
                "fdc0aba035cdfff243df1a191431c84cfd7638fd0d94d877c7b37b29d5bc6875"
            ),
            "source_runtime_manifest_byte_sha256": (
                "09700937c1fac82b2e3bbd405f1cbe7d31e71faea6a6c71e2d57d0c8c2b87b04"
            ),
        }
    elif document["version"] == PREDECESSOR_BINDING_V2_VERSION:
        expected_source = {
            "source_git_commit": "1da7ffb636a3254940525d746549d15e827f06ba",
            "source_package_manifest_byte_sha256": (
                "80dd615730ccf94eb453664936b0f265180dc68c18651e90932ce05fa3fb1428"
            ),
            "source_runtime_manifest_byte_sha256": (
                "a7d27d0d16153d5b55558cd608a9155dd3b968d2721135ba77d777d409a7e63c"
            ),
        }
    elif document["version"] == PREDECESSOR_BINDING_V3_VERSION:
        expected_source = {
            "source_git_commit": V4_V3_SOURCE_GIT_COMMIT,
            "source_package_manifest_byte_sha256": V4_V3_PACKAGE_MANIFEST_SHA256,
            "source_runtime_manifest_byte_sha256": V4_V3_RUNTIME_MANIFEST_SHA256,
        }
    else:
        expected_source = {
            "source_git_commit": V4_SOURCE_GIT_COMMIT,
            "source_package_manifest_byte_sha256": V4_PACKAGE_MANIFEST_SHA256,
            "source_runtime_manifest_byte_sha256": V4_RUNTIME_MANIFEST_SHA256,
        }
    if (
        document["protocol_version"] != "myquant.v17.v5"
        or document["source_protocol_version"] != "myquant.v17.v4"
        or document["source_git_commit"] != expected_source["source_git_commit"]
        or document["source_package_manifest_byte_sha256"]
        != expected_source["source_package_manifest_byte_sha256"]
        or document["source_runtime_manifest_byte_sha256"]
        != expected_source["source_runtime_manifest_byte_sha256"]
    ):
        raise ArtifactContractError("V17 v4 predecessor binding identity mismatch")
    if document["version"] in {PREDECESSOR_BINDING_V3_VERSION, PREDECESSOR_BINDING_VERSION}:
        try:
            for field in (
                "regime_evidence_v3_runtime_sha256",
                "regime_evidence_v3_schema_sha256",
                "regime_inference_policy_v2_sha256",
                "v2_cli_source_sha256",
            ):
                require_sha256(document[field], label=field)
        except (IdentityContractError, KeyError, TypeError) as exc:
            raise ArtifactContractError("V17 v4 predecessor bounded binding is invalid") from exc
        is_v3 = document["version"] == PREDECESSOR_BINDING_V3_VERSION
        if (
            document["source_package_asset_count"] != (109 if is_v3 else 114)
            or document["source_runtime_source_count"] != (32 if is_v3 else 34)
            or document["regime_evidence_v3_schema_sha256"] != V4_REGIME_EVIDENCE_V3_SCHEMA_SHA256
            or document["regime_inference_policy_v2_sha256"] != V4_REGIME_INFERENCE_POLICY_V2_SHA256
            or document["regime_evidence_v3_runtime_sha256"] != V4_REGIME_EVIDENCE_V3_RUNTIME_SHA256
            or document["v2_cli_source_sha256"]
            != (V4_V3_PUBLICATION_BLOCK_CLI_SHA256 if is_v3 else V4_V2_PUBLICATION_BLOCK_CLI_SHA256)
            or document["v2_publication_status"] != "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE"
        ):
            raise ArtifactContractError("V17 v4 predecessor bounded metadata mismatch")
    return document


def validate_typed_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> dict[str, Any]:
    if not schema_checked:
        from .schema_validation import validate_schema_version

        validate_schema_version(payload, payload.get("version"))
    if payload.get("version") in {
        PREDECESSOR_BINDING_V1_VERSION,
        PREDECESSOR_BINDING_V2_VERSION,
        PREDECESSOR_BINDING_V3_VERSION,
        PREDECESSOR_BINDING_VERSION,
    }:
        return _validate_predecessor_binding(payload)
    if payload.get("version") == FACTOR_DIAGNOSTIC_VERSION:
        return _validate_factor_diagnostic(payload)
    if payload.get("version") == FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION:
        return _validate_factor_lifecycle_diagnostic(payload)
    if payload.get("version") in {
        FACTOR_REGIME_ORIGIN_INVENTORY_V1_VERSION,
        FACTOR_REGIME_ORIGIN_INVENTORY_V2_VERSION,
        FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    }:
        return _validate_factor_regime_origin_inventory(payload)
    if payload.get("version") in {
        REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V1_VERSION,
        REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V2_VERSION,
        REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION,
    }:
        return _validate_regime_conditioned_factor_diagnostic(payload)
    raise ArtifactContractError("unsupported V17 v5 artifact version")


__all__ = [
    "ArtifactContractError",
    "FACTOR_DIAGNOSTIC_POLICY_BYTE_SHA256",
    "FACTOR_DIAGNOSTIC_POLICY_ID",
    "FACTOR_DIAGNOSTIC_POLICY_PATH",
    "FACTOR_DIAGNOSTIC_POLICY_SEMANTIC_SHA256",
    "FACTOR_DIAGNOSTIC_POLICY_VERSION",
    "FACTOR_DIAGNOSTIC_VERSION",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_ID",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_BYTE_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_ID",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_SEMANTIC_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_VERSION",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_BYTE_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_ID",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_SEMANTIC_SHA256",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION",
    "FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION",
    "FACTOR_REGIME_ORIGIN_INVENTORY_V1_VERSION",
    "FACTOR_REGIME_ORIGIN_INVENTORY_V2_VERSION",
    "FACTOR_REGIME_ORIGIN_INVENTORY_VERSION",
    "NO_AUTHORITY",
    "PREDECESSOR_BINDING_V1_VERSION",
    "PREDECESSOR_BINDING_V2_VERSION",
    "PREDECESSOR_BINDING_V3_VERSION",
    "PREDECESSOR_BINDING_VERSION",
    "REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V1_VERSION",
    "REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_V2_VERSION",
    "REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION",
    "V4_COMPATIBILITY_POLICY_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_ID",
    "V4_COMPATIBILITY_POLICY_PATH",
    "V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256",
    "V4_COMPATIBILITY_POLICY_V1_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_V1_ID",
    "V4_COMPATIBILITY_POLICY_V1_PATH",
    "V4_COMPATIBILITY_POLICY_V1_SEMANTIC_SHA256",
    "V4_COMPATIBILITY_POLICY_V1_VERSION",
    "V4_COMPATIBILITY_POLICY_V2_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_V2_ID",
    "V4_COMPATIBILITY_POLICY_V2_PATH",
    "V4_COMPATIBILITY_POLICY_V2_SEMANTIC_SHA256",
    "V4_COMPATIBILITY_POLICY_V2_VERSION",
    "V4_COMPATIBILITY_POLICY_V3_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_V3_ID",
    "V4_COMPATIBILITY_POLICY_V3_PATH",
    "V4_COMPATIBILITY_POLICY_V3_SEMANTIC_SHA256",
    "V4_COMPATIBILITY_POLICY_V3_VERSION",
    "V4_COMPATIBILITY_POLICY_VERSION",
    "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256",
    "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_ID",
    "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH",
    "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_SEMANTIC_SHA256",
    "V4_FACTOR_EVIDENCE_ADAPTER_POLICY_VERSION",
    "V4_PACKAGE_MANIFEST_SHA256",
    "V4_RUNTIME_MANIFEST_SHA256",
    "V4_SOURCE_GIT_COMMIT",
    "validate_typed_artifact",
    "validate_v3_excluded_regime_origin_row",
]
