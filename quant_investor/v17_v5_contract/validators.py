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
PREDECESSOR_BINDING_VERSION: Final = "myquant.v17.v5.v4-predecessor-binding.v1"
FACTOR_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-diagnostic.v1"
FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-lifecycle-diagnostic.v1"
FACTOR_REGIME_ORIGIN_INVENTORY_VERSION: Final = "myquant.v17.v5.factor-regime-origin-inventory.v1"
REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION: Final = (
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v1"
)
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
FACTOR_REGIME_DIAGNOSTIC_POLICY_ID: Final = "v17.v5.factor.regime.diagnostic.policy.sprint1b"
FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION: Final = "myquant.v17.v5.factor-regime-diagnostic-policy.v1"
FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/" "factor_regime_diagnostic_policy.v1.json"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256: Final = (
    "4884be00e368cbd92a7edb7f947ce98b4fe04bee9555e721da330ff6a31cd607"
)
FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256: Final = (
    "6e406c5c7b66be9f85550c9a5931b866452a73762fc5b46b7505ba8473082ca6"
)
V4_COMPATIBILITY_POLICY_ID: Final = "v17.v4.compatibility.policy.sprint1a"
V4_COMPATIBILITY_POLICY_VERSION: Final = "myquant.v17.v5.v4-compatibility-policy.v1"
V4_COMPATIBILITY_POLICY_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/v4_compatibility_policy.v1.json"
)
V4_COMPATIBILITY_POLICY_BYTE_SHA256: Final = (
    "bfb29a67fcee1e440ebc70d9d7299b28636cbcf7d38b6a88d0a5d720ec8a95ca"
)
V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256: Final = (
    "73439952d7844949694df4c1259db70dd46b0ed870700c98ec9aee088db47c53"
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
V4_SOURCE_GIT_COMMIT: Final = "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e"
V4_PACKAGE_MANIFEST_SHA256: Final = (
    "fdc0aba035cdfff243df1a191431c84cfd7638fd0d94d877c7b37b29d5bc6875"
)
V4_RUNTIME_MANIFEST_SHA256: Final = (
    "09700937c1fac82b2e3bbd405f1cbe7d31e71faea6a6c71e2d57d0c8c2b87b04"
)


class ArtifactContractError(ValueError):
    """Raised when a schema-valid V17 v5 artifact violates semantics."""

    exit_code = 2


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


def _factor_regime_policy_ref() -> dict[str, str]:
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
        or document["policy_ref"] != _factor_regime_policy_ref()
        or created_at < cutoff
    ):
        raise ArtifactContractError("factor regime origin inventory contract mismatch")
    rows = document["origin_rows"]
    if document["origin_count"] != len(rows):
        raise ArtifactContractError("factor regime origin count mismatch")
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
    for row in rows:
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
        counts[row["regime_state"]] = counts.get(row["regime_state"], 0) + 1
    if rows != expected_order or document["regime_counts"] != [
        {
            "origin_count": counts[key],
            "regime_state": key,
        }
        for key in sorted(counts)
    ]:
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
        or document["policy_ref"] != _factor_regime_policy_ref()
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
    if policy != {
        "artifact_id": V4_COMPATIBILITY_POLICY_ID,
        "byte_sha256": V4_COMPATIBILITY_POLICY_BYTE_SHA256,
        "relative_path": V4_COMPATIBILITY_POLICY_PATH,
        "semantic_sha256": V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256,
        "version": V4_COMPATIBILITY_POLICY_VERSION,
    }:
        raise ArtifactContractError("V17 v4 compatibility policy identity mismatch")
    if (
        document["protocol_version"] != "myquant.v17.v5"
        or document["source_protocol_version"] != "myquant.v17.v4"
        or document["source_git_commit"] != V4_SOURCE_GIT_COMMIT
        or document["source_package_manifest_byte_sha256"] != V4_PACKAGE_MANIFEST_SHA256
        or document["source_runtime_manifest_byte_sha256"] != V4_RUNTIME_MANIFEST_SHA256
    ):
        raise ArtifactContractError("V17 v4 predecessor binding identity mismatch")
    return document


def validate_typed_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> dict[str, Any]:
    if not schema_checked:
        from .schema_validation import validate_schema_version

        validate_schema_version(payload, payload.get("version"))
    if payload.get("version") == PREDECESSOR_BINDING_VERSION:
        return _validate_predecessor_binding(payload)
    if payload.get("version") == FACTOR_DIAGNOSTIC_VERSION:
        return _validate_factor_diagnostic(payload)
    if payload.get("version") == FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION:
        return _validate_factor_lifecycle_diagnostic(payload)
    if payload.get("version") == FACTOR_REGIME_ORIGIN_INVENTORY_VERSION:
        return _validate_factor_regime_origin_inventory(payload)
    if payload.get("version") == REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION:
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
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION",
    "FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION",
    "FACTOR_REGIME_ORIGIN_INVENTORY_VERSION",
    "NO_AUTHORITY",
    "PREDECESSOR_BINDING_VERSION",
    "REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION",
    "V4_COMPATIBILITY_POLICY_BYTE_SHA256",
    "V4_COMPATIBILITY_POLICY_ID",
    "V4_COMPATIBILITY_POLICY_PATH",
    "V4_COMPATIBILITY_POLICY_SEMANTIC_SHA256",
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
]
