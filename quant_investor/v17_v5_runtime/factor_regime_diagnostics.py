"""Regime-conditioned descriptive factor diagnostics for V17 v5 Sprint 1B.

The diagnostic unit is one already verified factor-origin RankIC observation
bound to the sealed regime state available at the origin decision point.  This
module never reads V4 data, scans latest pointers, infers origin observations
from aggregates, emits weights, or creates governance actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import math
import re
from typing import Any, Final, Mapping, Sequence

from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_semantic_sha,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.validators import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    NO_AUTHORITY,
    REGIME_CONDITIONING_STATES,
)
from quant_investor.v17_v5_runtime.factor_regime_origin_inventory import (
    FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
)

PROTOCOL_VERSION: Final = "myquant.v17.v5"
REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION: Final = (
    "myquant.v17.v5.regime-conditioned-factor-diagnostic.v3"
)
HORIZON_SESSIONS: Final = 20
MINIMUM_DESCRIPTIVE_ORIGINS: Final = 20
MINIMUM_STABILITY_ORIGINS: Final = 60
NEWEY_WEST_LAG: Final = 19
OUTPUT_SCALE: Final = Decimal("0.000000000001")
POLICY_V3_VERSION: Final = "myquant.v17.v5.factor-regime-diagnostic-policy.v3"
POLICY_V3_PATH: Final = (
    "quant_investor/v17_v5_contract/resources/factor_regime_diagnostic_policy.v3.json"
)
_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_DECIMAL_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
    re.ASCII,
)
_LIMITATION_CODE_RE: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    re.ASCII,
)
_FORBIDDEN_KEY_PARTS: Final = (
    "factor_weight",
    "recommended_weight",
    "target_weight",
    "production_weight",
    "portfolio_weight",
    "lifecycle_action",
    "validity",
)
_FORBIDDEN_KEYS: Final = {
    "tier",
    "promotion",
    "production_apply",
    "buy_signal",
    "sell_signal",
}


class FactorRegimeDiagnosticError(ValueError):
    """Raised when regime-conditioned diagnostic input is malformed."""

    exit_code = 2


def _fail(message: str) -> None:
    raise FactorRegimeDiagnosticError(message)


def _limitation_code(value: Any, *, label: str) -> str:
    if type(value) is not str or _LIMITATION_CODE_RE.fullmatch(value) is None:
        _fail(f"{label} must be a stable limitation code")
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _fail(f"{label} must be a second-precision UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorRegimeDiagnosticError(f"{label} is not a valid UTC timestamp") from exc


def _decimal(value: Any, *, label: str, minimum: Decimal, maximum: Decimal) -> Decimal:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise FactorRegimeDiagnosticError(f"{label} is not a finite decimal") from exc
    if not parsed.is_finite() or parsed < minimum or parsed > maximum:
        _fail(f"{label} is out of range")
    if parsed.is_zero() and value.startswith("-"):
        _fail(f"{label} is not canonical")
    return parsed


def _render(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = value.quantize(OUTPUT_SCALE, rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _sample_std(values: Sequence[Decimal]) -> Decimal | None:
    if len(values) < 2:
        return None
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / Decimal(len(values) - 1)
    return variance.sqrt()


def _nearest_rank_p10(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(Decimal("0.10") * Decimal(len(ordered))))
    return ordered[rank - 1]


def _newey_west(
    values: Sequence[Decimal], *, lag: int = NEWEY_WEST_LAG
) -> tuple[str | None, str | None]:
    if len(values) < lag + 1:
        return None, None
    average = _mean(values)
    centered = [value - average for value in values]
    count = Decimal(len(centered))
    with localcontext() as context:
        context.prec = 50
        gamma_zero = sum(value * value for value in centered) / count
        long_run_variance = gamma_zero
        for step in range(1, lag + 1):
            covariance = (
                sum(
                    centered[index] * centered[index - step] for index in range(step, len(centered))
                )
                / count
            )
            weight = Decimal(1) - (Decimal(step) / Decimal(lag + 1))
            long_run_variance += Decimal(2) * weight * covariance
        if long_run_variance <= 0:
            return None, None
        se = (long_run_variance / count).sqrt()
        if se <= 0:
            return None, None
        t_stat = average / se
    return _render(se), _render(t_stat)


def _forbidden_key_scan(value: Any, *, path: str = "$") -> None:
    if path == "$.authority":
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is str:
                lowered = key.lower()
                if lowered in _FORBIDDEN_KEYS or any(
                    part in lowered for part in _FORBIDDEN_KEY_PARTS
                ):
                    _fail(f"forbidden governance or weight field at {path}.{key}")
            _forbidden_key_scan(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_key_scan(child, path=f"{path}[{index}]")


def _policy_ref(policy_ref: Mapping[str, Any]) -> dict[str, Any]:
    if type(policy_ref) is not dict or set(policy_ref) != {
        "artifact_id",
        "byte_sha256",
        "relative_path",
        "semantic_sha256",
        "version",
    }:
        _fail("policy_ref must be an object")
    try:
        document = {
            "artifact_id": require_identifier(
                policy_ref["artifact_id"], label="policy artifact_id"
            ),
            "byte_sha256": require_sha256(policy_ref["byte_sha256"], label="policy byte_sha256"),
            "semantic_sha256": require_sha256(
                policy_ref["semantic_sha256"],
                label="policy semantic_sha256",
            ),
            "version": require_identifier(policy_ref["version"], label="policy version"),
        }
    except (KeyError, IdentityContractError) as exc:
        raise FactorRegimeDiagnosticError("policy_ref is invalid") from exc
    try:
        document["relative_path"] = require_relative_path(
            policy_ref["relative_path"],
            label="policy relative_path",
        )
    except IdentityContractError as exc:
        raise FactorRegimeDiagnosticError(str(exc)) from exc
    expected_current = {
        "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    }
    if document == expected_current and document["version"] == POLICY_V3_VERSION:
        return document
    if document["version"] in {
        "myquant.v17.v5.factor-regime-diagnostic-policy.v1",
        "myquant.v17.v5.factor-regime-diagnostic-policy.v2",
    }:
        _fail("Sprint 1E-0B regime diagnostics must bind policy v3")
    if document["version"] != POLICY_V3_VERSION or document["relative_path"] != POLICY_V3_PATH:
        _fail("policy_ref does not bind the sealed Sprint 1E-0B policy v3")
    return document


def _content_ref(ref: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if type(ref) is not dict:
        _fail(f"{label} must be an object")
    try:
        document = {
            "artifact_id": require_identifier(ref["artifact_id"], label=f"{label}.artifact_id"),
            "byte_sha256": require_sha256(ref["byte_sha256"], label=f"{label}.byte_sha256"),
            "semantic_sha256": require_sha256(
                ref["semantic_sha256"],
                label=f"{label}.semantic_sha256",
            ),
            "version": require_identifier(ref["version"], label=f"{label}.version"),
        }
    except (KeyError, IdentityContractError) as exc:
        raise FactorRegimeDiagnosticError(f"{label} is invalid") from exc
    if "relative_path" in ref:
        _fail(f"{label} must be a pathless V5 content ref")
    return document


def _inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_semantic_sha(document)
    except Exception as exc:
        raise FactorRegimeDiagnosticError("origin inventory is invalid") from exc
    if payload.get("version") != FACTOR_REGIME_ORIGIN_INVENTORY_VERSION:
        _fail("origin inventory version mismatch")
    if payload.get("authority") != NO_AUTHORITY:
        _fail("origin inventory grants authority")
    if payload.get("horizon_sessions") != HORIZON_SESSIONS:
        _fail("origin inventory horizon mismatch")
    if payload.get("policy_ref", {}).get("version") != POLICY_V3_VERSION:
        _fail("origin inventory must bind Sprint 1E-0B policy v3")
    return payload


def _rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = inventory.get("origin_rows")
    if not isinstance(rows, list):
        _fail("origin inventory rows are absent")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict:
            _fail("origin inventory row is invalid")
        rank_ic = row.get("rank_ic")
        normalized.append(
            {
                "comparable_symbol_count": int(row["comparable_symbol_count"]),
                "coverage": _decimal(
                    row["coverage"],
                    label="origin coverage",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                ),
                "decision_session": str(row["decision_session"]),
                "eligible_symbol_count": int(row["eligible_symbol_count"]),
                "rank_ic": (
                    None
                    if rank_ic is None
                    else _decimal(
                        rank_ic,
                        label="origin rank_ic",
                        minimum=Decimal("-1"),
                        maximum=Decimal("1"),
                    )
                ),
                "regime_evidence_ref": dict(row["regime_evidence_ref"]),
                "regime_state": str(row["regime_state"]),
                "state_probabilities": row.get("state_probabilities"),
            }
        )
    return normalized


def _metrics(
    rows: Sequence[Mapping[str, Any]], *, unconditional_mean: Decimal | None
) -> dict[str, Any]:
    origin_count = len(rows)
    available_rank_ic = [row["rank_ic"] for row in rows if isinstance(row["rank_ic"], Decimal)]
    coverage = [row["coverage"] for row in rows]
    eligible = [Decimal(row["eligible_symbol_count"]) for row in rows]
    se, t_stat = _newey_west(available_rank_ic)
    rank_ic_std = _sample_std(available_rank_ic)
    rank_ic_mean = _mean(available_rank_ic) if available_rank_ic else None
    if available_rank_ic and rank_ic_std is not None and rank_ic_std > 0:
        rank_icir = rank_ic_mean / rank_ic_std  # type: ignore[operator]
    else:
        rank_icir = None
    first_session = min((str(row["decision_session"]) for row in rows), default=None)
    last_session = max((str(row["decision_session"]) for row in rows), default=None)
    delta = None
    if rank_ic_mean is not None and unconditional_mean is not None:
        delta = rank_ic_mean - unconditional_mean
    limitations: list[str] = []
    if origin_count < MINIMUM_DESCRIPTIVE_ORIGINS:
        limitations.append("descriptive_origin_threshold_not_met")
    if origin_count < MINIMUM_STABILITY_ORIGINS:
        limitations.append("stability_origin_threshold_not_met")
    if len(available_rank_ic) < 2:
        limitations.append("insufficient_rank_ic_observations_for_sample_std")
    if len(available_rank_ic) < NEWEY_WEST_LAG + 1:
        limitations.append("insufficient_rank_ic_observations_for_newey_west_lag_19")
    return {
        "coverage_mean": _render(_mean(coverage)) if coverage else None,
        "coverage_min": _render(min(coverage)) if coverage else None,
        "coverage_p10": _render(_nearest_rank_p10(coverage)) if coverage else None,
        "delta_rank_ic_vs_unconditional": _render(delta) if delta is not None else None,
        "descriptive_threshold_met": origin_count >= MINIMUM_DESCRIPTIVE_ORIGINS,
        "eligible_symbol_count_mean": _render(_mean(eligible)) if eligible else None,
        "first_origin_session": first_session,
        "last_origin_session": last_session,
        "limitation_codes": limitations,
        "matured_origin_count": origin_count,
        "newey_west_se_lag_19": se,
        "newey_west_t_stat": t_stat,
        "origin_count": origin_count,
        "positive_rank_ic_rate": (
            _render(
                Decimal(sum(Decimal(row) > 0 for row in available_rank_ic))
                / Decimal(len(available_rank_ic))
            )
            if available_rank_ic
            else None
        ),
        "rank_ic_mean": _render(rank_ic_mean) if rank_ic_mean is not None else None,
        "rank_ic_median": _render(_median(available_rank_ic)) if available_rank_ic else None,
        "rank_ic_std": _render(rank_ic_std) if rank_ic_std is not None else None,
        "rank_icir": _render(rank_icir) if rank_icir is not None else None,
        "stability_threshold_met": origin_count >= MINIMUM_STABILITY_ORIGINS,
        "status": "ACCUMULATING" if origin_count else "UNOBSERVED",
    }


def _posterior_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    hard_state_probabilities: list[Decimal] = []
    for row in rows:
        probabilities = row.get("state_probabilities")
        if not isinstance(probabilities, list):
            continue
        matches = [
            item["probability"]
            for item in probabilities
            if type(item) is dict and item.get("regime_state") == row["regime_state"]
        ]
        if len(matches) != 1:
            continue
        value = matches[0]
        if value is None:
            continue
        hard_state_probabilities.append(
            _decimal(
                value,
                label="posterior hard-state probability",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        )
    if not hard_state_probabilities:
        return None
    return {
        "hard_state_probability_mean": _render(_mean(hard_state_probabilities)),
        "hard_state_probability_min": _render(min(hard_state_probabilities)),
        "hard_state_probability_p10": _render(_nearest_rank_p10(hard_state_probabilities)),
        "posterior_origin_count": len(hard_state_probabilities),
    }


def _unconditional_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row["rank_ic"] for row in rows if isinstance(row["rank_ic"], Decimal)]
    unconditional_mean = _mean(available) if available else None
    values = _metrics(rows, unconditional_mean=unconditional_mean)
    for field in (
        "delta_rank_ic_vs_unconditional",
        "descriptive_threshold_met",
        "limitation_codes",
        "origin_count",
        "stability_threshold_met",
        "status",
    ):
        values.pop(field)
    return values


def _validate_diagnostic(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_semantic_sha(document)
        require_identifier(payload["diagnostic_id"], label="diagnostic_id")
        require_identifier(payload["strategy_id"], label="strategy_id")
        require_identifier(payload["factor_name"], label="factor_name")
        _timestamp(payload["cutoff"], label="cutoff")
        _timestamp(payload["created_at"], label="created_at")
    except Exception as exc:
        raise FactorRegimeDiagnosticError("regime-conditioned diagnostic is invalid") from exc
    if (
        payload.get("status") == "UNAVAILABLE"
        and payload.get("factor_implementation_sha256") is None
    ):
        pass
    else:
        try:
            require_sha256(
                payload["factor_implementation_sha256"],
                label="factor_implementation_sha256",
            )
        except Exception as exc:
            raise FactorRegimeDiagnosticError(
                "regime-conditioned diagnostic factor implementation is invalid"
            ) from exc
    if payload.get("version") != REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION:
        _fail("regime-conditioned diagnostic version mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        _fail("regime-conditioned diagnostic protocol mismatch")
    if payload.get("authority") != NO_AUTHORITY:
        _fail("regime-conditioned diagnostic grants authority")
    if payload.get("status") not in {"UNAVAILABLE", "UNOBSERVED", "ACCUMULATING"}:
        _fail("regime-conditioned diagnostic status is invalid")
    occupancy = payload.get("regime_occupancy")
    by_regime = payload.get("by_regime")
    if type(occupancy) is not dict or type(by_regime) is not list:
        _fail("regime-conditioned diagnostic grouping is invalid")
    regime_counts = occupancy.get("regime_origin_counts")
    if type(regime_counts) is not list:
        _fail("regime-conditioned diagnostic occupancy is invalid")
    if any(
        type(row) is not dict or row.get("regime_state") not in REGIME_CONDITIONING_STATES
        for row in regime_counts
    ):
        _fail("regime occupancy contains an ineligible state")
    if any(
        type(row) is not dict or row.get("regime_state") not in REGIME_CONDITIONING_STATES
        for row in by_regime
    ):
        _fail("by-regime diagnostic contains an ineligible state")
    _forbidden_key_scan(payload)
    encoded = canonical_bytes(payload)
    if b"NaN" in encoded or b"Infinity" in encoded:
        _fail("regime-conditioned diagnostic contains non-finite JSON")
    identity_material = dict(payload)
    identity_material.pop("diagnostic_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if payload["diagnostic_id"] != f"regime-conditioned-factor-diagnostic-{identity[:32]}":
        _fail("regime-conditioned diagnostic identity mismatch")
    return payload


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    identity_material = dict(document)
    identity_material.pop("diagnostic_id", None)
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    document["diagnostic_id"] = f"regime-conditioned-factor-diagnostic-{identity[:32]}"
    return _validate_diagnostic(seal_semantic(document))


def build_unavailable_regime_conditioned_factor_diagnostic(
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str | None,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    unavailable_prerequisites: Sequence[str],
) -> dict[str, Any]:
    """Build a deterministic UNAVAILABLE diagnostic for explicit evidence gaps."""

    try:
        subject_strategy = require_identifier(strategy_id, label="strategy_id")
        subject_factor = require_identifier(factor_name, label="factor_name")
    except IdentityContractError as exc:
        raise FactorRegimeDiagnosticError(str(exc)) from exc
    if factor_implementation_sha256 is None:
        implementation_sha = None
    else:
        try:
            implementation_sha = require_sha256(
                factor_implementation_sha256,
                label="factor_implementation_sha256",
            )
        except IdentityContractError as exc:
            raise FactorRegimeDiagnosticError(str(exc)) from exc
    _timestamp(cutoff, label="cutoff")
    if _timestamp(created_at, label="created_at") < _timestamp(cutoff, label="cutoff"):
        _fail("created_at must not precede cutoff")
    if (
        isinstance(unavailable_prerequisites, (str, bytes))
        or not isinstance(unavailable_prerequisites, Sequence)
        or not unavailable_prerequisites
    ):
        _fail("unavailable_prerequisites must be a nonempty sequence")
    blockers: list[str] = []
    for value in unavailable_prerequisites:
        blockers.append(_limitation_code(value, label="unavailable prerequisite"))
    blockers = sorted(set(blockers))
    return _seal(
        {
            "authority": dict(NO_AUTHORITY),
            "by_regime": [],
            "created_at": created_at,
            "cutoff": cutoff,
            "diagnostic_id": "",
            "factor_evidence_ref": None,
            "factor_implementation_sha256": implementation_sha,
            "factor_name": subject_factor,
            "horizon_sessions": HORIZON_SESSIONS,
            "limitation_codes": blockers,
            "origin_inventory_ref": None,
            "policy_ref": _policy_ref(policy_ref),
            "protocol_version": PROTOCOL_VERSION,
            "regime_occupancy": {
                "ambiguous_regime_count": 0,
                "missing_regime_count": 0,
                "posterior_confidence_summary": None,
                "regime_concentration": None,
                "regime_origin_counts": [],
                "total_origin_count": 0,
            },
            "regime_source_refs": [],
            "status": "UNAVAILABLE",
            "strategy_id": subject_strategy,
            "unconditional_metrics": None,
            "version": REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION,
        }
    )


def build_regime_conditioned_factor_diagnostic(
    *,
    origin_inventory: Mapping[str, Any],
    origin_inventory_ref: Mapping[str, Any],
    factor_evidence_ref: Mapping[str, Any],
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
) -> dict[str, Any]:
    """Build ACCUMULATING/UNOBSERVED diagnostics from verified origin rows."""

    inventory = _inventory(origin_inventory)
    normalized_origin_inventory_ref = _content_ref(
        origin_inventory_ref,
        label="origin_inventory_ref",
    )
    expected_inventory_byte_sha = hashlib.sha256(canonical_resource_bytes(inventory)).hexdigest()
    if normalized_origin_inventory_ref != {
        "artifact_id": inventory["inventory_id"],
        "byte_sha256": expected_inventory_byte_sha,
        "semantic_sha256": inventory["semantic_sha256"],
        "version": FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    }:
        _fail("origin_inventory_ref does not bind the supplied inventory")
    _timestamp(cutoff, label="cutoff")
    _timestamp(created_at, label="created_at")
    if _timestamp(created_at, label="created_at") < _timestamp(cutoff, label="cutoff"):
        _fail("created_at must not precede cutoff")
    rows = _rows(inventory)
    excluded_origin_count = int(inventory.get("excluded_origin_count", 0))
    inventory_limitations = [
        _limitation_code(value, label="inventory limitation code")
        for value in inventory.get("limitation_codes", [])
    ]
    available = [row for row in rows if isinstance(row["rank_ic"], Decimal)]
    unconditional_mean = _mean([row["rank_ic"] for row in available]) if available else None
    unconditional = _unconditional_metrics(rows) if rows else None
    by_regime: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["regime_state"]] = counts.get(row["regime_state"], 0) + 1
    for regime_state in sorted(counts):
        regime_rows = [row for row in rows if row["regime_state"] == regime_state]
        metrics = _metrics(regime_rows, unconditional_mean=unconditional_mean)
        metrics["origin_share"] = _render(Decimal(len(regime_rows)) / Decimal(len(rows)))
        metrics["regime_state"] = regime_state
        by_regime.append(metrics)
    concentration = None
    if rows:
        concentration = _render(
            sum(
                ((Decimal(count) / Decimal(len(rows))) ** 2 for count in counts.values()),
                Decimal(0),
            )
        )
    source_refs: dict[str, dict[str, Any]] = {}
    for row in inventory["origin_rows"]:
        ref = dict(row["regime_evidence_ref"])
        source_refs[ref["artifact_id"]] = ref
    status = "ACCUMULATING" if rows else "UNOBSERVED"
    limitations: list[str] = []
    if status == "UNOBSERVED":
        limitations.append("regime_conditioned_no_observed_origins")
        if excluded_origin_count:
            limitations.append("NO_CONDITIONING_ELIGIBLE_ORIGIN")
    elif len(rows) < MINIMUM_DESCRIPTIVE_ORIGINS:
        limitations.append("descriptive_origin_threshold_not_met")
    if rows and len(rows) < MINIMUM_STABILITY_ORIGINS:
        limitations.append("stability_origin_threshold_not_met")
    limitations = sorted(set([*limitations, *inventory_limitations]))
    return _seal(
        {
            "authority": dict(NO_AUTHORITY),
            "by_regime": by_regime,
            "created_at": created_at,
            "cutoff": cutoff,
            "diagnostic_id": "",
            "factor_evidence_ref": _content_ref(factor_evidence_ref, label="factor_evidence_ref"),
            "factor_implementation_sha256": inventory["factor_implementation_sha256"],
            "factor_name": inventory["factor_name"],
            "horizon_sessions": HORIZON_SESSIONS,
            "limitation_codes": limitations,
            "origin_inventory_ref": normalized_origin_inventory_ref,
            "policy_ref": _policy_ref(policy_ref),
            "protocol_version": PROTOCOL_VERSION,
            "regime_occupancy": {
                "ambiguous_regime_count": 0,
                "missing_regime_count": excluded_origin_count,
                "posterior_confidence_summary": _posterior_summary(rows),
                "regime_concentration": concentration,
                "regime_origin_counts": [
                    {
                        "origin_count": counts[key],
                        "regime_state": key,
                    }
                    for key in sorted(counts)
                ],
                "total_origin_count": len(rows),
            },
            "regime_source_refs": [source_refs[key] for key in sorted(source_refs)],
            "status": status,
            "strategy_id": inventory["strategy_id"],
            "unconditional_metrics": unconditional,
            "version": REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION,
        }
    )


def validate_regime_conditioned_factor_diagnostic_replay(
    artifact: Mapping[str, Any],
    *,
    origin_inventory: Mapping[str, Any] | None = None,
    origin_inventory_ref: Mapping[str, Any] | None = None,
    factor_evidence_ref: Mapping[str, Any] | None = None,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    strategy_id: str | None = None,
    factor_name: str | None = None,
    factor_implementation_sha256: str | None = None,
    unavailable_prerequisites: Sequence[str] = (),
) -> dict[str, Any]:
    """Rebuild and compare a diagnostic without file reads or writes."""

    validated = _validate_diagnostic(artifact)
    if validated["status"] == "UNAVAILABLE":
        if strategy_id is None or factor_name is None or origin_inventory is not None:
            _fail("unavailable replay arguments are inconsistent")
        rebuilt = build_unavailable_regime_conditioned_factor_diagnostic(
            strategy_id=strategy_id,
            factor_name=factor_name,
            factor_implementation_sha256=factor_implementation_sha256,
            policy_ref=policy_ref,
            cutoff=cutoff,
            created_at=created_at,
            unavailable_prerequisites=unavailable_prerequisites,
        )
    else:
        if (
            origin_inventory is None
            or origin_inventory_ref is None
            or factor_evidence_ref is None
            or strategy_id is not None
            or factor_name is not None
            or factor_implementation_sha256 is not None
            or unavailable_prerequisites
        ):
            _fail("observed replay arguments are inconsistent")
        rebuilt = build_regime_conditioned_factor_diagnostic(
            origin_inventory=origin_inventory,
            origin_inventory_ref=origin_inventory_ref,
            factor_evidence_ref=factor_evidence_ref,
            policy_ref=policy_ref,
            cutoff=cutoff,
            created_at=created_at,
        )
    if canonical_bytes(validated) != canonical_bytes(rebuilt):
        _fail("regime-conditioned diagnostic replay mismatch")
    return validated


def validate_regime_conditioned_factor_diagnostic(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed regime-conditioned factor diagnostic artifact."""

    return _validate_diagnostic(artifact)


__all__ = [
    "REGIME_CONDITIONED_FACTOR_DIAGNOSTIC_VERSION",
    "FactorRegimeDiagnosticError",
    "build_regime_conditioned_factor_diagnostic",
    "build_unavailable_regime_conditioned_factor_diagnostic",
    "validate_regime_conditioned_factor_diagnostic",
    "validate_regime_conditioned_factor_diagnostic_replay",
]
