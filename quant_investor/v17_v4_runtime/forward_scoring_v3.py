"""Pure deterministic scoring helpers for the additive V17 v4 forward v3 lane.

The helpers in this module perform no file, provider, model, publication, or
execution access.  Missing observations remain typed missing evidence; the
only synthesized zero is the policy-required exposure for a zero-MAD factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
from typing import Any, Final

import numpy as np

QUANT_SCORING_V3_VERSION: Final = "myquant.v17.v4.forward-quant-scoring.v3"
FUNDAMENTAL_SCORING_V3_VERSION: Final = "myquant.v17.v4.forward-fundamental-scoring.v3"
FUSION_SCORING_V3_VERSION: Final = "myquant.v17.v4.forward-fusion-scoring.v3"

NEUTRALIZER_FIELDS: Final = (
    "industry",
    "log_market_cap",
    "beta_252d",
    "amihud_20d",
)
FINANCIAL_QUALITY_METRICS: Final = (
    "roe",
    "ocf_to_profit",
    "debt_to_assets",
)
FUNDAMENTAL_COMPONENT_WEIGHTS: Final = {
    "financial_quality": Decimal("0.25"),
    "industry_cycle": Decimal("0.25"),
    "earnings_revision": Decimal("0.20"),
    "theme_narrative": Decimal("0.10"),
    "valuation": Decimal("0.15"),
    "governance": Decimal("0.05"),
}
OWNER_PROVIDED_COMPONENTS: Final = tuple(
    name for name in FUNDAMENTAL_COMPONENT_WEIGHTS if name != "financial_quality"
)


class ForwardScoringV3Error(ValueError):
    """Raised when a v3 scoring input is invalid or violates PIT boundaries."""


class ScoreStatusV3(str, Enum):
    AVAILABLE = "AVAILABLE"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceStatusV3(str, Enum):
    AVAILABLE = "AVAILABLE"
    ZERO_MAD = "ZERO_MAD"
    MISSING_VALUE = "MISSING_VALUE"
    MISSING_NEUTRALIZER = "MISSING_NEUTRALIZER"
    INSUFFICIENT_FINANCIAL_QUALITY = "INSUFFICIENT_FINANCIAL_QUALITY"


def _blocked(reason: str) -> None:
    raise ForwardScoringV3Error(f"V17_V4_FORWARD_SCORING_V3_BLOCKED:{reason}")


def _symbols(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("symbols")
    result = tuple(values)
    if len(set(result)) != len(result):
        _blocked("symbols_duplicate")
    for symbol in result:
        if type(symbol) is not str or not symbol:
            _blocked("symbol_invalid")
        try:
            symbol.encode("ascii")
        except UnicodeEncodeError:
            _blocked("symbol_non_ascii")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    if type(value) is bool or type(value) not in {int, float, str, Decimal}:
        _blocked(f"{label}_numeric")
    try:
        result = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, OverflowError):
        _blocked(f"{label}_numeric")
    if not math.isfinite(result):
        _blocked(f"{label}_nonfinite")
    return result


def _canonical_timestamp(value: Any, *, label: str) -> tuple[str, datetime]:
    if type(value) is not str or not value.endswith("Z"):
        _blocked(f"{label}_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _blocked(f"{label}_timestamp")
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond != 0
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        _blocked(f"{label}_timestamp")
    return value, parsed


def type7_quantile_v3(values: Sequence[Any], probability: Any) -> float:
    """Return the R/NumPy type-7 sample quantile."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("quantile_values")
    numeric = sorted(
        _finite_number(value, label=f"quantile_values[{index}]")
        for index, value in enumerate(values)
    )
    probability_value = _finite_number(probability, label="quantile_probability")
    if not 0.0 <= probability_value <= 1.0:
        _blocked("quantile_probability_range")
    if len(numeric) == 1:
        return numeric[0]
    position = (len(numeric) - 1) * probability_value
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return numeric[lower] + fraction * (numeric[upper] - numeric[lower])


def winsorize_type7_v3(
    values: Sequence[Any],
    *,
    lower_probability: Any = 0.01,
    upper_probability: Any = 0.99,
) -> list[float]:
    """Winsorize finite values using type-7 lower and upper quantiles."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("winsor_values")
    numeric = [
        _finite_number(value, label=f"winsor_values[{index}]") for index, value in enumerate(values)
    ]
    lower = _finite_number(lower_probability, label="lower_probability")
    upper = _finite_number(upper_probability, label="upper_probability")
    if not 0.0 <= lower <= upper <= 1.0:
        _blocked("winsor_probability_range")
    floor = type7_quantile_v3(numeric, lower)
    ceiling = type7_quantile_v3(numeric, upper)
    return [min(max(value, floor), ceiling) for value in numeric]


def average_tie_percentiles_v3(scores: Mapping[str, Any]) -> dict[str, float]:
    """Return average-rank weak percentiles over score-present observations."""

    if not isinstance(scores, Mapping):
        _blocked("percentile_scores")
    numeric = {
        key: _finite_number(value, label=f"percentile_scores.{key}")
        for key, value in scores.items()
    }
    if not numeric:
        return {}
    count = len(numeric)
    result: dict[str, float] = {}
    for key, score in numeric.items():
        lower = sum(candidate < score for candidate in numeric.values())
        tied = sum(candidate == score for candidate in numeric.values())
        result[key] = (lower + (tied + 1.0) / 2.0) / count
    return result


def _median(values: Sequence[float]) -> float:
    return type7_quantile_v3(values, 0.5)


def _clean_zero(value: float) -> float:
    return 0.0 if abs(value) <= 1e-14 else float(value)


def _industry_residuals(
    values: Mapping[str, float],
    industries: Mapping[str, str],
) -> dict[str, float]:
    grouped: dict[str, list[str]] = {}
    for symbol in values:
        grouped.setdefault(industries[symbol], []).append(symbol)
    result: dict[str, float] = {}
    for group_symbols in grouped.values():
        mean = sum(values[symbol] for symbol in group_symbols) / len(group_symbols)
        result.update({symbol: _clean_zero(values[symbol] - mean) for symbol in group_symbols})
    return result


def _single_residuals(
    values: Mapping[str, float],
    predictor: Mapping[str, float],
) -> dict[str, float]:
    symbols = tuple(values)
    y_mean = sum(values.values()) / len(symbols)
    x_mean = sum(predictor[symbol] for symbol in symbols) / len(symbols)
    denominator = sum((predictor[symbol] - x_mean) ** 2 for symbol in symbols)
    if denominator <= 1e-24:
        slope = 0.0
    else:
        slope = (
            sum((predictor[symbol] - x_mean) * (values[symbol] - y_mean) for symbol in symbols)
            / denominator
        )
    intercept = y_mean - slope * x_mean
    return {
        symbol: _clean_zero(values[symbol] - intercept - slope * predictor[symbol])
        for symbol in symbols
    }


def _joint_residuals(
    values: Mapping[str, float],
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> dict[str, float]:
    symbols = tuple(values)
    design = np.asarray(
        [[1.0, first[symbol], second[symbol]] for symbol in symbols],
        dtype=float,
    )
    response = np.asarray([values[symbol] for symbol in symbols], dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    return {
        symbol: _clean_zero(float(residual))
        for symbol, residual in zip(symbols, residuals, strict=True)
    }


def _factor_specs(
    selected_factors: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    if (
        isinstance(selected_factors, (str, bytes))
        or not isinstance(selected_factors, Sequence)
        or not selected_factors
    ):
        _blocked("selected_factors")
    result: list[tuple[str, str]] = []
    for index, row in enumerate(selected_factors):
        if not isinstance(row, Mapping):
            _blocked(f"selected_factors[{index}]")
        name = row.get("name")
        family = row.get("family")
        if type(name) is not str or not name or type(family) is not str or not family:
            _blocked(f"selected_factors[{index}]_identity")
        result.append((name, family))
    names = [name for name, _ in result]
    if len(set(names)) != len(names):
        _blocked("selected_factor_duplicate")
    return tuple(result)


def _validate_mapping_domain(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _blocked(f"{label}_mapping")
    if not set(value) <= allowed:
        _blocked(f"{label}_outside_pool")


def _neutralizers(
    *,
    symbols: tuple[str, ...],
    neutralizer_inputs: Mapping[str, Mapping[str, Any]],
    cutoff: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, ...]]]:
    if not isinstance(neutralizer_inputs, Mapping):
        _blocked("neutralizer_inputs")
    _validate_mapping_domain(
        neutralizer_inputs,
        allowed=set(symbols),
        label="neutralizer_inputs",
    )
    _, cutoff_at = _canonical_timestamp(cutoff, label="cutoff")
    available: dict[str, dict[str, Any]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for symbol in symbols:
        row = neutralizer_inputs.get(symbol)
        if row is None:
            missing[symbol] = NEUTRALIZER_FIELDS
            continue
        if not isinstance(row, Mapping):
            _blocked(f"neutralizer_inputs.{symbol}")
        normalized: dict[str, Any] = {}
        missing_fields: list[str] = []
        for field in NEUTRALIZER_FIELDS:
            entry = row.get(field)
            if entry is None:
                missing_fields.append(field)
                continue
            if (
                not isinstance(entry, Mapping)
                or "value" not in entry
                or "available_at" not in entry
            ):
                _blocked(f"neutralizer_inputs.{symbol}.{field}_timed_value")
            _, available_at = _canonical_timestamp(
                entry["available_at"],
                label=f"neutralizer_inputs.{symbol}.{field}.available_at",
            )
            if available_at > cutoff_at:
                _blocked(f"neutralizer_inputs.{symbol}.{field}_after_cutoff")
            raw = entry["value"]
            if raw is None:
                missing_fields.append(field)
            elif field == "industry":
                if type(raw) is not str or not raw:
                    _blocked(f"neutralizer_inputs.{symbol}.industry")
                normalized[field] = raw
            else:
                normalized[field] = _finite_number(
                    raw,
                    label=f"neutralizer_inputs.{symbol}.{field}",
                )
        if missing_fields:
            missing[symbol] = tuple(missing_fields)
        else:
            available[symbol] = normalized
    return available, missing


def score_quant_forward_v3(
    *,
    symbols: Sequence[str],
    selected_factors: Sequence[Mapping[str, Any]],
    factor_values: Mapping[str, Mapping[str, Any]],
    neutralizer_inputs: Mapping[str, Mapping[str, Any]],
    cutoff: str,
) -> dict[str, Any]:
    """Score Quant v3 with robust normalization and sequential neutralization."""

    pool = _symbols(symbols)
    specs = _factor_specs(selected_factors)
    if not isinstance(factor_values, Mapping):
        _blocked("factor_values")
    selected_names = {name for name, _ in specs}
    if not set(factor_values) <= selected_names:
        _blocked("factor_values_unselected_factor")
    neutralizers, missing_neutralizers = _neutralizers(
        symbols=pool,
        neutralizer_inputs=neutralizer_inputs,
        cutoff=cutoff,
    )
    exposure_by_factor: dict[str, dict[str, float]] = {}
    evidence_by_factor: dict[str, dict[str, dict[str, Any]]] = {}
    allowed_symbols = set(pool)

    for factor_name, family in specs:
        declared = factor_values.get(factor_name, {})
        _validate_mapping_domain(
            declared,
            allowed=allowed_symbols,
            label=f"factor_values.{factor_name}",
        )
        raw: dict[str, float] = {}
        for symbol in pool:
            value = declared.get(symbol)
            if value is not None:
                raw[symbol] = _finite_number(
                    value,
                    label=f"factor_values.{factor_name}.{symbol}",
                )
        ordered_present = [symbol for symbol in pool if symbol in raw]
        winsorized: dict[str, float] = {}
        robust_z: dict[str, float] = {}
        zero_mad = False
        if ordered_present:
            clipped = winsorize_type7_v3([raw[symbol] for symbol in ordered_present])
            winsorized = dict(zip(ordered_present, clipped, strict=True))
            center = _median(clipped)
            mad = _median([abs(value - center) for value in clipped])
            zero_mad = mad == 0.0
            robust_z = {
                symbol: (0.0 if zero_mad else (winsorized[symbol] - center) / (1.4826 * mad))
                for symbol in ordered_present
            }

        eligible = [symbol for symbol in pool if symbol in robust_z and symbol in neutralizers]
        industry_residual: dict[str, float] = {}
        cap_residual: dict[str, float] = {}
        if zero_mad:
            exposure = {symbol: 0.0 for symbol in eligible}
        elif eligible:
            industry_residual = _industry_residuals(
                {symbol: robust_z[symbol] for symbol in eligible},
                {symbol: str(neutralizers[symbol]["industry"]) for symbol in eligible},
            )
            cap_residual = _single_residuals(
                industry_residual,
                {symbol: float(neutralizers[symbol]["log_market_cap"]) for symbol in eligible},
            )
            exposure = _joint_residuals(
                cap_residual,
                {symbol: float(neutralizers[symbol]["beta_252d"]) for symbol in eligible},
                {symbol: float(neutralizers[symbol]["amihud_20d"]) for symbol in eligible},
            )
        else:
            exposure = {}
        exposure_by_factor[factor_name] = exposure

        factor_evidence: dict[str, dict[str, Any]] = {}
        for symbol in pool:
            if symbol not in raw:
                status = EvidenceStatusV3.MISSING_VALUE
            elif symbol not in neutralizers:
                status = EvidenceStatusV3.MISSING_NEUTRALIZER
            elif zero_mad:
                status = EvidenceStatusV3.ZERO_MAD
            else:
                status = EvidenceStatusV3.AVAILABLE
            factor_evidence[symbol] = {
                "exposure": exposure.get(symbol),
                "factor_name": factor_name,
                "family": family,
                "industry_residual": industry_residual.get(symbol),
                "log_market_cap_residual": cap_residual.get(symbol),
                "missing_neutralizers": list(missing_neutralizers.get(symbol, ())),
                "raw_value": raw.get(symbol),
                "robust_z": robust_z.get(symbol),
                "status": status.value,
                "winsorized_value": winsorized.get(symbol),
            }
        evidence_by_factor[factor_name] = factor_evidence

    selected_factor_count = len(specs)
    selected_families = tuple(dict.fromkeys(family for _, family in specs))
    selected_family_count = len(selected_families)
    records: list[dict[str, Any]] = []
    for symbol in pool:
        family_factor_values: dict[str, list[tuple[str, float]]] = {}
        for factor_name, family in specs:
            if symbol in exposure_by_factor[factor_name]:
                family_factor_values.setdefault(family, []).append(
                    (factor_name, exposure_by_factor[factor_name][symbol])
                )
        family_scores: list[dict[str, Any]] = [
            {
                "available_factor_count": len(values),
                "factor_names": [factor_name for factor_name, _ in values],
                "family": family,
                "score": sum(value for _, value in values) / len(values),
                "status": EvidenceStatusV3.AVAILABLE.value,
            }
            for family in selected_families
            if (values := family_factor_values.get(family))
        ]
        available_factor_count = sum(int(row["available_factor_count"]) for row in family_scores)
        available_family_count = len(family_scores)
        factor_coverage = available_factor_count / selected_factor_count
        family_coverage = available_family_count / selected_family_count
        coverage = min(factor_coverage, family_coverage)
        raw_composite = (
            sum(float(row["score"]) for row in family_scores) / available_family_count
            if family_scores
            else None
        )
        effective = _clean_zero(raw_composite * coverage) if raw_composite is not None else None
        reasons: list[str] = []
        if available_factor_count < 2:
            reasons.append("AVAILABLE_FACTOR_COUNT_BELOW_2")
        if available_family_count < 2:
            reasons.append("AVAILABLE_FAMILY_COUNT_BELOW_2")
        if coverage < 0.5:
            reasons.append("COVERAGE_BELOW_0_5")
        row_status = ScoreStatusV3.UNAVAILABLE if reasons else ScoreStatusV3.AVAILABLE
        records.append(
            {
                "available_factor_count": available_factor_count,
                "available_family_count": available_family_count,
                "composite_score": raw_composite,
                "confidence_penalty": 1.0 - coverage,
                "coverage": coverage,
                "effective_score": (effective if row_status is ScoreStatusV3.AVAILABLE else None),
                "factor_coverage": factor_coverage,
                "factor_evidence": [
                    evidence_by_factor[factor_name][symbol] for factor_name, _ in specs
                ],
                "family_coverage": family_coverage,
                "family_scores": family_scores,
                "raw_composite_score": raw_composite,
                "score_present": row_status is ScoreStatusV3.AVAILABLE,
                "status": row_status.value,
                "symbol": symbol,
                "unavailability_reasons": reasons,
            }
        )
    return {
        "cutoff": _canonical_timestamp(cutoff, label="cutoff")[0],
        "records": records,
        "selected_factor_count": selected_factor_count,
        "selected_factors": [{"family": family, "name": name} for name, family in specs],
        "selected_family_count": selected_family_count,
        "version": QUANT_SCORING_V3_VERSION,
    }


def _optional_score(
    entry: Any,
    *,
    label: str,
    cutoff_at: datetime,
) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        if "score" not in entry or "available_at" not in entry:
            _blocked(f"{label}_pit_score")
        _, available_at = _canonical_timestamp(
            entry["available_at"],
            label=f"{label}.available_at",
        )
        if available_at > cutoff_at:
            _blocked(f"{label}_after_cutoff")
        entry = entry["score"]
        if entry is None:
            return None
    score = _finite_number(entry, label=label)
    if not 0.0 <= score <= 1.0:
        _blocked(f"{label}_range")
    return score


def score_fundamental_forward_v3(
    *,
    symbols: Sequence[str],
    financial_quality_values: Mapping[str, Mapping[str, Any]],
    owner_component_scores: Mapping[str, Mapping[str, Any]],
    cutoff: str,
) -> dict[str, Any]:
    """Score Fundamental v3 with exact configured-weight missing reweighting."""

    pool = _symbols(symbols)
    allowed_symbols = set(pool)
    if not isinstance(financial_quality_values, Mapping):
        _blocked("financial_quality_values")
    if not set(financial_quality_values) <= set(FINANCIAL_QUALITY_METRICS):
        _blocked("financial_quality_unknown_metric")
    if not isinstance(owner_component_scores, Mapping):
        _blocked("owner_component_scores")
    if not set(owner_component_scores) <= set(OWNER_PROVIDED_COMPONENTS):
        _blocked("owner_component_unknown")
    cutoff_text, cutoff_at = _canonical_timestamp(cutoff, label="cutoff")

    metric_values: dict[str, dict[str, float]] = {}
    metric_percentiles: dict[str, dict[str, float]] = {}
    metric_winsorized: dict[str, dict[str, float]] = {}
    for metric in FINANCIAL_QUALITY_METRICS:
        declared = financial_quality_values.get(metric, {})
        _validate_mapping_domain(
            declared,
            allowed=allowed_symbols,
            label=f"financial_quality_values.{metric}",
        )
        present: dict[str, float] = {}
        for symbol in pool:
            value = declared.get(symbol)
            if value is None:
                continue
            numeric = _finite_number(
                value,
                label=f"financial_quality_values.{metric}.{symbol}",
            )
            if metric == "debt_to_assets" and not 0.0 <= numeric <= 1.0:
                _blocked(f"financial_quality_values.{metric}.{symbol}_range")
            present[symbol] = numeric
        metric_values[metric] = present
        ordered_present = [symbol for symbol in pool if symbol in present]
        if ordered_present:
            clipped = winsorize_type7_v3([present[symbol] for symbol in ordered_present])
            metric_winsorized[metric] = dict(zip(ordered_present, clipped, strict=True))
            metric_percentiles[metric] = average_tie_percentiles_v3(metric_winsorized[metric])
        else:
            metric_winsorized[metric] = {}
            metric_percentiles[metric] = {}

    financial_quality: dict[str, float] = {}
    financial_evidence: dict[str, dict[str, Any]] = {}
    for symbol in pool:
        metric_rows: list[dict[str, Any]] = []
        component_values: list[float] = []
        for metric in FINANCIAL_QUALITY_METRICS:
            percentile = metric_percentiles[metric].get(symbol)
            component_score = (
                1.0 - percentile
                if metric == "debt_to_assets" and percentile is not None
                else percentile
            )
            if component_score is not None:
                component_values.append(component_score)
            metric_rows.append(
                {
                    "component_score": component_score,
                    "metric": metric,
                    "percentile": percentile,
                    "raw_value": metric_values[metric].get(symbol),
                    "status": (
                        EvidenceStatusV3.AVAILABLE.value
                        if percentile is not None
                        else EvidenceStatusV3.MISSING_VALUE.value
                    ),
                    "winsorized_value": metric_winsorized[metric].get(symbol),
                }
            )
        available = len(component_values) >= 2
        if available:
            financial_quality[symbol] = sum(component_values) / len(component_values)
        financial_evidence[symbol] = {
            "available_metric_count": len(component_values),
            "metrics": metric_rows,
            "score": financial_quality.get(symbol),
            "status": (
                EvidenceStatusV3.AVAILABLE.value
                if available
                else EvidenceStatusV3.INSUFFICIENT_FINANCIAL_QUALITY.value
            ),
        }

    owner_scores: dict[str, dict[str, float]] = {}
    for component in OWNER_PROVIDED_COMPONENTS:
        declared = owner_component_scores.get(component, {})
        _validate_mapping_domain(
            declared,
            allowed=allowed_symbols,
            label=f"owner_component_scores.{component}",
        )
        owner_scores[component] = {}
        for symbol in pool:
            score = _optional_score(
                declared.get(symbol),
                label=f"owner_component_scores.{component}.{symbol}",
                cutoff_at=cutoff_at,
            )
            if score is not None:
                owner_scores[component][symbol] = score

    records: list[dict[str, Any]] = []
    for symbol in pool:
        scores: dict[str, float] = {}
        if symbol in financial_quality:
            scores["financial_quality"] = financial_quality[symbol]
        for component in OWNER_PROVIDED_COMPONENTS:
            if symbol in owner_scores[component]:
                scores[component] = owner_scores[component][symbol]
        available_weight = sum(
            (FUNDAMENTAL_COMPONENT_WEIGHTS[name] for name in scores),
            Decimal("0"),
        )
        raw_score = (
            sum(
                (
                    FUNDAMENTAL_COMPONENT_WEIGHTS[name] * Decimal(str(scores[name]))
                    for name in scores
                ),
                Decimal("0"),
            )
            / available_weight
            if available_weight
            else None
        )
        coverage = float(available_weight)
        effective_score = (
            _clean_zero(float(raw_score * available_weight)) if raw_score is not None else None
        )
        if available_weight == Decimal("1"):
            status = ScoreStatusV3.COMPLETE
        elif available_weight >= Decimal("0.25"):
            status = ScoreStatusV3.PARTIAL
        else:
            status = ScoreStatusV3.UNAVAILABLE
        component_evidence: list[dict[str, Any]] = []
        for component, weight in FUNDAMENTAL_COMPONENT_WEIGHTS.items():
            if component == "financial_quality":
                evidence: dict[str, Any] = financial_evidence[symbol]
            else:
                component_score = owner_scores[component].get(symbol)
                evidence = {
                    "score": component_score,
                    "status": (
                        EvidenceStatusV3.AVAILABLE.value
                        if component_score is not None
                        else EvidenceStatusV3.MISSING_VALUE.value
                    ),
                }
            component_evidence.append(
                {
                    "component": component,
                    "configured_weight": float(weight),
                    "evidence": evidence,
                    "score": scores.get(component),
                    "status": evidence["status"],
                }
            )
        records.append(
            {
                "available_component_count": len(scores),
                "available_weight": coverage,
                "component_evidence": component_evidence,
                "confidence_penalty": 1.0 - coverage,
                "coverage": coverage,
                "effective_score": effective_score,
                "raw_score": float(raw_score) if raw_score is not None else None,
                "score_present": status is not ScoreStatusV3.UNAVAILABLE,
                "status": status.value,
                "symbol": symbol,
            }
        )
    return {
        "component_weights": {
            name: float(weight) for name, weight in FUNDAMENTAL_COMPONENT_WEIGHTS.items()
        },
        "cutoff": cutoff_text,
        "records": records,
        "version": FUNDAMENTAL_SCORING_V3_VERSION,
    }


def fuse_forward_scores_v3(
    *,
    symbols: Sequence[str],
    quant_scores: Mapping[str, Any],
    fundamental_scores: Mapping[str, Any],
    fundamental_coverages: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse v3 branch scores with per-symbol Fundamental coverage attenuation."""

    pool = _symbols(symbols)
    allowed_symbols = set(pool)
    _validate_mapping_domain(
        quant_scores,
        allowed=allowed_symbols,
        label="quant_scores",
    )
    if set(quant_scores) != allowed_symbols:
        _blocked("quant_scores_quant_always")
    _validate_mapping_domain(
        fundamental_scores,
        allowed=allowed_symbols,
        label="fundamental_scores",
    )
    _validate_mapping_domain(
        fundamental_coverages,
        allowed=allowed_symbols,
        label="fundamental_coverages",
    )
    quant = {
        symbol: _finite_number(quant_scores[symbol], label=f"quant_scores.{symbol}")
        for symbol in pool
    }
    declared_coverages: dict[str, float] = {}
    for symbol, value in fundamental_coverages.items():
        if value is None:
            continue
        coverage = _finite_number(
            value,
            label=f"fundamental_coverages.{symbol}",
        )
        if not 0.0 <= coverage <= 1.0:
            _blocked(f"fundamental_coverages.{symbol}_range")
        declared_coverages[symbol] = coverage

    fundamental: dict[str, float] = {}
    coverages: dict[str, float] = {}
    for symbol in pool:
        score = fundamental_scores.get(symbol)
        if score is None:
            continue
        if symbol not in declared_coverages:
            _blocked(f"fundamental_coverages.{symbol}_missing")
        coverage = declared_coverages[symbol]
        if coverage == 0.0:
            _blocked(f"fundamental_coverages.{symbol}_range")
        fundamental[symbol] = _finite_number(
            score,
            label=f"fundamental_scores.{symbol}",
        )
        coverages[symbol] = coverage

    quant_percentiles = average_tie_percentiles_v3(quant)
    fundamental_percentiles = average_tie_percentiles_v3(fundamental)
    records: list[dict[str, Any]] = []
    for symbol in pool:
        quant_weight = 0.5
        fundamental_available = symbol in fundamental
        fundamental_weight = 0.5 * coverages[symbol] if fundamental_available else 0.0
        available_weight = quant_weight + fundamental_weight
        weighted_sum = quant_weight * quant_percentiles[symbol]
        if fundamental_available:
            weighted_sum += fundamental_weight * fundamental_percentiles[symbol]
        raw_score = weighted_sum / available_weight
        coverage = available_weight
        effective_score = _clean_zero(raw_score * coverage)
        records.append(
            {
                "available_weight": available_weight,
                "branch_evidence": [
                    {
                        "branch": "quant",
                        "configured_weight": 0.5,
                        "effective_configured_weight": 0.5,
                        "percentile": quant_percentiles[symbol],
                        "score": quant[symbol],
                        "status": EvidenceStatusV3.AVAILABLE.value,
                    },
                    {
                        "branch": "fundamental",
                        "configured_weight": 0.5,
                        "effective_configured_weight": fundamental_weight,
                        "fundamental_coverage": (
                            coverages[symbol] if fundamental_available else None
                        ),
                        "percentile": fundamental_percentiles.get(symbol),
                        "score": fundamental.get(symbol),
                        "status": (
                            EvidenceStatusV3.AVAILABLE.value
                            if fundamental_available
                            else ScoreStatusV3.UNAVAILABLE.value
                        ),
                    },
                ],
                "confidence_penalty": 1.0 - coverage,
                "coverage": coverage,
                "effective_score": effective_score,
                "raw_score": raw_score,
                "status": ScoreStatusV3.AVAILABLE.value,
                "symbol": symbol,
            }
        )
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row["effective_score"]),
            str(row["symbol"]).encode("ascii"),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return {
        "base_weights": {"fundamental": 0.5, "quant": 0.5},
        "records": ordered,
        "version": FUSION_SCORING_V3_VERSION,
    }


score_quant_v3 = score_quant_forward_v3
score_fundamental_v3 = score_fundamental_forward_v3
score_fusion_v3 = fuse_forward_scores_v3


__all__ = [
    "EvidenceStatusV3",
    "FINANCIAL_QUALITY_METRICS",
    "FUNDAMENTAL_COMPONENT_WEIGHTS",
    "FUNDAMENTAL_SCORING_V3_VERSION",
    "FUSION_SCORING_V3_VERSION",
    "ForwardScoringV3Error",
    "NEUTRALIZER_FIELDS",
    "OWNER_PROVIDED_COMPONENTS",
    "QUANT_SCORING_V3_VERSION",
    "ScoreStatusV3",
    "average_tie_percentiles_v3",
    "fuse_forward_scores_v3",
    "score_fundamental_forward_v3",
    "score_fundamental_v3",
    "score_fusion_v3",
    "score_quant_forward_v3",
    "score_quant_v3",
    "type7_quantile_v3",
    "winsorize_type7_v3",
]
