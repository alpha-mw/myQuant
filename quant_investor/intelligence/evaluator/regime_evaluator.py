"""Pure regime-conditioned aggregation for Forward Research Evaluation.

The helper in this module consumes already validated, per-origin research
metrics and the selected states emitted by the I0 one-step Markov filter.  It
does not consume posterior probabilities and never performs filtering,
smoothing, fitting, persistence, or authority mutation.

Version-one evaluation is deliberately limited to ``GLOBAL_BREADTH``
subjects.  Market, industry, and theme are evaluated as three independent
one-dimensional strata in their fixed I0 state order; they are not combined
into a joint regime label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import re
from typing import Any, Final

REGIME_EVALUATION_SCOPE: Final = "GLOBAL_BREADTH"
ALLOWED_SUBJECT_TYPES: Final = ("factor", "industry", "theme")
ALLOWED_METRIC_IDS: Final = (
    "rank_ic",
    "icir_base",
    "icir",
    "long_short_spread",
    "cost_adjusted_return",
    "joint_coverage",
    "neutralized_alpha",
    "stability",
    "turnover",
    "drawdown",
)
SCALAR_ORIGIN_METRIC_IDS: Final = (
    "rank_ic",
    "long_short_spread",
    "cost_adjusted_return",
    "joint_coverage",
    "neutralized_alpha",
)
ORIGIN_INPUT_METRIC_IDS: Final = SCALAR_ORIGIN_METRIC_IDS + ("q5_long_only_cost_adjusted_return",)

MARKET_STATES: Final = ("BULL", "RANGE", "HIGH_VOL", "BEAR")
INDUSTRY_STATES: Final = (
    "EARLY_EXPANSION",
    "EXPANSION",
    "PEAK",
    "DECLINE",
    "RECOVERY",
)
THEME_STATES: Final = (
    "EMERGING",
    "ACCELERATING",
    "MAINSTREAM",
    "CROWDED",
    "DECLINING",
)
LAYER_STATE_ORDER: Final = (
    ("market", MARKET_STATES),
    ("industry", INDUSTRY_STATES),
    ("theme", THEME_STATES),
)

_DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
_SUBJECT_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SYMBOL_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BASE_LIMITATIONS: Final = (
    "GLOBAL_BREADTH_ONLY",
    "ONE_DIMENSIONAL_LAYERS_NOT_JOINT_REGIMES",
    "SELECTED_STATES_ONLY_NO_POSTERIOR_INPUT",
    "NO_BACKWARD_SMOOTHING",
    "NO_MODEL_TRAINING_OR_PARAMETER_UPDATE",
    "ICIR_NAIVE_ANNUALIZATION_IGNORES_SERIAL_CORRELATION",
)


class RegimeEvaluationError(ValueError):
    """Raised when a supposedly validated regime-evaluation input is malformed."""


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RegimeEvaluationError(f"{label} must be a sequence")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _SUBJECT_ID_RE.fullmatch(value) is None:
        raise RegimeEvaluationError(f"{label} must be a canonical identifier")
    return value


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise RegimeEvaluationError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise RegimeEvaluationError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise RegimeEvaluationError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is bool:
        raise RegimeEvaluationError(f"{label} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RegimeEvaluationError(f"{label} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise RegimeEvaluationError(f"{label} must be a finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = format(value.quantize(_DECIMAL_QUANTUM), "f")
    return "0.000000000000" if rendered == "-0.000000000000" else rendered


def _mean(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal("0")) / Decimal(len(values))


def _sample_standard_deviation(values: Sequence[Decimal]) -> Decimal:
    average = _mean(values)
    with localcontext() as context:
        context.prec = 50
        variance = sum(
            ((value - average) * (value - average) for value in values),
            Decimal("0"),
        ) / Decimal(len(values) - 1)
        return variance.sqrt()


def _deduplicated(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_subject_specs(
    subject_ids: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, value in enumerate(_sequence(subject_ids, label="subject_ids")):
        if type(value) is not dict or set(value) != {
            "scope",
            "subject_id",
            "subject_type",
        }:
            raise RegimeEvaluationError(
                f"subject_ids[{index}] must contain scope/subject_id/subject_type"
            )
        subject_type = value["subject_type"]
        if subject_type not in ALLOWED_SUBJECT_TYPES:
            raise RegimeEvaluationError(f"subject_ids[{index}].subject_type is not supported")
        if value["scope"] != REGIME_EVALUATION_SCOPE:
            raise RegimeEvaluationError("regime evaluation v1 is GLOBAL_BREADTH only")
        normalized.append(
            {
                "scope": REGIME_EVALUATION_SCOPE,
                "subject_id": _identifier(
                    value["subject_id"], label=f"subject_ids[{index}].subject_id"
                ),
                "subject_type": subject_type,
            }
        )
    keys = [(row["subject_type"], row["subject_id"]) for row in normalized]
    if len(keys) != len(set(keys)):
        raise RegimeEvaluationError("subject_ids must be unique")
    return sorted(
        normalized,
        key=lambda row: (
            row["subject_type"].encode("ascii"),
            row["subject_id"].encode("ascii"),
        ),
    )


def _normalize_weights(value: Any, *, label: str) -> dict[str, Decimal] | None:
    if value is None:
        return None
    if type(value) is not dict or not value:
        raise RegimeEvaluationError(f"{label} must be a non-empty weight mapping or null")
    normalized: dict[str, Decimal] = {}
    for symbol, raw_weight in value.items():
        if type(symbol) is not str or _SYMBOL_RE.fullmatch(symbol) is None:
            raise RegimeEvaluationError(f"{label} contains an invalid symbol")
        weight = _decimal(raw_weight, label=f"{label}.{symbol}")
        if weight < 0:
            raise RegimeEvaluationError(f"{label} contains a negative weight")
        normalized[symbol] = weight
    if sum(normalized.values(), Decimal("0")) != Decimal("1"):
        raise RegimeEvaluationError(f"{label} weights must sum exactly to 1")
    return dict(sorted(normalized.items(), key=lambda item: item[0].encode("ascii")))


def _normalize_subject_row(
    value: Mapping[str, Any],
    *,
    label: str,
    admitted_subjects: set[tuple[str, str]],
) -> tuple[tuple[str, str], dict[str, Any]]:
    if type(value) is not dict or set(value) != {
        "metrics",
        "q5_weights",
        "scope",
        "subject_id",
        "subject_type",
    }:
        raise RegimeEvaluationError(
            f"{label} must contain metrics/q5_weights/scope/subject_id/subject_type"
        )
    subject_type = value["subject_type"]
    if subject_type not in ALLOWED_SUBJECT_TYPES:
        raise RegimeEvaluationError(f"{label}.subject_type is not supported")
    subject_id = _identifier(value["subject_id"], label=f"{label}.subject_id")
    key = (subject_type, subject_id)
    if key not in admitted_subjects:
        raise RegimeEvaluationError(f"{label} is not declared in subject_ids")
    if value["scope"] != REGIME_EVALUATION_SCOPE:
        raise RegimeEvaluationError("regime evaluation v1 is GLOBAL_BREADTH only")
    metrics = value["metrics"]
    if type(metrics) is not dict or not set(metrics).issubset(ORIGIN_INPUT_METRIC_IDS):
        raise RegimeEvaluationError(f"{label}.metrics contains an unsupported metric")
    normalized_metrics = {
        metric_id: (
            None
            if metrics.get(metric_id) is None
            else _decimal(metrics[metric_id], label=f"{label}.metrics.{metric_id}")
        )
        for metric_id in ORIGIN_INPUT_METRIC_IDS
    }
    return key, {
        "metrics": normalized_metrics,
        "q5_weights": _normalize_weights(value["q5_weights"], label=f"{label}.q5_weights"),
    }


def _normalize_states(value: Any, *, label: str) -> dict[str, str | None]:
    if value is None:
        return {layer: None for layer, _states in LAYER_STATE_ORDER}
    if type(value) is not dict or set(value) != {layer for layer, _states in LAYER_STATE_ORDER}:
        raise RegimeEvaluationError(
            f"{label} must contain exactly market/industry/theme or be null"
        )
    normalized: dict[str, str | None] = {}
    for layer, states in LAYER_STATE_ORDER:
        selected = value[layer]
        if selected is not None and selected not in states:
            raise RegimeEvaluationError(f"{label}.{layer} is outside the fixed state domain")
        normalized[layer] = selected
    return normalized


def _normalize_origins(
    origin_rows: Sequence[Mapping[str, Any]],
    *,
    admitted_subjects: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    required_fields = {
        "label_session",
        "next_open_session",
        "origin_id",
        "origin_session",
        "states",
        "subjects",
    }
    for index, value in enumerate(_sequence(origin_rows, label="origin_rows")):
        if type(value) is not dict or set(value) != required_fields:
            raise RegimeEvaluationError(
                f"origin_rows[{index}] must contain the closed origin row shape"
            )
        origin_id = _identifier(value["origin_id"], label=f"origin_rows[{index}].origin_id")
        origin_session = _session(
            value["origin_session"], label=f"origin_rows[{index}].origin_session"
        )
        label_session = _session(
            value["label_session"], label=f"origin_rows[{index}].label_session"
        )
        next_open_session = _session(
            value["next_open_session"],
            label=f"origin_rows[{index}].next_open_session",
        )
        if label_session < origin_session:
            raise RegimeEvaluationError("label_session cannot precede origin_session")
        if next_open_session <= origin_session:
            raise RegimeEvaluationError("next_open_session must follow origin_session")
        subjects: dict[tuple[str, str], dict[str, Any]] = {}
        for subject_index, subject in enumerate(
            _sequence(value["subjects"], label=f"origin_rows[{index}].subjects")
        ):
            key, row = _normalize_subject_row(
                subject,
                label=f"origin_rows[{index}].subjects[{subject_index}]",
                admitted_subjects=admitted_subjects,
            )
            if key in subjects:
                raise RegimeEvaluationError("an origin cannot repeat a subject")
            subjects[key] = row
        normalized.append(
            {
                "label_session": label_session,
                "next_open_session": next_open_session,
                "origin_id": origin_id,
                "origin_session": origin_session,
                "states": _normalize_states(value["states"], label=f"origin_rows[{index}].states"),
                "subjects": subjects,
            }
        )
    origin_ids = [row["origin_id"] for row in normalized]
    origin_sessions = [row["origin_session"] for row in normalized]
    if len(origin_ids) != len(set(origin_ids)):
        raise RegimeEvaluationError("origin_id values must be unique")
    if len(origin_sessions) != len(set(origin_sessions)):
        raise RegimeEvaluationError("origin_session values must be unique")
    return sorted(
        normalized,
        key=lambda row: (
            row["origin_session"],
            row["origin_id"].encode("ascii"),
        ),
    )


def _metric_row(
    *,
    available_origin_count: int,
    blocker_codes: Sequence[str],
    metric_id: str,
    status: str,
    subject_id: str,
    subject_type: str,
    value: Decimal | None,
) -> dict[str, Any]:
    return {
        "available_origin_count": available_origin_count,
        "blocker_codes": _deduplicated(blocker_codes),
        "metric_id": metric_id,
        "status": status,
        "subject_id": subject_id,
        "subject_type": subject_type,
        "value": None if value is None else _decimal_text(value),
    }


def _state_below_minimum_row(
    *,
    available_origin_count: int,
    metric_id: str,
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    return _metric_row(
        available_origin_count=available_origin_count,
        blocker_codes=("MIN_STRATUM_ORIGINS_NOT_MET",),
        metric_id=metric_id,
        status="UNAVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=None,
    )


def _scalar_metric_row(
    *,
    metric_id: str,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    subject_key = (subject_type, subject_id)
    values: list[Decimal] = []
    missing_subject = False
    missing_metric = False
    for origin in state_origins:
        subject = origin["subjects"].get(subject_key)
        if subject is None:
            missing_subject = True
            continue
        value = subject["metrics"][metric_id]
        if value is None:
            missing_metric = True
            continue
        values.append(value)
    blockers: list[str] = []
    if missing_subject:
        blockers.append("SUBJECT_ORIGIN_DATA_MISSING")
    if missing_metric:
        blockers.append("METRIC_ORIGIN_DATA_MISSING")
    if not values:
        blockers.append("METRIC_VALUES_UNAVAILABLE")
        return _metric_row(
            available_origin_count=0,
            blocker_codes=blockers,
            metric_id=metric_id,
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    return _metric_row(
        available_origin_count=len(values),
        blocker_codes=blockers,
        metric_id=metric_id,
        status="AVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=_mean(values),
    )


def _rank_ic_values(
    state_origins: Sequence[Mapping[str, Any]], subject_key: tuple[str, str]
) -> tuple[list[Decimal], list[str]]:
    values: list[Decimal] = []
    blockers: list[str] = []
    for origin in state_origins:
        subject = origin["subjects"].get(subject_key)
        if subject is None:
            blockers.append("SUBJECT_ORIGIN_DATA_MISSING")
            continue
        value = subject["metrics"]["rank_ic"]
        if value is None:
            blockers.append("RANK_IC_ORIGIN_DATA_MISSING")
            continue
        values.append(value)
    return values, _deduplicated(blockers)


def _icir_row(
    *,
    horizon_sessions: int,
    metric_id: str,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    values, blockers = _rank_ic_values(state_origins, (subject_type, subject_id))
    if len(values) < 2:
        blockers.append("ICIR_REQUIRES_TWO_RANK_IC_ORIGINS")
        return _metric_row(
            available_origin_count=len(values),
            blocker_codes=blockers,
            metric_id=metric_id,
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    standard_deviation = _sample_standard_deviation(values)
    if standard_deviation == 0:
        blockers.append("RANK_IC_ZERO_VARIANCE")
        return _metric_row(
            available_origin_count=len(values),
            blocker_codes=blockers,
            metric_id=metric_id,
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    with localcontext() as context:
        context.prec = 50
        icir_base = _mean(values) / standard_deviation
        value = (
            icir_base
            if metric_id == "icir_base"
            else icir_base * (Decimal("252") / Decimal(horizon_sessions)).sqrt()
        )
    return _metric_row(
        available_origin_count=len(values),
        blocker_codes=blockers,
        metric_id=metric_id,
        status="AVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=value,
    )


def _stability_row(
    *,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    values, blockers = _rank_ic_values(state_origins, (subject_type, subject_id))
    if not values:
        blockers.append("STABILITY_REQUIRES_RANK_IC_ORIGINS")
        return _metric_row(
            available_origin_count=0,
            blocker_codes=blockers,
            metric_id="stability",
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    positive = sum(1 for value in values if value > 0)
    return _metric_row(
        available_origin_count=len(values),
        blocker_codes=blockers,
        metric_id="stability",
        status="AVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=Decimal(positive) / Decimal(len(values)),
    )


def _weight_turnover(previous: Mapping[str, Decimal], current: Mapping[str, Decimal]) -> Decimal:
    symbols = set(previous) | set(current)
    with localcontext() as context:
        context.prec = 50
        return Decimal("0.5") * sum(
            (
                abs(current.get(symbol, Decimal("0")) - previous.get(symbol, Decimal("0")))
                for symbol in symbols
            ),
            Decimal("0"),
        )


def _turnover_row(
    *,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    subject_key = (subject_type, subject_id)
    turnovers: list[Decimal] = []
    blockers: list[str] = []
    for previous, current in zip(state_origins, state_origins[1:]):
        if previous["next_open_session"] != current["origin_session"]:
            blockers.append("NONCONSECUTIVE_ORIGIN_PAIRS_EXCLUDED")
            continue
        previous_subject = previous["subjects"].get(subject_key)
        current_subject = current["subjects"].get(subject_key)
        if previous_subject is None or current_subject is None:
            blockers.append("SUBJECT_ORIGIN_DATA_MISSING")
            continue
        previous_weights = previous_subject["q5_weights"]
        current_weights = current_subject["q5_weights"]
        if previous_weights is None or current_weights is None:
            blockers.append("Q5_WEIGHTS_UNAVAILABLE")
            continue
        turnovers.append(_weight_turnover(previous_weights, current_weights))
    if not turnovers:
        blockers.append("SAME_STATE_CONSECUTIVE_TURNOVER_UNAVAILABLE")
        return _metric_row(
            available_origin_count=0,
            blocker_codes=blockers,
            metric_id="turnover",
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    return _metric_row(
        available_origin_count=len(turnovers),
        blocker_codes=blockers,
        metric_id="turnover",
        status="AVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=_mean(turnovers),
    )


def _drawdown_row(
    *,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
) -> dict[str, Any]:
    subject_key = (subject_type, subject_id)
    blockers: list[str] = []
    for index, earlier in enumerate(state_origins):
        if any(
            later["origin_session"] <= earlier["label_session"]
            for later in state_origins[index + 1 :]
        ):
            return _metric_row(
                available_origin_count=0,
                blocker_codes=("OVERLAPPING_FORWARD_WINDOWS",),
                metric_id="drawdown",
                status="UNAVAILABLE",
                subject_id=subject_id,
                subject_type=subject_type,
                value=None,
            )
    values: list[Decimal] = []
    for origin in state_origins:
        subject = origin["subjects"].get(subject_key)
        if subject is None:
            blockers.append("SUBJECT_ORIGIN_DATA_MISSING")
            continue
        value = subject["metrics"]["q5_long_only_cost_adjusted_return"]
        if value is None:
            blockers.append("Q5_LONG_ONLY_COST_ADJUSTED_RETURN_MISSING")
            continue
        values.append(value)
    if len(values) != len(state_origins):
        blockers.append("COMPLETE_DRAWDOWN_PATH_UNAVAILABLE")
        return _metric_row(
            available_origin_count=len(values),
            blocker_codes=blockers,
            metric_id="drawdown",
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    if any(value < Decimal("-1") for value in values):
        return _metric_row(
            available_origin_count=len(values),
            blocker_codes=("RETURN_NOT_COMPOUNDABLE",),
            metric_id="drawdown",
            status="UNAVAILABLE",
            subject_id=subject_id,
            subject_type=subject_type,
            value=None,
        )
    wealth = Decimal("1")
    peak = wealth
    maximum_drawdown = Decimal("0")
    with localcontext() as context:
        context.prec = 50
        for value in values:
            wealth *= Decimal("1") + value
            if wealth > peak:
                peak = wealth
            if peak > 0:
                maximum_drawdown = max(maximum_drawdown, (peak - wealth) / peak)
    return _metric_row(
        available_origin_count=len(values),
        blocker_codes=blockers,
        metric_id="drawdown",
        status="AVAILABLE",
        subject_id=subject_id,
        subject_type=subject_type,
        value=maximum_drawdown,
    )


def _subject_metric_rows(
    *,
    state_origins: Sequence[Mapping[str, Any]],
    subject_id: str,
    subject_type: str,
    minimum_met: bool,
    horizon_sessions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_id in ALLOWED_METRIC_IDS:
        if not minimum_met:
            row = _state_below_minimum_row(
                available_origin_count=len(state_origins),
                metric_id=metric_id,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        elif metric_id in SCALAR_ORIGIN_METRIC_IDS:
            row = _scalar_metric_row(
                metric_id=metric_id,
                state_origins=state_origins,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        elif metric_id in {"icir_base", "icir"}:
            row = _icir_row(
                horizon_sessions=horizon_sessions,
                metric_id=metric_id,
                state_origins=state_origins,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        elif metric_id == "turnover":
            row = _turnover_row(
                state_origins=state_origins,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        elif metric_id == "drawdown":
            row = _drawdown_row(
                state_origins=state_origins,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        else:
            row = _stability_row(
                state_origins=state_origins,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        rows.append(row)
    return rows


def evaluate_regimes(
    *,
    origin_rows: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[Mapping[str, Any]],
    horizon_sessions: int,
    min_stratum_origins: int,
) -> dict[str, Any]:
    """Evaluate subjects independently within each selected I0 regime state.

    ``origin_rows`` must already be causally validated by the forward
    coordinator.  Each origin contains a selected-state mapping (or ``None``)
    and zero or more declared subject rows.  Per-origin ``metrics`` contain
    only the five equal-origin scalar outputs plus the Q5 long-only
    cost-adjusted path input; ICIR, stability, turnover, and drawdown are
    recomputed inside each stratum.

    Turnover uses only adjacent, same-state origins whose session boundary is
    exact (``previous.next_open_session == current.origin_session``). Drawdown
    is available only when every later origin is strictly after every earlier
    label session and every origin supplies
    ``q5_long_only_cost_adjusted_return``; no overlapping subset is selected.
    A state with fewer than ``min_stratum_origins`` emits all ten metric rows
    as ``UNAVAILABLE``.
    """

    if type(horizon_sessions) is not int or horizon_sessions not in {1, 5, 10, 20, 60}:
        raise RegimeEvaluationError("horizon_sessions is not supported")
    if type(min_stratum_origins) is not int or min_stratum_origins < 1:
        raise RegimeEvaluationError("min_stratum_origins must be a positive integer")

    subjects = _normalize_subject_specs(subject_ids)
    if not subjects:
        raise RegimeEvaluationError("subject_ids must not be empty")
    admitted = {(row["subject_type"], row["subject_id"]) for row in subjects}
    origins = _normalize_origins(origin_rows, admitted_subjects=admitted)

    layer_rows: list[dict[str, Any]] = []
    any_missing_states = False
    for layer, states in LAYER_STATE_ORDER:
        missing_origin_ids = [row["origin_id"] for row in origins if row["states"][layer] is None]
        any_missing_states = any_missing_states or bool(missing_origin_ids)
        state_rows: list[dict[str, Any]] = []
        for state in states:
            state_origins = [row for row in origins if row["states"][layer] == state]
            minimum_met = len(state_origins) >= min_stratum_origins
            factor_metric_rows: list[dict[str, Any]] = []
            for subject in subjects:
                factor_metric_rows.extend(
                    _subject_metric_rows(
                        state_origins=state_origins,
                        subject_id=subject["subject_id"],
                        subject_type=subject["subject_type"],
                        minimum_met=minimum_met,
                        horizon_sessions=horizon_sessions,
                    )
                )
            state_rows.append(
                {
                    "factor_metric_rows": factor_metric_rows,
                    "origin_ids": [row["origin_id"] for row in state_origins],
                    "state": state,
                    "status": "AVAILABLE" if minimum_met else "UNAVAILABLE",
                }
            )
        layer_rows.append(
            {
                "layer": layer,
                "missing_origin_ids": missing_origin_ids,
                "state_rows": state_rows,
            }
        )

    limitations = list(_BASE_LIMITATIONS)
    if any_missing_states:
        limitations.append("MISSING_SELECTED_STATE_ORIGINS_DISCLOSED")
    return {
        "horizon_sessions": horizon_sessions,
        "layer_rows": layer_rows,
        "limitations": limitations,
        "scope": REGIME_EVALUATION_SCOPE,
        "unconditional_factor_refs": [],
    }


__all__ = [
    "ALLOWED_METRIC_IDS",
    "ALLOWED_SUBJECT_TYPES",
    "INDUSTRY_STATES",
    "LAYER_STATE_ORDER",
    "MARKET_STATES",
    "ORIGIN_INPUT_METRIC_IDS",
    "REGIME_EVALUATION_SCOPE",
    "RegimeEvaluationError",
    "THEME_STATES",
    "evaluate_regimes",
]
