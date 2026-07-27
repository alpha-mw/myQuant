"""Same-pool branch validation and deterministic Quant/Fundamental fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

from .decimal_normalization import DECIMAL_PRECISION, normalize_decimal

ALLOWED_BRANCH_STATUSES = frozenset({"READY", "UNAVAILABLE"})
MIN_FUSION_WEIGHT = Decimal("0.25")
MAX_FUSION_WEIGHT = Decimal("0.75")
FUSION_WEIGHT_STEP = Decimal("0.05")
DEFAULT_FUSION_TOP_N = 24


class BranchFusionError(ValueError):
    """Raised when a branch violates the same-pool fusion contract."""


@dataclass(frozen=True)
class BranchRecord:
    symbol: str
    status: str
    score: object | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BranchOutput:
    branch: str
    ordered_domain: tuple[str, ...]
    bindings: Mapping[str, str]
    records: tuple[BranchRecord, ...]


@dataclass(frozen=True)
class FusionDisposition:
    symbol: str
    status: str
    reason: str | None = None
    quant_percentile: object | None = None
    fundamental_percentile: object | None = None
    fusion_score: object | None = None
    selected: bool = False


@dataclass(frozen=True)
class FusionResult:
    status: str
    quant_weight: object
    fundamental_weight: object
    ordered_domain: tuple[str, ...]
    common_ready_domain: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    dispositions: tuple[FusionDisposition, ...]
    blockers: tuple[str, ...] = ()


def _canonical_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BranchFusionError(f"{label} must be a canonical non-empty string")
    return value


def _finite_score(value: object, *, label: str) -> Decimal:
    try:
        return normalize_decimal(value, label=label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise BranchFusionError(f"{label} must be finite numeric") from exc


def _bindings(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise BranchFusionError(f"{label} must be a mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _canonical_string(raw_key, label=f"{label}.key")
        normalized[key] = _canonical_string(raw_value, label=f"{label}.{key}")
    return MappingProxyType(normalized)


def _record(value: BranchRecord | Mapping[str, object]) -> BranchRecord:
    if isinstance(value, BranchRecord):
        record = value
    elif isinstance(value, Mapping):
        raw_reason = value.get("reason")
        if raw_reason is not None and not isinstance(raw_reason, str):
            raise BranchFusionError("record.reason must be a string or null")
        record = BranchRecord(
            symbol=_canonical_string(value.get("symbol"), label="record.symbol"),
            status=str(value.get("status")),
            score=value.get("score"),
            reason=raw_reason,
        )
    else:
        raise BranchFusionError("branch records must be dataclasses or mappings")
    _canonical_string(record.symbol, label="record.symbol")
    if record.status not in ALLOWED_BRANCH_STATUSES:
        raise BranchFusionError(f"{record.symbol}.status must be READY or UNAVAILABLE")
    if record.status == "READY":
        _finite_score(record.score, label=f"{record.symbol}.score")
        if record.reason is not None:
            raise BranchFusionError(f"{record.symbol}.READY must not carry a reason")
    else:
        if record.score is not None:
            raise BranchFusionError(f"{record.symbol}.UNAVAILABLE must not carry a score")
        _canonical_string(record.reason, label=f"{record.symbol}.reason")
    return record


def _branch_output(value: BranchOutput | Mapping[str, object]) -> BranchOutput:
    if isinstance(value, BranchOutput):
        output = value
    elif isinstance(value, Mapping):
        raw_domain = value.get("ordered_domain")
        raw_records = value.get("records")
        if isinstance(raw_domain, (str, bytes)) or not isinstance(raw_domain, Sequence):
            raise BranchFusionError("ordered_domain must be a sequence")
        if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
            raise BranchFusionError("records must be a sequence")
        output = BranchOutput(
            branch=_canonical_string(value.get("branch"), label="branch"),
            ordered_domain=tuple(str(item) for item in raw_domain),
            bindings=_bindings(value.get("bindings"), label="bindings"),
            records=tuple(_record(item) for item in raw_records),
        )
    else:
        raise BranchFusionError("branch output must be a dataclass or mapping")
    return output


def validate_branch_output(
    output: BranchOutput | Mapping[str, object],
    *,
    ordered_pool: Sequence[str],
    expected_bindings: Mapping[str, str] | None = None,
) -> BranchOutput:
    """Validate one exact ordered record per pool symbol and exact bindings."""

    normalized = _branch_output(output)
    _canonical_string(normalized.branch, label="branch")
    pool = tuple(_canonical_string(symbol, label="ordered_pool.symbol") for symbol in ordered_pool)
    if len(pool) != len(set(pool)):
        raise BranchFusionError("ordered pool contains duplicate symbols")
    declared_domain = tuple(
        _canonical_string(symbol, label="ordered_domain.symbol")
        for symbol in normalized.ordered_domain
    )
    if declared_domain != pool:
        raise BranchFusionError("branch ordered domain does not exactly match ordered pool")
    records = tuple(_record(item) for item in normalized.records)
    record_domain = tuple(item.symbol for item in records)
    if record_domain != pool:
        raise BranchFusionError("branch records do not exactly match ordered pool")
    normalized_bindings = _bindings(normalized.bindings, label="bindings")
    if expected_bindings is not None:
        expected = _bindings(expected_bindings, label="expected_bindings")
        if dict(normalized_bindings) != dict(expected):
            raise BranchFusionError("branch bindings do not exactly match expected bindings")
    return BranchOutput(
        branch=normalized.branch,
        ordered_domain=pool,
        bindings=normalized_bindings,
        records=records,
    )


def _average_percentiles(values: Mapping[str, Decimal]) -> dict[str, Decimal]:
    ordered = sorted(values, key=lambda symbol: (values[symbol], symbol))
    count = len(ordered)
    if count == 1:
        return {ordered[0]: Decimal("1")}
    percentiles: dict[str, Decimal] = {}
    cursor = 0
    while cursor < count:
        stop = cursor + 1
        value = values[ordered[cursor]]
        while stop < count and values[ordered[stop]] == value:
            stop += 1
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            average_rank = Decimal(cursor + 1 + stop) / Decimal("2")
            percentile = (average_rank - Decimal("1")) / Decimal(count - 1)
        percentile = normalize_decimal(percentile, label="branch_percentile")
        for symbol in ordered[cursor:stop]:
            percentiles[symbol] = percentile
        cursor = stop
    return percentiles


def _validate_quant_weight(value: object) -> Decimal:
    weight = _finite_score(value, label="quant_weight")
    scaled = (weight - MIN_FUSION_WEIGHT) / FUSION_WEIGHT_STEP
    integral = scaled.to_integral_value()
    canonical = MIN_FUSION_WEIGHT + integral * FUSION_WEIGHT_STEP
    if not MIN_FUSION_WEIGHT <= weight <= MAX_FUSION_WEIGHT or scaled != integral:
        raise BranchFusionError("quant_weight must be 0.25..0.75 in 0.05 steps")
    return normalize_decimal(canonical, label="quant_weight")


def fuse_branches(
    quant_branch: BranchOutput | Mapping[str, object],
    fundamental_branch: BranchOutput | Mapping[str, object],
    *,
    ordered_pool: Sequence[str],
    quant_weight: object,
    quant_bindings: Mapping[str, str] | None = None,
    fundamental_bindings: Mapping[str, str] | None = None,
    top_n: int = DEFAULT_FUSION_TOP_N,
) -> FusionResult:
    """Fuse percentiles on the common READY domain without backfilling."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise BranchFusionError("top_n must be a positive integer")
    pool = tuple(ordered_pool)
    quant = validate_branch_output(
        quant_branch,
        ordered_pool=pool,
        expected_bindings=quant_bindings,
    )
    fundamental = validate_branch_output(
        fundamental_branch,
        ordered_pool=pool,
        expected_bindings=fundamental_bindings,
    )
    if quant.branch != "quant":
        raise BranchFusionError("quant_branch.branch must be quant")
    if fundamental.branch != "fundamental":
        raise BranchFusionError("fundamental_branch.branch must be fundamental")
    if dict(quant.bindings) != dict(fundamental.bindings):
        raise BranchFusionError("branch bindings do not exactly match each other")
    weight = _validate_quant_weight(quant_weight)
    fundamental_weight = Decimal("1") - weight
    quant_records = {item.symbol: item for item in quant.records}
    fundamental_records = {item.symbol: item for item in fundamental.records}
    common = tuple(
        symbol
        for symbol in pool
        if quant_records[symbol].status == "READY" and fundamental_records[symbol].status == "READY"
    )
    quant_scores = {
        symbol: _finite_score(quant_records[symbol].score, label=f"quant.{symbol}.score")
        for symbol in common
    }
    fundamental_scores = {
        symbol: _finite_score(
            fundamental_records[symbol].score,
            label=f"fundamental.{symbol}.score",
        )
        for symbol in common
    }
    quant_percentiles = _average_percentiles(quant_scores) if common else {}
    fundamental_percentiles = _average_percentiles(fundamental_scores) if common else {}
    fusion_scores: dict[str, Decimal] = {}
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        for symbol in common:
            raw_score = (
                weight * quant_percentiles[symbol]
                + fundamental_weight * fundamental_percentiles[symbol]
            )
            fusion_scores[symbol] = normalize_decimal(
                raw_score,
                label=f"{symbol}.fusion_score",
            )
    ranked = tuple(sorted(common, key=lambda symbol: (-fusion_scores[symbol], symbol)))
    selected_symbols = ranked[:top_n]
    selected = frozenset(selected_symbols)
    blockers = (f"common_ready_below_top_n:{len(common)}:{top_n}",) if len(common) < top_n else ()
    dispositions: list[FusionDisposition] = []
    for symbol in pool:
        if symbol not in fusion_scores:
            unavailable = []
            if quant_records[symbol].status != "READY":
                unavailable.append(f"quant:{quant_records[symbol].reason}")
            if fundamental_records[symbol].status != "READY":
                unavailable.append(f"fundamental:{fundamental_records[symbol].reason}")
            dispositions.append(
                FusionDisposition(
                    symbol=symbol,
                    status="UNAVAILABLE",
                    reason=";".join(unavailable),
                )
            )
        else:
            dispositions.append(
                FusionDisposition(
                    symbol=symbol,
                    status="READY",
                    quant_percentile=normalize_decimal(
                        quant_percentiles[symbol],
                        label=f"{symbol}.quant_percentile",
                    ),
                    fundamental_percentile=normalize_decimal(
                        fundamental_percentiles[symbol],
                        label=f"{symbol}.fundamental_percentile",
                    ),
                    fusion_score=fusion_scores[symbol],
                    selected=symbol in selected,
                )
            )
    return FusionResult(
        status="READY" if not blockers else "UNAVAILABLE",
        quant_weight=normalize_decimal(weight, label="quant_weight"),
        fundamental_weight=normalize_decimal(fundamental_weight, label="fundamental_weight"),
        ordered_domain=pool,
        common_ready_domain=common,
        selected_symbols=selected_symbols,
        dispositions=tuple(dispositions),
        blockers=blockers,
    )


__all__ = [
    "BranchFusionError",
    "BranchOutput",
    "BranchRecord",
    "FusionDisposition",
    "FusionResult",
    "fuse_branches",
    "validate_branch_output",
]
