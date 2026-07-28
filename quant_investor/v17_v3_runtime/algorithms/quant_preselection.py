"""Strict, PIT-only Quant preselection over an ordered full-A universe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

from .decimal_normalization import DECIMAL_PRECISION, normalize_decimal

MIN_HISTORY = 120
MIN_COVERAGE = Decimal("0.60")
DEFAULT_TOP_N = 500


class QuantPreselectionError(ValueError):
    """Raised when the factor contract or ordered universe is invalid."""


class FactorInventoryConflict(QuantPreselectionError):
    """Raised when the branch inventory conflicts with the sealed contract."""


@dataclass(frozen=True)
class FactorIdentity:
    name: str
    definition_hash: str
    family: str
    lineage: str


@dataclass(frozen=True)
class FactorSpec(FactorIdentity):
    weight: Decimal
    lookback: int = 0
    warmup: int = 0
    minimum_coverage: Decimal = MIN_COVERAGE


@dataclass(frozen=True)
class SymbolObservation:
    symbol: str
    factor_values: Mapping[str, object]
    history_count: int
    data_ready: bool = True
    tradable: bool = True
    liquid: bool = True
    research_eligible: bool = True


@dataclass(frozen=True)
class QuantDisposition:
    symbol: str
    status: str
    reasons: tuple[str, ...] = ()
    score: Decimal | None = None
    selected: bool = False


@dataclass(frozen=True)
class QuantPreselectionResult:
    status: str
    history_required: int
    ordered_domain: tuple[str, ...]
    ready_domain: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    dispositions: tuple[QuantDisposition, ...]
    factor_coverage: Mapping[str, Decimal]
    blockers: tuple[str, ...] = ()

    @property
    def scores(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.symbol: item.score
                for item in self.dispositions
                if item.status == "READY" and item.score is not None
            }
        )


def _canonical_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise QuantPreselectionError(f"{label} must be a canonical non-empty string")
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise QuantPreselectionError(f"{label} must be boolean")
    return value


def _finite_decimal(value: object, *, label: str) -> Decimal | None:
    try:
        return normalize_decimal(value, label=label)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _factor_identity(value: FactorIdentity | Mapping[str, object]) -> FactorIdentity:
    if isinstance(value, FactorIdentity):
        identity = value
    else:
        if not isinstance(value, Mapping):
            raise QuantPreselectionError("factor inventory entries must be mappings")
        identity = FactorIdentity(
            name=_canonical_name(value.get("name"), label="factor.name"),
            definition_hash=_canonical_name(
                value.get("definition_hash", value.get("definition")),
                label="factor.definition_hash",
            ),
            family=_canonical_name(value.get("family"), label="factor.family"),
            lineage=_canonical_name(value.get("lineage"), label="factor.lineage"),
        )
    _canonical_name(identity.name, label="factor.name")
    _canonical_name(identity.definition_hash, label="factor.definition_hash")
    _canonical_name(identity.family, label="factor.family")
    _canonical_name(identity.lineage, label="factor.lineage")
    return identity


def _factor_spec(value: FactorSpec | Mapping[str, object]) -> FactorSpec:
    if isinstance(value, FactorSpec):
        spec = value
    elif isinstance(value, Mapping):
        identity = _factor_identity(value)
        weight = _finite_decimal(value.get("weight"), label=f"{identity.name}.weight")
        if weight is None or weight == 0:
            raise QuantPreselectionError(
                f"factor {identity.name} weight must be finite and nonzero"
            )
        raw_lookback = value.get("lookback", 0)
        raw_warmup = value.get("warmup", 0)
        if (
            isinstance(raw_lookback, bool)
            or not isinstance(raw_lookback, int)
            or isinstance(raw_warmup, bool)
            or not isinstance(raw_warmup, int)
        ):
            raise QuantPreselectionError(f"factor {identity.name} lookback/warmup must be integers")
        lookback = raw_lookback
        warmup = raw_warmup
        coverage = _finite_decimal(
            value.get("minimum_coverage", MIN_COVERAGE),
            label=f"{identity.name}.minimum_coverage",
        )
        if coverage is None:
            raise QuantPreselectionError(f"factor {identity.name} minimum coverage must be finite")
        spec = FactorSpec(
            **identity.__dict__,
            weight=weight,
            lookback=lookback,
            warmup=warmup,
            minimum_coverage=coverage,
        )
    else:
        raise QuantPreselectionError("factor contract entries must be mappings")
    _canonical_name(spec.name, label="factor.name")
    _canonical_name(spec.definition_hash, label=f"{spec.name}.definition_hash")
    _canonical_name(spec.family, label=f"{spec.name}.family")
    _canonical_name(spec.lineage, label=f"{spec.name}.lineage")
    weight = _finite_decimal(spec.weight, label=f"{spec.name}.weight")
    if weight is None or weight == 0:
        raise QuantPreselectionError(f"factor {spec.name} weight must be finite and nonzero")
    if (
        isinstance(spec.lookback, bool)
        or isinstance(spec.warmup, bool)
        or int(spec.lookback) != spec.lookback
        or int(spec.warmup) != spec.warmup
        or spec.lookback < 0
        or spec.warmup < 0
    ):
        raise QuantPreselectionError(f"factor {spec.name} lookback/warmup must be nonnegative")
    coverage = _finite_decimal(
        spec.minimum_coverage,
        label=f"{spec.name}.minimum_coverage",
    )
    if coverage is None or not Decimal("0") <= coverage <= Decimal("1"):
        raise QuantPreselectionError(f"factor {spec.name} minimum coverage must be within [0,1]")
    return FactorSpec(
        name=spec.name,
        definition_hash=spec.definition_hash,
        family=spec.family,
        lineage=spec.lineage,
        weight=weight,
        lookback=int(spec.lookback),
        warmup=int(spec.warmup),
        minimum_coverage=coverage,
    )


def _observation(value: SymbolObservation | Mapping[str, object]) -> SymbolObservation:
    if isinstance(value, SymbolObservation):
        result = value
    elif isinstance(value, Mapping):
        factors = value.get("factor_values", value.get("factors"))
        if not isinstance(factors, Mapping):
            raise QuantPreselectionError("observation factor_values must be a mapping")
        history = value.get("history_count", value.get("history"))
        if isinstance(history, bool):
            raise QuantPreselectionError("observation history_count must be an integer")
        if not isinstance(history, int):
            raise QuantPreselectionError("observation history_count must be an integer")
        history_count = history
        result = SymbolObservation(
            symbol=_canonical_name(value.get("symbol"), label="observation.symbol"),
            factor_values=factors,
            history_count=history_count,
            data_ready=_strict_bool(value.get("data_ready"), label="data_ready"),
            tradable=_strict_bool(value.get("tradable"), label="tradable"),
            liquid=_strict_bool(value.get("liquid"), label="liquid"),
            research_eligible=_strict_bool(
                value.get("research_eligible"), label="research_eligible"
            ),
        )
    else:
        raise QuantPreselectionError("observations must be dataclasses or mappings")
    _canonical_name(result.symbol, label="observation.symbol")
    if not isinstance(result.factor_values, Mapping):
        raise QuantPreselectionError("observation factor_values must be a mapping")
    if (
        isinstance(result.history_count, bool)
        or not isinstance(result.history_count, int)
        or result.history_count < 0
    ):
        raise QuantPreselectionError("observation history_count must be nonnegative integer")
    for field_name in ("data_ready", "tradable", "liquid", "research_eligible"):
        _strict_bool(getattr(result, field_name), label=field_name)
    return result


def validate_disjoint_factor_inventories(
    contract: Sequence[FactorSpec | Mapping[str, object]],
    branch_inventory: Sequence[FactorIdentity | Mapping[str, object]],
) -> tuple[FactorSpec, ...]:
    """Validate that preselector and downstream Quant inventories are disjoint."""

    specs = tuple(_factor_spec(item) for item in contract)
    inventory = tuple(_factor_identity(item) for item in branch_inventory)
    if not specs or not inventory:
        raise QuantPreselectionError(
            "preselector and Quant branch factor inventories must not be empty"
        )
    names = tuple(item.name for item in specs)
    if len(names) != len(set(names)):
        raise QuantPreselectionError("factor contract names must be unique")
    inventory_names = tuple(item.name for item in inventory)
    if len(inventory_names) != len(set(inventory_names)):
        raise QuantPreselectionError("Quant branch factor inventory names must be unique")
    for preselector_factor in specs:
        for quant_factor in inventory:
            for field_name in ("definition_hash", "family", "lineage"):
                if getattr(preselector_factor, field_name) == getattr(quant_factor, field_name):
                    raise FactorInventoryConflict(
                        "factor inventory conflict:"
                        f"{preselector_factor.name}:{quant_factor.name}:{field_name}"
                    )
    return specs


def validate_factor_inventory(
    contract: Sequence[FactorSpec | Mapping[str, object]],
    branch_inventory: Sequence[FactorIdentity | Mapping[str, object]],
) -> tuple[FactorSpec, ...]:
    """Compatibility name for the explicit disjoint-inventory gate."""

    return validate_disjoint_factor_inventories(contract, branch_inventory)


def _average_ranks(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [Decimal("0")] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        average = Decimal(cursor + 1 + stop) / Decimal("2")
        for index in order[cursor:stop]:
            ranks[index] = average
        cursor = stop
    return tuple(ranks)


def run_quant_preselection(
    observations: Sequence[SymbolObservation | Mapping[str, object]],
    *,
    factor_contract: Sequence[FactorSpec | Mapping[str, object]],
    branch_inventory: Sequence[FactorIdentity | Mapping[str, object]],
    top_n: int = DEFAULT_TOP_N,
) -> QuantPreselectionResult:
    """Apply strict gates and rank a complete PIT factor matrix without imputation."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 0 < top_n <= DEFAULT_TOP_N:
        raise QuantPreselectionError("top_n must be within [1, 500]")
    specs = validate_factor_inventory(factor_contract, branch_inventory)
    rows = tuple(_observation(item) for item in observations)
    symbols = tuple(row.symbol for row in rows)
    if len(symbols) != len(set(symbols)):
        raise QuantPreselectionError("ordered full-A observations contain duplicate symbols")
    history_required = max(
        MIN_HISTORY,
        *(max(int(spec.lookback), int(spec.warmup)) for spec in specs),
    )
    reasons: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    gate_rows: list[SymbolObservation] = []
    for row in rows:
        if not row.data_ready:
            reasons[row.symbol].append("data_unavailable")
        if not row.tradable:
            reasons[row.symbol].append("not_tradable")
        if not row.liquid:
            reasons[row.symbol].append("liquidity_gate_failed")
        if not row.research_eligible:
            reasons[row.symbol].append("research_gate_failed")
        if row.history_count < history_required:
            reasons[row.symbol].append(f"history_below_required:{history_required}")
        if not reasons[row.symbol]:
            gate_rows.append(row)

    factor_coverage: dict[str, Decimal] = {}
    finite_values: dict[str, dict[str, Decimal]] = {}
    for row in gate_rows:
        finite_values[row.symbol] = {}
        for spec in specs:
            value = _finite_decimal(
                row.factor_values.get(spec.name),
                label=f"{row.symbol}.{spec.name}",
            )
            if value is None:
                reasons[row.symbol].append(f"factor_missing_or_nonfinite:{spec.name}")
            else:
                finite_values[row.symbol][spec.name] = value
    gate_count = len(gate_rows)
    blockers: list[str] = []
    for spec in specs:
        available = sum(spec.name in finite_values[row.symbol] for row in gate_rows)
        coverage = (
            normalize_decimal(
                Decimal(available) / Decimal(gate_count),
                label=f"{spec.name}.coverage",
            )
            if gate_count
            else Decimal("0").quantize(Decimal("0.000000000001"))
        )
        factor_coverage[spec.name] = coverage
        threshold = max(MIN_COVERAGE, spec.minimum_coverage)
        if coverage < threshold:
            blockers.append(
                f"factor_coverage_below_threshold:{spec.name}:{coverage:.12f}:{threshold:.12f}"
            )
    ready_rows = tuple(row for row in gate_rows if not reasons[row.symbol])
    if not ready_rows:
        blockers.append("ready_domain_empty")
    matrix: dict[str, tuple[Decimal, ...]] = {}
    for spec in specs:
        values = tuple(finite_values[row.symbol][spec.name] for row in ready_rows)
        if len(values) < 2 or len(set(values)) == 1:
            blockers.append(f"ready_factor_constant:{spec.name}")
        matrix[spec.name] = values

    scores: dict[str, Decimal] = {}
    if not blockers:
        blended = [Decimal("0")] * len(ready_rows)
        denominator = sum((abs(spec.weight) for spec in specs), Decimal("0"))
        if denominator <= 0:
            raise QuantPreselectionError("factor absolute weight sum must be positive")
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            for spec in specs:
                ranks = _average_ranks(matrix[spec.name])
                mean = sum(ranks, Decimal("0")) / Decimal(len(ranks))
                variance = sum(
                    ((rank - mean) * (rank - mean) for rank in ranks),
                    Decimal("0"),
                ) / Decimal(len(ranks))
                standard_deviation = variance.sqrt()
                for index, rank in enumerate(ranks):
                    standardized = (rank - mean) / (standard_deviation + Decimal("0.000000001"))
                    clipped = min(
                        Decimal("3"),
                        max(Decimal("-3"), standardized),
                    ) / Decimal("3")
                    blended[index] += spec.weight * clipped
            blended = [value / denominator for value in blended]
        scores = {
            row.symbol: normalize_decimal(score, label=f"{row.symbol}.score")
            for row, score in zip(ready_rows, blended, strict=True)
        }

    ranked = tuple(sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[:top_n])
    selected = frozenset(ranked)
    disposition_rows: list[QuantDisposition] = []
    matrix_blocked = bool(blockers)
    for row in rows:
        row_reasons = tuple(reasons[row.symbol])
        if row_reasons:
            disposition_rows.append(QuantDisposition(row.symbol, "UNAVAILABLE", row_reasons))
        elif matrix_blocked:
            disposition_rows.append(QuantDisposition(row.symbol, "UNAVAILABLE", tuple(blockers)))
        else:
            disposition_rows.append(
                QuantDisposition(
                    row.symbol,
                    "READY",
                    (),
                    scores[row.symbol],
                    row.symbol in selected,
                )
            )
    return QuantPreselectionResult(
        status="READY" if not blockers else "UNAVAILABLE",
        history_required=history_required,
        ordered_domain=symbols,
        ready_domain=tuple(row.symbol for row in ready_rows) if not blockers else (),
        selected_symbols=ranked if not blockers else (),
        dispositions=tuple(disposition_rows),
        factor_coverage=MappingProxyType(dict(factor_coverage)),
        blockers=tuple(dict.fromkeys(blockers)),
    )


__all__ = [
    "DEFAULT_TOP_N",
    "FactorIdentity",
    "FactorInventoryConflict",
    "FactorSpec",
    "QuantDisposition",
    "QuantPreselectionError",
    "QuantPreselectionResult",
    "SymbolObservation",
    "run_quant_preselection",
    "validate_disjoint_factor_inventories",
    "validate_factor_inventory",
]
