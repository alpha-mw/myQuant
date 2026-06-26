from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


REGIME_SCOPE_FULL_MARKET = "full_market"
REGIME_SCOPE_MARKET_REFERENCE = "market_reference"
REGIME_SCOPE_SUBSET = "subset"
REGIME_SCOPE_INSUFFICIENT = "insufficient"

FULL_MARKET_UNIVERSE_KEYS = {
    "all",
    "all_a",
    "full",
    "full_a",
    "full_cn",
    "full_market",
    "full_us",
}


@dataclass(frozen=True)
class RegimeScope:
    regime_scope: str
    scope_key: str
    market: str
    base_universe_key: str
    source_universe_key: str
    requested_symbol_count: int
    source_symbol_count: int
    explicit_symbol_count: int
    unsampled_symbol_count: int
    sampled: bool
    production_eligible: bool
    source_description: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = int(default)
    return max(numeric, 0)


def deterministic_symbol_sample(symbols: Iterable[Any], limit: int) -> tuple[list[str], bool, int]:
    ordered = sorted({_text(symbol).upper() for symbol in symbols if _text(symbol)})
    unsampled_count = len(ordered)
    max_count = _positive_int(limit, unsampled_count)
    if max_count <= 0 or unsampled_count <= max_count:
        return ordered, False, unsampled_count
    return ordered[:max_count], True, unsampled_count


def reference_universe_key_for_market(
    market: str,
    config_obj: Any,
) -> str:
    market_text = _text(market).upper()
    if market_text == "US":
        return _text(getattr(config_obj, "MARKOV_REGIME_REFERENCE_UNIVERSE_US", ""), "full_us")
    return _text(getattr(config_obj, "MARKOV_REGIME_REFERENCE_UNIVERSE_CN", ""), "full_a")


def build_scope_key(
    *,
    market: str,
    regime_scope: str,
    source_universe_key: str,
    source_symbol_count: int,
    unsampled_symbol_count: int,
    sampled: bool,
) -> str:
    sample_part = (
        f"sample_{_positive_int(source_symbol_count)}_of_{_positive_int(unsampled_symbol_count)}"
        if sampled
        else f"symbols_{_positive_int(source_symbol_count)}"
    )
    return ":".join(
        [
            _text(market).upper() or "UNKNOWN",
            _text(regime_scope, REGIME_SCOPE_INSUFFICIENT),
            _text(source_universe_key, "unknown_universe"),
            sample_part,
        ]
    )


def build_regime_scope(
    *,
    market: str,
    base_universe_key: str,
    source_universe_key: str,
    requested_symbol_count: int,
    source_symbol_count: int,
    explicit_symbol_count: int,
    unsampled_symbol_count: int,
    sampled: bool,
    min_market_sample: int,
    source_description: str,
    diagnostics: Iterable[str] | None = None,
    force_scope: str | None = None,
) -> RegimeScope:
    diag = [_text(item) for item in list(diagnostics or []) if _text(item)]
    requested = _positive_int(requested_symbol_count)
    source_count = _positive_int(source_symbol_count)
    explicit_count = _positive_int(explicit_symbol_count)
    unsampled_count = _positive_int(unsampled_symbol_count, source_count)
    min_sample = max(_positive_int(min_market_sample, 1), 1)
    source_key = _text(source_universe_key, _text(base_universe_key, "unknown_universe"))
    base_key = _text(base_universe_key, source_key)

    if force_scope:
        regime_scope = _text(force_scope, REGIME_SCOPE_INSUFFICIENT)
    elif source_count < min_sample:
        regime_scope = REGIME_SCOPE_INSUFFICIENT
    elif explicit_count <= 0 and not sampled and source_key.lower() in FULL_MARKET_UNIVERSE_KEYS:
        regime_scope = REGIME_SCOPE_FULL_MARKET
    elif explicit_count <= 0 and not sampled and source_count >= min_sample:
        regime_scope = REGIME_SCOPE_MARKET_REFERENCE
    elif source_count >= min_sample and source_key.lower() in FULL_MARKET_UNIVERSE_KEYS:
        regime_scope = REGIME_SCOPE_MARKET_REFERENCE
    else:
        regime_scope = REGIME_SCOPE_SUBSET

    production_eligible = regime_scope in {
        REGIME_SCOPE_FULL_MARKET,
        REGIME_SCOPE_MARKET_REFERENCE,
    } and source_count >= min_sample
    if not production_eligible:
        if regime_scope == REGIME_SCOPE_INSUFFICIENT:
            diag.append(
                f"markov_market_scope_insufficient:source_symbol_count={source_count},min_market_sample={min_sample}"
            )
        elif regime_scope == REGIME_SCOPE_SUBSET:
            diag.append("markov_market_scope_subset_not_production_eligible")

    return RegimeScope(
        regime_scope=regime_scope,
        scope_key=build_scope_key(
            market=market,
            regime_scope=regime_scope,
            source_universe_key=source_key,
            source_symbol_count=source_count,
            unsampled_symbol_count=unsampled_count,
            sampled=sampled,
        ),
        market=_text(market).upper(),
        base_universe_key=base_key,
        source_universe_key=source_key,
        requested_symbol_count=requested,
        source_symbol_count=source_count,
        explicit_symbol_count=explicit_count,
        unsampled_symbol_count=unsampled_count,
        sampled=bool(sampled),
        production_eligible=production_eligible,
        source_description=_text(source_description, "unknown"),
        diagnostics=diag,
    )


def scope_from_mapping(
    value: RegimeScope | Mapping[str, Any] | None,
    *,
    market: str,
    universe_key: str,
    sample_count: int,
) -> RegimeScope:
    if isinstance(value, RegimeScope):
        return value
    if isinstance(value, Mapping):
        return RegimeScope(
            regime_scope=_text(value.get("regime_scope"), REGIME_SCOPE_INSUFFICIENT),
            scope_key=_text(value.get("scope_key")),
            market=_text(value.get("market"), market).upper(),
            base_universe_key=_text(value.get("base_universe_key"), universe_key),
            source_universe_key=_text(value.get("source_universe_key"), universe_key),
            requested_symbol_count=_positive_int(value.get("requested_symbol_count"), sample_count),
            source_symbol_count=_positive_int(value.get("source_symbol_count"), sample_count),
            explicit_symbol_count=_positive_int(value.get("explicit_symbol_count"), 0),
            unsampled_symbol_count=_positive_int(value.get("unsampled_symbol_count"), sample_count),
            sampled=bool(value.get("sampled", False)),
            production_eligible=bool(value.get("production_eligible", False)),
            source_description=_text(value.get("source_description"), "mapping"),
            diagnostics=[
                _text(item)
                for item in list(value.get("diagnostics", []) or [])
                if _text(item)
            ],
        )
    source_count = _positive_int(sample_count, 0)
    return build_regime_scope(
        market=market,
        base_universe_key=universe_key,
        source_universe_key=universe_key,
        requested_symbol_count=source_count,
        source_symbol_count=source_count,
        explicit_symbol_count=0,
        unsampled_symbol_count=source_count,
        sampled=False,
        min_market_sample=1,
        source_description="engine_default_scope",
    )
