"""Migrate `mined-factor-registry.v1` records onto the Factor-v4 schema.

`assess_factor_record_v4` requires `family`, `slot`, `weight`, `maturity`,
`bh_q_value`, `fdr_method`, `runtime_contract`, `evidence` and fresh `health`.
The registry carries none of them -- every record has `family=None`,
`slot=None`, and every candidate has `weight=0.0` -- so rules 1, 4 and 5 of the
v4 health contract fail on the first field read and no record has ever reached
the assessment. `production_factor_count` has therefore always been 0.

That is a schema migration gap, not a statistics problem, so nothing here moves
a threshold. This module owns the deterministic half of the migration: which
five factors, in which family, at which weight, with which params. Maturity, BH
q-values and fresh health are measurements, are taken separately, and are
attached by the caller.

Why exactly one factor per family: v4 wants at least
`MIN_NEW_RISK_FACTOR_COUNT` (5) factors, no factor above
`MAX_FACTOR_ABS_WEIGHT` (0.20) and no family above `MAX_FAMILY_ABS_WEIGHT`
(0.35) of normalized absolute weight. At five factors the per-factor cap forces
all five to exactly 0.20, and two in one family would be 0.40 > 0.35. So five
factors implies five families, and the registry's eight production-level
records supply exactly five after de-duplication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from quant_investor.factors.governance_protocol_v4 import (
    MIN_NEW_RISK_FACTOR_COUNT,
)

# Written out rather than inferred from the category string. Inference would
# silently re-family a record when mining invents a new category, and family is
# what the 35% cap is enforced against.
CATEGORY_TO_FAMILY: Final[dict[str, str]] = {
    "liquidity": "liquidity",
    "trading_activity": "trading_activity",
    "trading_activity_momentum_liquidity": "activity_momentum_liquidity_blend",
    "growth": "growth",
    "formulaic_research": "formulaic_research",
}

# The residualization baseline as it stood when the eight gates were evaluated
# (mining run daily_factor_mining_20260711_211654_codex, whose recorded
# existing_composite_blocker is empty). Pinning to this exact baseline is what
# keeps the recorded gate evidence describing what production will compute; a
# different baseline would be a different factor needing a fresh re-gate.
RESIDUAL_BASELINE_PIN: Final[dict[str, Any]] = {
    "schema_version": "factor-residual-baseline.v1",
    "factors": [
        {
            "name": "pv_low_dollar_volume_5d",
            "implementation": "price_volume:pv_low_dollar_volume_5d",
            "weight": 0.05,
            "direction": 1.0,
        }
    ],
    "baseline_sha256": (
        "22b3bb6c7b10bccc0018a47000fee2854e5028957969d5faa002ca8f35b5dbe5"
    ),
}

_ELIGIBLE_STATES: Final = frozenset({"production_factor", "production_candidate"})
_REQUIRED_GATE_IDS: Final = tuple(range(1, 9))
_FORMULAIC_IMPLEMENTATION: Final = "research_formula:rank_blend"


class RegistryV4MigrationError(ValueError):
    """Raised when the registry cannot yield a valid v4 production set."""


def _passed_all_eight_gates(record: Mapping[str, Any]) -> bool:
    gates = {
        int(item.get("gate_id", -1)): bool(item.get("passed"))
        for item in record.get("gate_results") or []
        if isinstance(item, Mapping)
    }
    return all(gates.get(gate_id) is True for gate_id in _REQUIRED_GATE_IDS)


def _icir(record: Mapping[str, Any]) -> float:
    raw: Any = (record.get("metrics") or {}).get("icir")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return float("-inf")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def select_production_set(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pick one gate-passing factor per family, best recorded ICIR first.

    De-duplication is real here: the registry holds `formula_mom120_*` at three
    blend weights and `fund_fin_net_profit_yoy` at two windows, which are near
    copies rather than independent factors. Choosing the highest ICIR among
    near-copies is a mild selection effect on top of the mining run's own
    search, so the BH correction applied later should be computed against the
    honest candidate count, not against the five survivors.

    Ties break on name so the result never depends on input order.
    """

    by_family: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        if str(raw.get("state") or "") not in _ELIGIBLE_STATES:
            continue
        if not _passed_all_eight_gates(raw):
            continue
        category = str(raw.get("category") or "")
        family = CATEGORY_TO_FAMILY.get(category)
        if family is None:
            continue
        item = deepcopy(dict(raw))
        item["family"] = family
        by_family.setdefault(family, []).append(item)

    missing = sorted(set(CATEGORY_TO_FAMILY.values()) - set(by_family))
    if missing:
        raise RegistryV4MigrationError(
            f"registry has no gate-passing factor for famil(ies): {', '.join(missing)}"
        )

    selected = [
        sorted(items, key=lambda item: (-_icir(item), str(item.get("name") or "")))[0]
        for _family, items in sorted(by_family.items())
    ]
    if len(selected) < MIN_NEW_RISK_FACTOR_COUNT:
        raise RegistryV4MigrationError(
            f"v4 needs {MIN_NEW_RISK_FACTOR_COUNT} families, registry yields "
            f"{len(selected)}"
        )
    return selected


def assign_production_weights(
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Weight the set equally.

    With one factor per family, equal weights are the only allocation that
    satisfies both caps at five factors, so there is no optimization to do and
    nothing to tune after the fact.
    """

    if not selected:
        raise RegistryV4MigrationError("cannot weight an empty production set")
    weight = 1.0 / len(selected)
    return [{**deepcopy(dict(item)), "weight": weight} for item in selected]


def _pinned_formulaic_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite `<primitive>_resid_existing` to the equivalent pinned form.

    The arithmetic is unchanged: the same primitive is residualized against the
    same baseline. Only the baseline's provenance moves from "whatever is
    currently selectable" to a content-hashed pin, which is what makes the
    factor replayable and what `production_set_dependent_primitives` demands.
    """

    updated = deepcopy(dict(params))
    for key in ("left", "right"):
        value = str(updated.get(key) or "")
        if value.endswith("_resid_existing"):
            updated[key] = value.removesuffix("_resid_existing")
            updated["residualize_right_against"] = deepcopy(RESIDUAL_BASELINE_PIN)
    if "residualize_right_against" not in updated:
        raise RegistryV4MigrationError(
            "formulaic record referenced no residualized primitive to pin"
        )
    return updated


def migrate_registry_to_v4_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the v4-shaped production records. Never mutates ``records``.

    Maturity, `bh_q_value`, `health`, `runtime_contract` and `evidence` are
    measurements and are deliberately not invented here; the caller attaches
    them and `assess_factor_record_v4` will keep blocking until it does.
    """

    weighted = assign_production_weights(select_production_set(records))
    migrated: list[dict[str, Any]] = []
    for index, item in enumerate(sorted(weighted, key=lambda r: str(r["family"]))):
        record = deepcopy(item)
        record["state"] = "production_factor"
        record["slot"] = f"v4-slot-{index + 1:02d}-{record['family']}"
        record["fdr_method"] = "benjamini_hochberg_by_family"
        metadata = dict(record.get("metadata") or {})
        if str(record.get("implementation") or "") == _FORMULAIC_IMPLEMENTATION:
            metadata["params"] = _pinned_formulaic_params(metadata.get("params") or {})
        record["metadata"] = metadata
        migrated.append(record)
    return migrated
