"""Offline, source-bound daily Investment Intelligence compilation.

This module composes the stable domain builders without restoring the retired
version-named runtime.  It performs no provider, model, System, Mainline,
portfolio, Paper, broker, order, trade, or persistence operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
import hashlib
import math
import re
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_authority import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
)
from quant_investor.factors.production_observation import (
    validate_factor_production_observation,
)
from quant_investor.market.tushare._core import content_ref as tushare_content_ref
from quant_investor.market.tushare.industry_membership import (
    validate_industry_membership_capture,
    validate_industry_membership_partition_capture,
)
from quant_investor.market.tushare.industry_taxonomy import (
    validate_industry_membership_execution_plan,
    validate_industry_taxonomy_capture,
    validate_industry_taxonomy_execution_plan,
)
from quant_investor.market.tushare.theme_capture import (
    derive_tdx_fallback_company_keyset,
    project_theme_provider_capture,
    validate_theme_provider_capture,
    validate_theme_provider_execution_plan,
)

from ._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    canonical_value,
    company_code,
    decimal_text,
    decimal_value,
    identifier,
    require_artifact_ref,
    require_no_future,
    sha256,
    timestamp,
    validate_artifact_ref,
)
from .decision_context import build_decision_context
from .industry import assess_industry
from .investment_decision import make_investment_decision
from .runtime import compile_evidence, evaluate, forward

POLICY_KIND: Final = "daily_research_policy"
RANK_KIND: Final = "factor_research_rank"
INDUSTRY_PROJECTION_KIND: Final = "industry_source_projection"
THEME_PROJECTION_KIND: Final = "theme_membership_projection"

_FACTOR_ALIASES: Final = {
    "LOW": LOW_DOLLAR_VOLUME,
    "W80": BLEND_W80,
}
_RESEARCH_STAGES: Final = (
    "quant_rank",
    "industry",
    "theme_gate",
    "fundamental",
    "decision",
)
_HEX_FLOAT_RE: Final = re.compile(r"^-?0x(?:[0-9a-f]+(?:\.[0-9a-f]*)?|\.[0-9a-f]+)p[+-]?[0-9]+$")


def _codes(values: Sequence[Any], *, label: str, allow_empty: bool = False) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError(f"{label} must be a sequence")
    rows = [company_code(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or rows != sorted(
        set(rows), key=lambda item: item.encode("ascii")
    ):
        raise IntelligenceError(f"{label} must be unique and ASCII sorted")
    return rows


def _identifiers(values: Sequence[Any], *, label: str, allow_empty: bool = False) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError(f"{label} must be a sequence")
    rows = [identifier(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or rows != sorted(
        set(rows), key=lambda item: item.encode("ascii")
    ):
        raise IntelligenceError(f"{label} must be unique and ASCII sorted")
    return rows


def _ref_rows(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError(f"{label} must be a sequence")
    rows = [
        validate_artifact_ref(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if len(rows) != len({(row["kind"], row["artifact_id"]) for row in rows}):
        raise IntelligenceError(f"{label} contains duplicate identities")
    return sorted(
        rows, key=lambda row: (row["kind"].encode("ascii"), row["artifact_id"].encode("ascii"))
    )


def _company_set_sha256(companies: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(companies))).hexdigest()


def build_daily_research_policy(
    *,
    strategy_id: str,
    effective_from: str,
    effective_signal_date: str,
    effective_to: str | None,
    factor_rows: Sequence[Mapping[str, Any]],
    pool_policy: Mapping[str, Any],
    decision_thresholds: Mapping[str, Any],
    technology_theme_ids: Sequence[str],
    technology_policy_state: str,
    theme_provider_precedence: Sequence[str],
    fundamental_freshness: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Seal one explicit, effective-dated, non-authorizing research policy."""

    strategy = identifier(strategy_id, label="strategy_id")
    start = timestamp(effective_from, label="effective_from")
    end = None if effective_to is None else timestamp(effective_to, label="effective_to")
    instant = timestamp(created_at, label="created_at")
    if start > instant or (end is not None and (end < start or instant > end)):
        raise IntelligenceError("daily research policy chronology is invalid")
    if isinstance(factor_rows, (str, bytes)) or not isinstance(factor_rows, Sequence):
        raise IntelligenceError("factor_rows must be a sequence")
    normalized_factors: list[dict[str, str]] = []
    for index, row in enumerate(factor_rows):
        if type(row) is not dict or set(row) != {
            "direction",
            "factor_alias",
            "factor_id",
            "weight",
        }:
            raise IntelligenceError(f"factor_rows[{index}] shape is invalid")
        alias = identifier(row["factor_alias"], label="factor_alias")
        factor_id = identifier(row["factor_id"], label="factor_id")
        if alias not in _FACTOR_ALIASES or _FACTOR_ALIASES[alias] != factor_id:
            raise IntelligenceError("daily research factor identity is invalid")
        if row["direction"] != "HIGHER_IS_BETTER":
            raise IntelligenceError("daily research factor direction is invalid")
        weight = decimal_value(
            row["weight"], label="factor weight", minimum=Decimal("0"), maximum=Decimal("1")
        )
        normalized_factors.append(
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": alias,
                "factor_id": factor_id,
                "weight": decimal_text(weight),
            }
        )
    normalized_factors.sort(key=lambda row: row["factor_alias"].encode("ascii"))
    if [row["factor_alias"] for row in normalized_factors] != ["LOW", "W80"] or any(
        row["weight"] != "0.500000000000" for row in normalized_factors
    ):
        raise IntelligenceError("daily research factor set must be exact LOW/W80 at 50/50")
    required_pool = {
        "minimum_cohort",
        "missing_rule",
        "normalization",
        "pool_boundary_rule",
        "pool_size",
        "sort_key",
        "tie_rule",
    }
    if type(pool_policy) is not dict or set(pool_policy) != required_pool:
        raise IntelligenceError("pool_policy shape is invalid")
    pool_size = pool_policy["pool_size"]
    minimum_cohort = pool_policy["minimum_cohort"]
    if (
        type(pool_size) is not int
        or type(minimum_cohort) is not int
        or pool_size <= 0
        or pool_size > 200
        or minimum_cohort < pool_size
        or minimum_cohort > 10000
    ):
        raise IntelligenceError("pool size policy is invalid")
    expected_pool_constants = {
        "missing_rule": "BLOCK_ON_ANY_MISSING_OR_NONFINITE",
        "normalization": "AVERAGE_TIE_PERCENTILE_ASCENDING_ZERO_ONE",
        "pool_boundary_rule": "EXACT_LIMIT_ASCII_SYMBOL_TIEBREAK",
        "sort_key": "DESC_COMBINED_PERCENTILE_ASCII_SYMBOL",
        "tie_rule": "AVERAGE_ORDINAL_PERCENTILE",
    }
    if any(pool_policy[field] != value for field, value in expected_pool_constants.items()):
        raise IntelligenceError("pool policy algorithm differs from registered compiler")
    if type(decision_thresholds) is not dict or set(decision_thresholds) != {
        "paper_candidate",
        "research_approved",
    }:
        raise IntelligenceError("decision_thresholds shape is invalid")
    research = decimal_value(
        decision_thresholds["research_approved"],
        label="research threshold",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    paper = decimal_value(
        decision_thresholds["paper_candidate"],
        label="paper threshold",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if paper < research:
        raise IntelligenceError("paper threshold cannot be below research threshold")
    if (
        type(effective_signal_date) is not str
        or re.fullmatch(r"[0-9]{8}", effective_signal_date) is None
    ):
        raise IntelligenceError("effective_signal_date must be YYYYMMDD")
    try:
        datetime.strptime(effective_signal_date, "%Y%m%d")
    except ValueError as exc:
        raise IntelligenceError("effective_signal_date is invalid") from exc
    if technology_policy_state not in {"ACTIVE", "UNCONFIGURED"}:
        raise IntelligenceError("technology policy state is invalid")
    technologies = _identifiers(
        technology_theme_ids,
        label="technology_theme_ids",
        allow_empty=technology_policy_state == "UNCONFIGURED",
    )
    if (technology_policy_state == "UNCONFIGURED") != (not technologies):
        raise IntelligenceError("technology policy state and IDs are inconsistent")
    providers = _identifiers(theme_provider_precedence, label="theme_provider_precedence")
    if providers != ["TUSHARE_DC", "TUSHARE_TDX"]:
        raise IntelligenceError("Theme provider precedence must be exact DC then TDX")
    if any(
        not any(theme_id.startswith(f"{provider}:") for provider in providers)
        for theme_id in technologies
    ):
        raise IntelligenceError("technology theme IDs must be provider-qualified")
    if type(fundamental_freshness) is not dict or set(fundamental_freshness) != {"policy"}:
        raise IntelligenceError("fundamental freshness policy shape is invalid")
    if fundamental_freshness["policy"] != "ADVISORY_NO_FIXED_MAXIMUM":
        raise IntelligenceError("fundamental freshness policy is invalid")
    normalized_pool = {
        **expected_pool_constants,
        "minimum_cohort": minimum_cohort,
        "pool_size": pool_size,
    }
    fields = {
        "decision_thresholds": {
            "paper_candidate": decimal_text(paper),
            "research_approved": decimal_text(research),
        },
        "effective_from": start,
        "effective_signal_date": effective_signal_date,
        "effective_to": end,
        "factor_rows": normalized_factors,
        "fundamental_freshness": {"policy": "ADVISORY_NO_FIXED_MAXIMUM"},
        "pool_policy": normalized_pool,
        "strategy_id": strategy,
        "technology_theme_ids": technologies,
        "technology_policy_state": technology_policy_state,
        "theme_provider_precedence": providers,
    }
    return build_artifact(
        kind=POLICY_KIND,
        identity_field="policy_id",
        identity=business_identity(
            kind=POLICY_KIND,
            identity_inputs={
                "effective_signal_date": effective_signal_date,
                "strategy_id": strategy,
                "technology_policy_state": technology_policy_state,
            },
        ),
        fields=fields,
        created_at=instant,
    )


def validate_daily_research_policy(artifact: Mapping[str, Any] | bytes) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind=POLICY_KIND)
    rebuilt = build_daily_research_policy(
        strategy_id=payload["strategy_id"],
        effective_from=payload["effective_from"],
        effective_signal_date=payload["effective_signal_date"],
        effective_to=payload["effective_to"],
        factor_rows=payload["factor_rows"],
        pool_policy=payload["pool_policy"],
        decision_thresholds=payload["decision_thresholds"],
        technology_theme_ids=payload["technology_theme_ids"],
        technology_policy_state=payload["technology_policy_state"],
        theme_provider_precedence=payload["theme_provider_precedence"],
        fundamental_freshness=payload["fundamental_freshness"],
        created_at=normalized["created_at"],
    )
    if rebuilt != normalized:
        raise IntelligenceError("daily research policy does not replay")
    return normalized


def rank_factor_signals(
    *,
    signal_values: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Apply the one policy-sealed LOW/W80 cross-sectional rank algorithm."""

    policy_artifact = validate_daily_research_policy(policy)
    if type(signal_values) is not dict or set(signal_values) != {
        LOW_DOLLAR_VOLUME,
        BLEND_W80,
    }:
        raise IntelligenceError("Factor signal map shape is invalid")
    common = sorted(signal_values[LOW_DOLLAR_VOLUME], key=lambda item: item.encode("ascii"))
    if common != sorted(signal_values[BLEND_W80], key=lambda item: item.encode("ascii")):
        raise IntelligenceError("Factor LOW/W80 common symbol set differs")
    minimum = policy_artifact["payload"]["pool_policy"]["minimum_cohort"]
    if len(common) < minimum:
        raise IntelligenceError("Factor research cohort is below policy minimum")
    parsed: dict[str, dict[str, Decimal]] = {"LOW": {}, "W80": {}}
    for alias, factor_id in _FACTOR_ALIASES.items():
        for symbol in common:
            company_code(symbol)
            raw = signal_values[factor_id][symbol]
            if type(raw) is not str or _HEX_FLOAT_RE.fullmatch(raw) is None:
                raise IntelligenceError(f"{alias}.{symbol} must be a canonical hex float")
            try:
                observed = float.fromhex(raw)
            except ValueError as exc:
                raise IntelligenceError(f"{alias}.{symbol} hex float is invalid") from exc
            if not math.isfinite(observed) or observed.hex() != raw:
                raise IntelligenceError(f"{alias}.{symbol} must be finite canonical hex float")
            parsed[alias][symbol] = Decimal.from_float(observed)

    def percentiles(rows: Mapping[str, Decimal]) -> dict[str, Decimal]:
        ordered = sorted(rows, key=lambda symbol: (rows[symbol], symbol.encode("ascii")))
        result: dict[str, Decimal] = {}
        denominator = Decimal(max(len(ordered) - 1, 1))
        index = 0
        while index < len(ordered):
            end = index
            while end + 1 < len(ordered) and rows[ordered[end + 1]] == rows[ordered[index]]:
                end += 1
            rank = (Decimal(index) + Decimal(end)) / Decimal(2)
            percentile = rank / denominator if len(ordered) > 1 else Decimal("1")
            for offset in range(index, end + 1):
                result[ordered[offset]] = percentile
            index = end + 1
        return result

    by_alias = {
        alias: {
            symbol: Decimal(decimal_text(value))
            for symbol, value in percentiles(parsed[alias]).items()
        }
        for alias in ("LOW", "W80")
    }
    weights = {
        row["factor_alias"]: Decimal(row["weight"])
        for row in policy_artifact["payload"]["factor_rows"]
    }
    combined = {
        symbol: sum(
            (by_alias[alias][symbol] * weights[alias] for alias in weights),
            Decimal("0"),
        )
        for symbol in common
    }
    pool_size = policy_artifact["payload"]["pool_policy"]["pool_size"]
    selected = sorted(
        common,
        key=lambda symbol: (-combined[symbol], symbol.encode("ascii")),
    )[:pool_size]
    return {
        "common_symbol_count": len(common),
        "common_symbol_set_sha256": hashlib.sha256(canonical_json_bytes(common)).hexdigest(),
        "pool_rows": [
            {
                "combined_percentile": decimal_text(combined[symbol]),
                "factor_percentiles": {
                    alias: decimal_text(by_alias[alias][symbol]) for alias in ("LOW", "W80")
                },
                "symbol": symbol,
            }
            for symbol in selected
        ],
    }


def _validate_active_factor_policy(
    active_rows: Any,
    *,
    policy_payload: Mapping[str, Any],
) -> None:
    if type(active_rows) is not list:
        raise IntelligenceError("Factor active policy rows are invalid")
    active_by_id = {row.get("factor_id"): row for row in active_rows if type(row) is dict}
    if set(active_by_id) != set(_FACTOR_ALIASES.values()) or len(active_rows) != 2:
        raise IntelligenceError("Factor active set differs from exact LOW/W80")
    policy_by_id = {row["factor_id"]: row for row in policy_payload["factor_rows"]}
    for factor_id in _FACTOR_ALIASES.values():
        active_row = active_by_id[factor_id]
        policy_row = policy_by_id[factor_id]
        if (
            active_row.get("direction") != "HIGHER_IS_BETTER"
            or active_row.get("weight") != "0.500000000000"
            or active_row.get("role") != "BOOTSTRAP"
            or active_row.get("selectable") is not True
            or policy_row["direction"] != active_row["direction"]
            or policy_row["weight"] != active_row["weight"]
        ):
            raise IntelligenceError("daily rank policy differs from active Factor generation")


def _require_policy_signal_date(
    policy_payload: Mapping[str, Any],
    *,
    signal_date: str,
) -> None:
    if signal_date < policy_payload["effective_signal_date"]:
        raise IntelligenceError("Factor signal date predates daily research policy")


def build_factor_research_rank(
    *,
    snapshot: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any] | bytes],
    policy: Mapping[str, Any] | bytes,
    as_of: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    sealed_at = cutoff if created_at is None else timestamp(created_at, label="created_at")
    if sealed_at < cutoff:
        raise IntelligenceError("daily rank cannot be sealed before its research cutoff")
    policy_artifact = validate_daily_research_policy(policy)
    policy_payload = policy_artifact["payload"]
    if not (
        policy_payload["effective_from"] <= cutoff
        and (policy_payload["effective_to"] is None or cutoff <= policy_payload["effective_to"])
    ):
        raise IntelligenceError("daily research policy is not effective at cutoff")
    required_snapshot = {
        "active_factor_rows",
        "factor_generation",
        "factor_generation_id",
        "factor_generation_ref",
        "factor_generation_sha256",
        "factor_pointer_sha256",
        "factor_rows",
        "signal_date",
        "signal_values",
    }
    if not required_snapshot.issubset(snapshot):
        raise IntelligenceError("Factor research snapshot is incomplete")
    signal_date = snapshot["signal_date"]
    if cutoff[:10].replace("-", "") != signal_date:
        raise IntelligenceError("Factor research cutoff differs from signal date")
    _require_policy_signal_date(policy_payload, signal_date=signal_date)
    generation_ref = validate_artifact_ref(
        snapshot["factor_generation_ref"], label="factor_generation_ref"
    )
    if generation_ref["kind"] != "factor.production_generation":
        raise IntelligenceError("Factor generation ref kind is invalid")
    require_artifact_ref(
        snapshot["factor_generation_ref"],
        snapshot["factor_generation"],
        label="factor_generation_ref",
    )
    if generation_ref["byte_sha256"] != snapshot["factor_generation_sha256"]:
        raise IntelligenceError("Factor generation ref binding differs")
    observed: dict[str, dict[str, Any]] = {}
    for raw in observations:
        artifact = validate_factor_production_observation(raw)
        payload = artifact["payload"]
        alias = payload["factor_alias"]
        if alias in observed:
            raise IntelligenceError("Factor observation alias is duplicated")
        observed[alias] = artifact
    if set(observed) != {"LOW", "W80"}:
        raise IntelligenceError("Factor observations must be exact LOW/W80")
    factor_rows = {row["factor_alias"]: row for row in snapshot["factor_rows"]}
    if set(factor_rows) != {"LOW", "W80"}:
        raise IntelligenceError("Factor snapshot rows must be exact LOW/W80")
    _validate_active_factor_policy(
        snapshot["active_factor_rows"],
        policy_payload=policy_payload,
    )
    for alias, artifact in observed.items():
        payload = artifact["payload"]
        row = factor_rows[alias]
        expected = {
            "factor_generation_id": snapshot["factor_generation_id"],
            "factor_generation_sha256": snapshot["factor_generation_sha256"],
            "factor_pointer_sha256": snapshot["factor_pointer_sha256"],
            "factor_id": row["factor_id"],
            "signal_date": signal_date,
            "signal_sha256": row["signal_sha256"],
            "signal_symbol_set_sha256": row["signal_symbol_set_sha256"],
            "symbol_count": row["symbol_count"],
        }
        if any(payload.get(field) != value for field, value in expected.items()):
            raise IntelligenceError("Factor observation differs from atomic production head")
    ranked = rank_factor_signals(signal_values=snapshot["signal_values"], policy=policy_artifact)
    symbol_set_sha = ranked["common_symbol_set_sha256"]
    expected_set_shas = {row["signal_symbol_set_sha256"] for row in factor_rows.values()}
    if len(expected_set_shas) != 1 or symbol_set_sha not in expected_set_shas:
        raise IntelligenceError("Factor research symbol set SHA differs from production")
    return build_artifact(
        kind=RANK_KIND,
        identity_field="rank_id",
        identity=business_identity(
            kind=RANK_KIND,
            identity_inputs={
                "factor_pointer_sha256": snapshot["factor_pointer_sha256"],
                "policy_id": policy_artifact["artifact_id"],
            },
        ),
        created_at=sealed_at,
        fields={
            "as_of": cutoff,
            "blocker_codes": [],
            "common_symbol_count": ranked["common_symbol_count"],
            "common_symbol_set_sha256": symbol_set_sha,
            "factor_generation_ref": generation_ref,
            "factor_pointer_sha256": sha256(
                snapshot["factor_pointer_sha256"], label="factor_pointer_sha256"
            ),
            "observation_refs": _ref_rows(
                [artifact_ref(observed[alias]) for alias in ("LOW", "W80")],
                label="observation_refs",
            ),
            "policy_ref": artifact_ref(policy_artifact),
            "pool_rows": ranked["pool_rows"],
            "signal_date": signal_date,
            "status": "READY",
            "strategy_id": policy_payload["strategy_id"],
        },
    )


def validate_factor_research_rank(
    artifact: Mapping[str, Any] | bytes,
    *,
    policy: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind=RANK_KIND)
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("status") != "READY"
        or payload.get("blocker_codes") != []
    ):
        raise IntelligenceError("Factor research rank authority or status is invalid")
    rank_as_of = timestamp(payload.get("as_of"), label="rank.as_of")
    signal_date = payload.get("signal_date")
    if (
        type(signal_date) is not str
        or not re.fullmatch(r"[0-9]{8}", signal_date)
        or signal_date != rank_as_of[:10].replace("-", "")
    ):
        raise IntelligenceError("Factor research rank signal date is invalid")
    identifier(payload.get("strategy_id"), label="rank.strategy_id")
    sha256(payload.get("factor_pointer_sha256"), label="rank.factor_pointer_sha256")
    sha256(payload.get("common_symbol_set_sha256"), label="rank.common_symbol_set_sha256")
    if type(payload.get("common_symbol_count")) is not int or payload["common_symbol_count"] <= 0:
        raise IntelligenceError("Factor research rank symbol count is invalid")
    generation_ref = validate_artifact_ref(
        payload.get("factor_generation_ref"), label="rank.factor_generation_ref"
    )
    if generation_ref["kind"] != "factor.production_generation":
        raise IntelligenceError("Factor research rank generation kind is invalid")
    observation_refs = _ref_rows(payload.get("observation_refs", []), label="observation_refs")
    if len(observation_refs) != 2 or any(
        row["kind"] != "factor.production_observation" for row in observation_refs
    ):
        raise IntelligenceError("Factor research rank observation closure is invalid")
    if payload["observation_refs"] != observation_refs:
        raise IntelligenceError("Factor research rank observation refs are not canonical")
    validate_artifact_ref(payload.get("policy_ref"), label="rank.policy_ref")
    rows = payload.get("pool_rows")
    if type(rows) is not list or not rows or len(rows) > payload["common_symbol_count"]:
        raise IntelligenceError("Factor research rank pool is invalid")
    normalized_rows: list[tuple[Decimal, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "combined_percentile",
            "factor_percentiles",
            "symbol",
        }:
            raise IntelligenceError(f"rank.pool_rows[{index}] shape is invalid")
        symbol = company_code(row["symbol"])
        if symbol in seen:
            raise IntelligenceError("Factor research rank pool is duplicated")
        seen.add(symbol)
        combined = decimal_value(
            row["combined_percentile"],
            label="combined_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if row["combined_percentile"] != decimal_text(combined):
            raise IntelligenceError("combined percentile is not canonical")
        factor_percentiles = row["factor_percentiles"]
        if type(factor_percentiles) is not dict or set(factor_percentiles) != {"LOW", "W80"}:
            raise IntelligenceError("factor percentile shape is invalid")
        for alias in ("LOW", "W80"):
            value = decimal_value(
                factor_percentiles[alias],
                label=f"factor_percentiles.{alias}",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
            if factor_percentiles[alias] != decimal_text(value):
                raise IntelligenceError("factor percentile is not canonical")
        normalized_rows.append((combined, symbol))
    if normalized_rows != sorted(
        normalized_rows,
        key=lambda row: (-row[0], row[1].encode("ascii")),
    ):
        raise IntelligenceError("Factor research rank ordering is invalid")
    if policy is not None:
        policy_artifact = validate_daily_research_policy(policy)
        require_artifact_ref(payload["policy_ref"], policy_artifact, label="rank.policy_ref")
        if len(rows) != policy_artifact["payload"]["pool_policy"]["pool_size"]:
            raise IntelligenceError("Factor research rank pool size differs from policy")
        weights = {
            row["factor_alias"]: Decimal(row["weight"])
            for row in policy_artifact["payload"]["factor_rows"]
        }
        for row in rows:
            expected = sum(
                (Decimal(row["factor_percentiles"][alias]) * weights[alias] for alias in weights),
                Decimal("0"),
            )
            if row["combined_percentile"] != decimal_text(expected):
                raise IntelligenceError("combined percentile does not replay policy")
    return normalized


def project_tushare_industry_source(
    *,
    taxonomy_plan: Mapping[str, Any],
    taxonomy_capture: Mapping[str, Any],
    membership_plan: Mapping[str, Any],
    membership_capture: Mapping[str, Any],
    partition_documents: Sequence[Mapping[str, Any]],
    companies: Sequence[str],
    as_of: str,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    company_set = _codes(companies, label="companies")
    tax_plan = validate_industry_taxonomy_execution_plan(taxonomy_plan)
    tax_capture = validate_industry_taxonomy_capture(taxonomy_capture, plan=tax_plan)
    member_plan = validate_industry_membership_execution_plan(
        membership_plan, taxonomy_plan=tax_plan, taxonomy_capture=tax_capture
    )
    partitions = [
        validate_industry_membership_partition_capture(
            row, membership_plan=member_plan, taxonomy_plan=tax_plan, taxonomy_capture=tax_capture
        )
        for row in partition_documents
    ]
    member_capture = validate_industry_membership_capture(
        membership_capture,
        membership_plan=member_plan,
        taxonomy_plan=tax_plan,
        taxonomy_capture=tax_capture,
        partition_documents=partitions,
    )
    if member_capture["status"] != "COMPLETE" or member_capture["timestamp"] > cutoff:
        raise IntelligenceError("Industry capture is incomplete or future-dated")
    taxonomy_l3 = {
        row["index_code"]
        for partition in tax_capture["partition_rows"]
        if partition["level"] == "L3"
        for row in partition["rows"]
    }
    session = cutoff[:10].replace("-", "")
    active_by_company: dict[str, set[str]] = {company: set() for company in company_set}
    for partition in partitions:
        for row in partition["rows"]:
            company = row["ts_code"]
            if company not in active_by_company:
                continue
            in_date = row["in_date"]
            out_date = row["out_date"]
            if row["l3_code"] not in taxonomy_l3 or type(in_date) is not str:
                raise IntelligenceError("Industry membership taxonomy binding is invalid")
            if in_date <= session and (out_date in {None, ""} or session <= out_date):
                active_by_company[company].add(f"TUSHARE_SW2021:{row['l3_code']}")
    company_rows = []
    blockers: list[str] = []
    for company in company_set:
        memberships = sorted(active_by_company[company], key=lambda item: item.encode("ascii"))
        status = (
            "AVAILABLE" if len(memberships) == 1 else "UNMAPPED" if not memberships else "AMBIGUOUS"
        )
        if status != "AVAILABLE":
            blockers.append(f"INDUSTRY_{status}:{company}")
        company_rows.append(
            {"company_code": company, "industry_ids": memberships, "status": status}
        )
    source_refs = [
        tushare_content_ref(tax_plan, identity_field="plan_id"),
        tushare_content_ref(tax_capture, identity_field="capture_id"),
        tushare_content_ref(member_plan, identity_field="membership_plan_id"),
        tushare_content_ref(member_capture, identity_field="capture_id"),
    ]
    return build_artifact(
        kind=INDUSTRY_PROJECTION_KIND,
        identity_field="projection_id",
        identity=business_identity(
            kind=INDUSTRY_PROJECTION_KIND,
            identity_inputs={
                "as_of": cutoff,
                "company_set_sha256": _company_set_sha256(company_set),
            },
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(blockers),
            "company_rows": company_rows,
            "company_set_sha256": _company_set_sha256(company_set),
            "provider": "TUSHARE_SW2021",
            "source_refs": _ref_rows(source_refs, label="source_refs"),
            "status": "READY" if not blockers else "PARTIAL",
        },
    )


def project_tushare_theme_source(
    *,
    dc_plan: Mapping[str, Any],
    dc_capture: Mapping[str, Any],
    dc_partitions: Sequence[Mapping[str, Any]],
    tdx_plan: Mapping[str, Any] | None,
    tdx_capture: Mapping[str, Any] | None,
    tdx_partitions: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | bytes,
    as_of: str,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    policy_artifact = validate_daily_research_policy(policy)
    if policy_artifact["payload"]["technology_policy_state"] != "ACTIVE":
        raise IntelligenceError("technology policy is not configured")
    dc_valid_plan = validate_theme_provider_execution_plan(dc_plan)
    daily_date = cutoff[:10].replace("-", "")
    if dc_valid_plan["provider"] != "TUSHARE_DC" or dc_valid_plan["trade_date"] != daily_date:
        raise IntelligenceError("Theme DC provider or date differs from daily cutoff")
    dc_valid_capture = validate_theme_provider_capture(
        dc_capture, plan=dc_valid_plan, partition_documents=dc_partitions
    )
    if dc_valid_capture["timestamp"] > cutoff:
        raise IntelligenceError("Theme DC capture is future-dated")
    dc_projection = project_theme_provider_capture(
        plan=dc_valid_plan, capture=dc_valid_capture, partition_documents=dc_partitions
    )
    fallback = derive_tdx_fallback_company_keyset(
        dc_plan=dc_valid_plan, dc_capture=dc_valid_capture, dc_partition_documents=dc_partitions
    )
    tdx_projection = None
    source_refs = [
        tushare_content_ref(dc_valid_plan, identity_field="plan_id"),
        tushare_content_ref(dc_valid_capture, identity_field="capture_id"),
    ]
    if fallback:
        if tdx_plan is None or tdx_capture is None:
            raise IntelligenceError("Theme TDX fallback capture is required")
        tdx_valid_plan = validate_theme_provider_execution_plan(tdx_plan)
        if (
            tdx_valid_plan["provider"] != "TUSHARE_TDX"
            or tdx_valid_plan["trade_date"] != dc_valid_plan["trade_date"]
            or tdx_valid_plan["company_keyset"] != fallback
        ):
            raise IntelligenceError("Theme TDX company keyset differs from registered fallback")
        tdx_valid_capture = validate_theme_provider_capture(
            tdx_capture, plan=tdx_valid_plan, partition_documents=tdx_partitions
        )
        if tdx_valid_capture["timestamp"] > cutoff:
            raise IntelligenceError("Theme TDX capture is future-dated")
        tdx_projection = project_theme_provider_capture(
            plan=tdx_valid_plan, capture=tdx_valid_capture, partition_documents=tdx_partitions
        )
        source_refs.extend(
            [
                tushare_content_ref(tdx_valid_plan, identity_field="plan_id"),
                tushare_content_ref(tdx_valid_capture, identity_field="capture_id"),
            ]
        )
        tdx_registry_ids = {
            f"TUSHARE_TDX:{row['ts_code']}" for row in tdx_projection["registry_rows"]
        }
        # ``tdx_member`` returns every TDX identity attached to one company,
        # including industry and broad-index rows.  Theme v2 admits only IDs
        # present in the exact same-date concept registry; all other rows stay
        # sealed in raw capture evidence but cannot enter membership or voting.
    technologies = set(policy_artifact["payload"]["technology_theme_ids"])
    company_rows = []
    blockers: list[str] = []
    for company in dc_valid_plan["company_keyset"]:
        source = dc_projection["membership_captures"][company]
        provider = "TUSHARE_DC"
        if company in fallback:
            if tdx_projection is None:
                raise IntelligenceError("Theme fallback projection is missing")
            source = tdx_projection["membership_captures"][company]
            provider = "TUSHARE_TDX"
        if source["status"] != "COMPLETE":
            status = "UNMAPPED"
            theme_ids: list[str] = []
        else:
            prefix = provider + ":"
            theme_ids = sorted(
                {prefix + row["ts_code"] for row in source["rows"]},
                key=lambda item: item.encode("ascii"),
            )
            if provider == "TUSHARE_TDX":
                theme_ids = [theme_id for theme_id in theme_ids if theme_id in tdx_registry_ids]
            status = "NO_MEMBERSHIP" if not theme_ids else "MEMBERSHIP_ONLY"
        matched = sorted(set(theme_ids) & technologies, key=lambda item: item.encode("ascii"))
        if status == "UNMAPPED":
            blockers.append(f"THEME_UNMAPPED:{company}")
        company_rows.append(
            {
                "company_code": company,
                "provider": provider,
                "status": status,
                "technology_theme_ids": matched,
                "theme_ids": theme_ids,
            }
        )
    companies = _codes(dc_valid_plan["company_keyset"], label="theme companies")
    trade_date = dc_valid_plan["trade_date"]
    if daily_date != trade_date:
        raise IntelligenceError("Theme capture date differs from daily cutoff")
    return build_artifact(
        kind=THEME_PROJECTION_KIND,
        identity_field="projection_id",
        identity=business_identity(
            kind=THEME_PROJECTION_KIND,
            identity_inputs={"policy_id": policy_artifact["artifact_id"], "trade_date": trade_date},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(blockers),
            "company_rows": company_rows,
            "company_set_sha256": _company_set_sha256(companies),
            "fallback_company_keyset": fallback,
            "policy_ref": artifact_ref(policy_artifact),
            "source_refs": _ref_rows(source_refs, label="source_refs"),
            "status": "READY" if not blockers else "PARTIAL",
            "trade_date": trade_date,
        },
    )


def compile_daily_intelligence(  # noqa: C901 - explicit cross-domain research closure
    *,
    as_of: str,
    strategy_id: str,
    rank: Mapping[str, Any] | bytes,
    policy: Mapping[str, Any] | bytes,
    industry_projection: Mapping[str, Any] | bytes | None,
    theme_projection: Mapping[str, Any] | bytes | None,
    exposure_evidence: Sequence[Mapping[str, Any] | bytes] = (),
    fundamental_frame: Any | None = None,
    fundamental_source: Mapping[str, Any] | None = None,
    market_risk_evidence: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    cutoff = timestamp(as_of, label="as_of")
    strategy = identifier(strategy_id, label="strategy_id")
    policy_artifact = validate_daily_research_policy(policy)
    if policy_artifact["payload"]["technology_policy_state"] != "ACTIVE":
        raise IntelligenceError("technology policy is not configured")
    if policy_artifact["payload"]["strategy_id"] != strategy:
        raise IntelligenceError("daily compiler strategy differs from policy")
    rank_artifact = validate_factor_research_rank(rank, policy=policy_artifact)
    rank_payload = rank_artifact["payload"]
    require_artifact_ref(rank_payload["policy_ref"], policy_artifact, label="rank.policy_ref")
    if (
        rank_payload["as_of"] != cutoff
        or rank_payload["strategy_id"] != strategy
        or rank_payload["status"] != "READY"
    ):
        raise IntelligenceError("daily compiler rank binding differs")
    pool_rows = rank_payload["pool_rows"]
    if type(pool_rows) is not list or not pool_rows:
        raise IntelligenceError("daily compiler pool is empty")
    companies = [company_code(row["symbol"]) for row in pool_rows]
    if len(companies) != len(set(companies)):
        raise IntelligenceError("daily compiler pool contains duplicate companies")
    company_set_sha = _company_set_sha256(sorted(companies, key=lambda item: item.encode("ascii")))
    industry_artifact = (
        None
        if industry_projection is None
        else artifact_payload(industry_projection, expected_kind=INDUSTRY_PROJECTION_KIND)[0]
    )
    theme_artifact = (
        None
        if theme_projection is None
        else artifact_payload(theme_projection, expected_kind=THEME_PROJECTION_KIND)[0]
    )
    for label, artifact in (
        ("industry projection", industry_artifact),
        ("theme projection", theme_artifact),
    ):
        if artifact is not None:
            require_no_future(artifact, as_of=cutoff, label=label)
            if artifact["payload"]["company_set_sha256"] != company_set_sha:
                raise IntelligenceError(f"{label} company keyset differs from Quant pool")
    if theme_artifact is not None:
        require_artifact_ref(
            theme_artifact["payload"]["policy_ref"], policy_artifact, label="theme.policy_ref"
        )
    exposure_artifact = None
    exposure_evidence_by_company: dict[str, dict[str, Any]] = {}
    if strategy == "aggressive_tech_manufacturing":
        from .storage import approved_theme_policy_v2
        from .daily_evidence import build_source_bound_economic_exposure_projection
        from .theme_governance import build_unverified_economic_exposure_projection

        if policy_artifact != approved_theme_policy_v2():
            raise IntelligenceError("aggressive Theme compiler requires approved v2 policy")
        if theme_artifact is None:
            raise IntelligenceError("aggressive Theme compiler requires source projection")
        if exposure_evidence:
            exposure_artifact, exposure_evidence_by_company = (
                build_source_bound_economic_exposure_projection(
                    as_of=cutoff,
                    daily_policy=policy_artifact,
                    theme_projection=theme_artifact,
                    evidence=exposure_evidence,
                )
            )
        else:
            exposure_artifact = build_unverified_economic_exposure_projection(
                as_of=cutoff,
                daily_policy=policy_artifact,
                theme_projection=theme_artifact,
            )
    industry_rows = (
        {}
        if industry_artifact is None
        else {row["company_code"]: row for row in industry_artifact["payload"]["company_rows"]}
    )
    theme_rows = (
        {}
        if theme_artifact is None
        else {row["company_code"]: row for row in theme_artifact["payload"]["company_rows"]}
    )
    artifacts: list[dict[str, Any]] = [policy_artifact, rank_artifact]
    if industry_artifact is not None:
        artifacts.append(industry_artifact)
    if theme_artifact is not None:
        artifacts.append(theme_artifact)
    if exposure_artifact is not None:
        artifacts.append(exposure_artifact)
    artifacts.extend(exposure_evidence_by_company.values())
    market_risk_artifact = None
    if market_risk_evidence is not None:
        from .daily_evidence import validate_market_risk_evidence

        market_risk_artifact = validate_market_risk_evidence(market_risk_evidence)
        require_no_future(market_risk_artifact, as_of=cutoff, label="market risk evidence")
        artifacts.append(market_risk_artifact)
    exposure_rows = (
        {}
        if exposure_artifact is None
        else {row["company_code"]: row for row in exposure_artifact["payload"]["company_rows"]}
    )
    industry_assessments: dict[str, dict[str, Any]] = {}
    for pool_row in pool_rows:
        company = pool_row["symbol"]
        industry_row = industry_rows.get(company)
        memberships = []
        if industry_row is not None and industry_row["status"] == "AVAILABLE":
            memberships = [
                {
                    "available_at": industry_artifact["created_at"],
                    "effective_from": industry_artifact["payload"]["as_of"],
                    "exposure": "1",
                    "industry_id": industry_row["industry_ids"][0],
                    "provider": industry_artifact["payload"]["provider"],
                    "retired": False,
                }
            ]
        industry = assess_industry(
            company=company,
            memberships=memberships,
            provider_precedence=["TUSHARE_SW2021"],
            as_of=cutoff,
        )
        artifacts.append(industry)
        industry_assessments[company] = industry

    theme_assessments: dict[str, dict[str, Any]] = {}
    if exposure_artifact is not None:
        from .daily_evidence import theme_assessment_from_exposure

        for company, row in exposure_rows.items():
            evidence_artifact = exposure_evidence_by_company.get(company)
            if row["economic_exposure_state"] == "UNVERIFIED" or evidence_artifact is None:
                continue
            assessment = theme_assessment_from_exposure(
                row=row,
                evidence=evidence_artifact,
                as_of=cutoff,
            )
            theme_assessments[company] = assessment
            artifacts.append(assessment)

    fundamental_assessments: dict[str, dict[str, Any]] = {}
    fundamental_source_artifacts: list[dict[str, Any]] = []
    if fundamental_frame is not None:
        from .daily_evidence import build_fundamental_assessments_from_frame

        if type(fundamental_source) is not dict or set(fundamental_source) != {
            "available_at",
            "path",
            "sha256",
        }:
            raise IntelligenceError("Fundamental source binding is invalid")
        fundamental_assessments, fundamental_source_artifacts = (
            build_fundamental_assessments_from_frame(
                frame=fundamental_frame,
                companies=companies,
                source_path=fundamental_source["path"],
                source_sha256=fundamental_source["sha256"],
                source_available_at=fundamental_source["available_at"],
                as_of=cutoff,
                industry_assessments=industry_assessments,
                theme_assessments=theme_assessments,
            )
        )
        artifacts.extend(fundamental_source_artifacts)
        artifacts.extend(fundamental_assessments.values())

    decisions: list[dict[str, Any]] = []
    decision_blockers: list[str] = []
    per_company: list[dict[str, Any]] = []
    for pool_row in pool_rows:
        company = pool_row["symbol"]
        industry = industry_assessments[company]
        theme_row = theme_rows.get(company)
        tech_match = bool(theme_row and theme_row["technology_theme_ids"])
        theme_source_available = theme_row is not None and theme_row["status"] != "UNMAPPED"
        evidence_refs = [artifact_ref(policy_artifact), artifact_ref(rank_artifact)]
        for artifact in (industry_artifact, theme_artifact):
            if artifact is not None:
                evidence_refs.append(artifact_ref(artifact))
        if exposure_artifact is not None:
            evidence_refs.append(artifact_ref(exposure_artifact))
        theme_assessment = theme_assessments.get(company)
        fundamental_assessment = fundamental_assessments.get(company)
        if theme_assessment is not None:
            evidence_refs.append(artifact_ref(theme_assessment))
        if fundamental_assessment is not None:
            evidence_refs.append(artifact_ref(fundamental_assessment))
        if market_risk_artifact is not None:
            evidence_refs.append(artifact_ref(market_risk_artifact))
        risk_codes = set(
            [] if theme_assessment is None else theme_assessment["payload"]["hard_veto_codes"]
        )
        if market_risk_artifact is not None:
            risk_codes.update(market_risk_artifact["payload"]["hard_risk_codes"])
        hypothesis_status = (
            "VALID"
            if theme_assessment is not None
            and fundamental_assessment is not None
            and fundamental_assessment["payload"]["status"] in {"COMPLETE", "PARTIAL"}
            else "UNTESTED"
        )
        context = build_decision_context(
            company=company,
            as_of=cutoff,
            hypothesis_status=hypothesis_status,
            risk_status="AVAILABLE" if market_risk_artifact is not None else "UNAVAILABLE",
            evidence_refs=evidence_refs,
            industry_assessment=industry,
            theme_assessment=theme_assessment,
            fundamental_assessment=fundamental_assessment,
            quant_ref=artifact_ref(rank_artifact),
            risk_codes=sorted(risk_codes, key=lambda value: value.encode("ascii")),
        )
        decision = make_investment_decision(
            context=context,
            deterministic_percentile=pool_row["combined_percentile"],
            thresholds=policy_artifact["payload"]["decision_thresholds"],
            as_of=cutoff,
        )
        artifacts.extend([context, decision])
        decisions.append(decision)
        state = decision["payload"]["state"]
        if state == "INSUFFICIENT_EVIDENCE":
            decision_blockers.extend(decision["payload"]["blocker_codes"])
        per_company.append(
            {
                "company_code": company,
                "decision_ref": artifact_ref(decision),
                "economic_exposure_ref": (
                    None if exposure_artifact is None else artifact_ref(exposure_artifact)
                ),
                "economic_exposure_state": (
                    None
                    if company not in exposure_rows
                    else exposure_rows[company]["economic_exposure_state"]
                ),
                "industry_ref": artifact_ref(industry),
                "technology_gate": (
                    "PASS"
                    if tech_match
                    else "REJECT_NON_TECH" if theme_source_available else "UNAVAILABLE"
                ),
                "theme_assessment_ref": (
                    None if theme_assessment is None else artifact_ref(theme_assessment)
                ),
                "fundamental_assessment_ref": (
                    None if fundamental_assessment is None else artifact_ref(fundamental_assessment)
                ),
                "state": state,
            }
        )
    research_portfolio = None
    if market_risk_artifact is not None:
        from .portfolio import construct_research_portfolio

        research_portfolio = construct_research_portfolio(
            strategy_id=strategy,
            decisions=decisions,
            candidate_data={},
            policy={
                "cash_floor": "1",
                "minimum_adv_cny": "0",
                "per_security_cap": "0",
                "target_gross": "0",
                "target_positions": 5,
                "turnover_cap": "0",
            },
            as_of=cutoff,
            market_risk={
                "blocker_codes": [],
                "effective_cash_floor": "1",
                "effective_gross_cap": "0",
                "effective_security_cap": "0",
                "hard_veto_codes": market_risk_artifact["payload"]["hard_risk_codes"],
                "status": "AVAILABLE",
            },
        )
        artifacts.append(research_portfolio)
    request = forward(
        {
            "as_of": cutoff,
            "input_refs": [artifact_ref(policy_artifact), artifact_ref(rank_artifact)]
            + ([] if industry_artifact is None else [artifact_ref(industry_artifact)])
            + ([] if theme_artifact is None else [artifact_ref(theme_artifact)]),
            "stages": list(_RESEARCH_STAGES),
            "strategy_id": strategy,
        },
        created_at=cutoff,
    )
    stage_results: dict[str, dict[str, Any]] = {
        stage: {"blocker_codes": [], "output_refs": [], "status": "COMPLETE"}
        for stage in _RESEARCH_STAGES
    }
    stage_results["quant_rank"]["output_refs"] = [artifact_ref(rank_artifact)]
    stage_results["industry"]["output_refs"] = [
        artifact_ref(industry)
        for industry in artifacts
        if industry.get("kind") == "industry_assessment"
    ]
    stage_results["theme_gate"]["output_refs"] = (
        []
        if theme_artifact is None
        else [artifact_ref(theme_artifact)]
        + ([] if exposure_artifact is None else [artifact_ref(exposure_artifact)])
    )
    if exposure_artifact is not None and exposure_artifact["payload"]["blocker_codes"]:
        stage_results["theme_gate"] = {
            "blocker_codes": exposure_artifact["payload"]["blocker_codes"],
            "output_refs": stage_results["theme_gate"]["output_refs"],
            "status": "BLOCKED",
        }
    if fundamental_frame is None:
        stage_results["fundamental"] = {
            "blocker_codes": ["FUNDAMENTAL_DETERMINISTIC_PRODUCER_UNAVAILABLE"],
            "output_refs": [],
            "status": "NOT_RUN",
        }
    else:
        stage_results["fundamental"] = {
            "blocker_codes": [],
            "output_refs": [
                artifact_ref(assessment) for assessment in fundamental_assessments.values()
            ],
            "status": "COMPLETE",
        }
    stage_results["decision"]["output_refs"] = [artifact_ref(decision) for decision in decisions]
    if decision_blockers:
        stage_results["decision"] = {
            "blocker_codes": sorted(set(decision_blockers)),
            "output_refs": stage_results["decision"]["output_refs"],
            "status": "BLOCKED",
        }
    evaluation = evaluate(request, stage_results=stage_results, evaluated_at=cutoff)
    bundle = compile_evidence(evaluation, evidence=artifacts, compiled_at=cutoff)
    inline_artifacts = sorted(
        [*artifacts, request, evaluation, bundle],
        key=lambda artifact: (
            artifact["kind"].encode("ascii"),
            artifact["artifact_id"].encode("utf-8"),
        ),
    )
    result = {
        "artifacts": inline_artifacts,
        "authority": dict(NO_AUTHORITY),
        "as_of": cutoff,
        "decisions": per_company,
        "evidence_bundle": bundle,
        "evaluation": evaluation,
        "production": False,
        "research_portfolio": research_portfolio,
        "research_only": True,
        "run_state": "INACTIVE",
        "status": "COMPLETE" if not decision_blockers else "PARTIAL",
        "strategy_id": strategy,
    }
    if len(canonical_json_bytes(result)) > 8 * 1024 * 1024:
        raise IntelligenceError("daily compiler output exceeds the fixed byte ceiling")
    return result


__all__ = [
    "build_daily_research_policy",
    "build_factor_research_rank",
    "compile_daily_intelligence",
    "project_tushare_industry_source",
    "project_tushare_theme_source",
    "rank_factor_signals",
    "validate_daily_research_policy",
    "validate_factor_research_rank",
]
