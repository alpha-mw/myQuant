"""Exact, non-claiming bootstrap Factor set and strict signal helpers."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from collections.abc import Sequence
from enum import Enum
from typing import Any, Final

import numpy as np
import pandas as pd

from quant_investor.contracts import (
    canonical_json_bytes,
    seal_artifact,
)

from .common import (
    artifact_ref,
    business_identity,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .errors import FactorGovernanceError

BOOTSTRAP_SET_KIND: Final = "factor.bootstrap_set"
BOOTSTRAP_LANE: Final = "BOOTSTRAP"
PROSPECTIVE_LANE: Final = "PROSPECTIVE"
NOT_CLAIMED: Final = "NOT_CLAIMED"
CANONICAL_PARQUET: Final = "PARQUET"

LOW_DOLLAR_VOLUME: Final = "pv_low_dollar_volume_5d"
BLEND_W80: Final = "pv_blend_volstab19x2_mom90_amihud5_w80"
BLEND_W75_CONTROL: Final = "pv_blend_volstab19x2_mom90_amihud5_w75"


class FactorSourceRole(str, Enum):
    """Finite code-owned source roles that may enter a Factor definition."""

    EXCHANGE_CALENDAR = "EXCHANGE_CALENDAR"
    MARKET = "MARKET"
    PIT_MEMBERSHIP = "PIT_MEMBERSHIP"
    FUNDAMENTAL = "FUNDAMENTAL"


BOOTSTRAP_REQUIRED_SOURCE_ROLES: Final = tuple(
    sorted(
        (
            FactorSourceRole.EXCHANGE_CALENDAR.value,
            FactorSourceRole.MARKET.value,
            FactorSourceRole.PIT_MEMBERSHIP.value,
        ),
        key=lambda value: value.encode("utf-8"),
    )
)

_INPUT_CONTRACT: Final = {
    "source_format": CANONICAL_PARQUET,
    "csv_fallback_allowed": False,
    "amount_field": "amount",
    "amount_aliases_allowed": False,
    "close_times_volume_fallback_allowed": False,
    "adjusted_close_field": "adj_close",
    "adjusted_close_aliases_allowed": False,
    "volume_field": "vol",
    "volume_aliases_allowed": False,
}


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _definition(
    *,
    factor_id: str,
    family: str,
    formula: str,
    parameters: Mapping[str, Any],
    input_fields: list[str],
    role: str,
    selectable: bool,
    bootstrap_weight: str,
) -> dict[str, Any]:
    spec_body = {
        "factor_id": factor_id,
        "family": family,
        "formula": formula,
        "parameters": dict(parameters),
        "direction": "HIGHER_IS_BETTER",
        "input_fields": sorted(input_fields),
        "required_source_roles": list(BOOTSTRAP_REQUIRED_SOURCE_ROLES),
        "role": role,
        "selectable": selectable,
        "bootstrap_weight": bootstrap_weight,
        "producer_identity": NOT_CLAIMED,
    }
    return {"spec_id": _identity(spec_body), **spec_body}


def required_source_roles_for_factor(factor_id: str) -> tuple[str, ...]:
    """Return the immutable code-owned source roles for one bootstrap identity."""

    if factor_id not in {LOW_DOLLAR_VOLUME, BLEND_W80, BLEND_W75_CONTROL}:
        raise FactorGovernanceError("factor source dependency identity is not installed")
    return BOOTSTRAP_REQUIRED_SOURCE_ROLES


def bootstrap_factor_definitions() -> list[dict[str, Any]]:
    """Return the sole canonical bootstrap definitions and W75 control."""

    definitions = [
        _definition(
            factor_id=LOW_DOLLAR_VOLUME,
            family="liquidity",
            formula="-log(mean(amount[t-4:t]))",
            parameters={"window_open_sessions": 5},
            input_fields=["amount"],
            role="BOOTSTRAP",
            selectable=True,
            bootstrap_weight="0.500000000000",
        ),
        _definition(
            factor_id=BLEND_W80,
            family="liquidity_stability_momentum",
            formula=(
                "rank(volume_stability_19x2)*0.80+"
                "rank(rank(momentum_90)*0.60+rank(amihud_5)*0.40)*0.20"
            ),
            parameters={
                "amihud_window_open_sessions": 5,
                "inner_amihud_weight": "0.400000000000",
                "inner_momentum_weight": "0.600000000000",
                "momentum_window_open_sessions": 90,
                "outer_volume_stability_weight": "0.800000000000",
                "volume_stability_base_open_sessions": 19,
                "volume_stability_smoothing_open_sessions": 2,
            },
            input_fields=["adj_close", "amount", "vol"],
            role="BOOTSTRAP",
            selectable=True,
            bootstrap_weight="0.500000000000",
        ),
        _definition(
            factor_id=BLEND_W75_CONTROL,
            family="liquidity_stability_momentum",
            formula=(
                "rank(volume_stability_19x2)*0.75+"
                "rank(rank(momentum_90)*0.60+rank(amihud_5)*0.40)*0.25"
            ),
            parameters={
                "amihud_window_open_sessions": 5,
                "inner_amihud_weight": "0.400000000000",
                "inner_momentum_weight": "0.600000000000",
                "momentum_window_open_sessions": 90,
                "outer_volume_stability_weight": "0.750000000000",
                "volume_stability_base_open_sessions": 19,
                "volume_stability_smoothing_open_sessions": 2,
            },
            input_fields=["adj_close", "amount", "vol"],
            role="CONTROL_ONLY",
            selectable=False,
            bootstrap_weight="0.000000000000",
        ),
    ]
    return copy.deepcopy(sorted(definitions, key=lambda row: row["factor_id"].encode("utf-8")))


def _set_rows(
    definitions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "factor_id": row["factor_id"],
            "spec_id": row["spec_id"],
            "direction": row["direction"],
            "required_source_roles": list(row["required_source_roles"]),
            "weight": row["bootstrap_weight"],
            "role": row["role"],
            "selectable": row["selectable"],
        }
        for row in definitions
    ]
    factor_rows = [row for row in rows if row["selectable"] is True]
    control_rows = [row for row in rows if row["role"] == "CONTROL_ONLY"]
    return factor_rows, control_rows


def derive_active_required_source_roles(
    factor_set_document: Mapping[str, Any] | bytes,
    *,
    implementation_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Replay active policy and installed identities, then return their source-role union."""

    factor_set = validate_bootstrap_factor_set(factor_set_document)
    rows = factor_set["payload"]["factor_rows"]
    active_rows = [
        row for row in rows if row["selectable"] is True and row["weight"] != "0.000000000000"
    ]
    active_ids = [row["factor_id"] for row in active_rows]
    if active_ids != sorted(
        [LOW_DOLLAR_VOLUME, BLEND_W80], key=lambda value: value.encode("utf-8")
    ):
        raise FactorGovernanceError("active bootstrap Factor identities are not exact")
    installed_by_id = _installed_rows_by_factor_id(
        implementation_rows,
        active_ids=active_ids,
    )

    from .implementations import installed_semantic_row

    role_union: set[str] = set()
    for active_row in active_rows:
        factor_id = active_row["factor_id"]
        expected_roles = list(required_source_roles_for_factor(factor_id))
        if active_row.get("required_source_roles") != expected_roles:
            raise FactorGovernanceError("active Factor source dependency identity differs")
        expected_semantic = installed_semantic_row(factor_id)
        installed_row = installed_by_id[factor_id]
        semantic_fields = set(expected_semantic)
        installed_fields = set(installed_row)
        if installed_fields not in (
            semantic_fields,
            semantic_fields | {"implementation_component_ref"},
        ):
            raise FactorGovernanceError("installed Factor semantic fields are not exact")
        if any(installed_row.get(field) != value for field, value in expected_semantic.items()):
            raise FactorGovernanceError("installed Factor semantic identity differs")
        if installed_row.get("required_source_roles") != expected_roles:
            raise FactorGovernanceError("installed Factor source dependency identity differs")
        role_union.update(expected_roles)
    return tuple(sorted(role_union, key=lambda value: value.encode("utf-8")))


def _installed_rows_by_factor_id(
    implementation_rows: Sequence[Mapping[str, Any]],
    *,
    active_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(implementation_rows, (str, bytes)) or not isinstance(
        implementation_rows, Sequence
    ):
        raise FactorGovernanceError("installed Factor implementation rows are invalid")
    installed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in implementation_rows:
        if not isinstance(row, Mapping):
            raise FactorGovernanceError("installed Factor implementation row is invalid")
        factor_id = row.get("factor_id")
        if type(factor_id) is not str or factor_id in installed_by_id:
            raise FactorGovernanceError("installed Factor implementation identities are invalid")
        installed_by_id[factor_id] = row
    if set(installed_by_id) != set(active_ids):
        raise FactorGovernanceError("installed and active Factor identities differ")
    return installed_by_id


def _factor_set_sha256(
    *,
    definitions: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "factor_definitions": definitions,
                "factor_rows": factor_rows,
                "control_rows": control_rows,
                "weighting_method": "EQUAL_WEIGHT",
                "weight_total": "1.000000000000",
            }
        )
    ).hexdigest()


def build_bootstrap_factor_set(
    *,
    bootstrap_exception_evidence: Mapping[str, Any] | bytes,
    created_at: str,
) -> dict[str, Any]:
    """Bind exact non-authorizing evidence into the sole bootstrap set."""

    from .bootstrap_evidence import validate_bootstrap_exception_evidence

    evidence = validate_bootstrap_exception_evidence(bootstrap_exception_evidence)
    if created_at < evidence["created_at"]:
        raise FactorGovernanceError("bootstrap set predates its evidence")
    definitions = bootstrap_factor_definitions()
    factor_rows, control_rows = _set_rows(definitions)
    factor_set_sha = _factor_set_sha256(
        definitions=definitions,
        factor_rows=factor_rows,
        control_rows=control_rows,
    )
    evidence_payload = evidence["payload"]
    if evidence_payload["factor_set_sha256"] != factor_set_sha:
        raise FactorGovernanceError("bootstrap evidence factor-set binding differs")
    evidence_ref = artifact_ref(evidence)
    payload = {
        "bootstrap_set_id": business_identity(
            "bootstrap",
            {
                "admission_route": "BOOTSTRAP_EXCEPTION",
                "bootstrap_exception_evidence_ref": evidence_ref,
                "factor_set_sha256": factor_set_sha,
            },
        ),
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": NOT_CLAIMED,
        "input_contract": dict(_INPUT_CONTRACT),
        "factor_definitions": definitions,
        "factor_rows": factor_rows,
        "control_rows": control_rows,
        "bootstrap_exception_evidence_ref": evidence_ref,
        "factor_set_sha256": factor_set_sha,
        "weighting_method": "EQUAL_WEIGHT",
        "weight_total": "1.000000000000",
        "prospective_evidence_claimed": False,
        "activation_authorized": False,
    }
    return seal_artifact(BOOTSTRAP_SET_KIND, payload, created_at=created_at)


def validate_bootstrap_factor_set(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate the exact intrinsic set and its immutable evidence reference."""

    normalized = validate_governance_artifact(document, expected_kind=BOOTSTRAP_SET_KIND)
    payload = normalized["payload"]
    definitions = bootstrap_factor_definitions()
    factor_rows, control_rows = _set_rows(definitions)
    factor_set_sha = _factor_set_sha256(
        definitions=definitions,
        factor_rows=factor_rows,
        control_rows=control_rows,
    )
    evidence_ref = validate_artifact_ref(
        payload.get("bootstrap_exception_evidence_ref"),
        label="bootstrap_exception_evidence_ref",
        expected_kind="factor.bootstrap_exception_evidence",
    )
    expected = {
        "bootstrap_set_id": business_identity(
            "bootstrap",
            {
                "admission_route": "BOOTSTRAP_EXCEPTION",
                "bootstrap_exception_evidence_ref": evidence_ref,
                "factor_set_sha256": factor_set_sha,
            },
        ),
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": NOT_CLAIMED,
        "input_contract": dict(_INPUT_CONTRACT),
        "factor_definitions": definitions,
        "factor_rows": factor_rows,
        "control_rows": control_rows,
        "bootstrap_exception_evidence_ref": evidence_ref,
        "factor_set_sha256": factor_set_sha,
        "weighting_method": "EQUAL_WEIGHT",
        "weight_total": "1.000000000000",
        "prospective_evidence_claimed": False,
        "activation_authorized": False,
    }
    if payload != expected:
        raise FactorGovernanceError("bootstrap factor set does not replay exactly")
    return normalized


def _strict_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise FactorGovernanceError(f"strict bootstrap input is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _ordered_strict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FactorGovernanceError("strict bootstrap input frame is empty")
    if "trade_date" not in frame.columns:
        raise FactorGovernanceError("strict bootstrap input requires trade_date")
    ordered = frame.copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"], errors="coerce")
    if ordered["trade_date"].isna().any() or ordered["trade_date"].duplicated().any():
        raise FactorGovernanceError("strict bootstrap trade_date is invalid or duplicated")
    return ordered.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def _strict_components(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    ordered = _ordered_strict_frame(frame)
    amount = _strict_numeric(ordered, "amount")
    close = _strict_numeric(ordered, "adj_close")
    volume = _strict_numeric(ordered, "vol")

    low_window = amount.tail(5)
    low_dollar = (
        -float(np.log(float(low_window.mean())))
        if len(low_window) == 5 and low_window.notna().all() and bool((low_window > 0.0).all())
        else math.nan
    )

    minimum_base = 5
    rolling_mean = volume.rolling(19, min_periods=minimum_base).mean()
    rolling_std = volume.rolling(19, min_periods=minimum_base).std(ddof=0)
    stability = -(rolling_std / rolling_mean.where(rolling_mean > 0.0))
    smoothed = stability.rolling(2, min_periods=2).mean().dropna()
    volume_stability = float(smoothed.iloc[-1]) if not smoothed.empty else math.nan

    momentum = math.nan
    if len(close) > 90:
        base = float(close.iloc[-91])
        latest = float(close.iloc[-1])
        if math.isfinite(base) and math.isfinite(latest) and base > 0.0:
            momentum = latest / base - 1.0

    returns = close.pct_change(fill_method=None).abs()
    amihud_base = returns / amount.where(amount > 0.0)
    amihud_window = amihud_base.replace([np.inf, -np.inf], np.nan).dropna().tail(5)
    amihud = float(amihud_window.mean()) if len(amihud_window) == 5 else math.nan
    return low_dollar, volume_stability, momentum, amihud


def compute_bootstrap_signals(
    frames: Mapping[str, pd.DataFrame],
    *,
    source_format: str,
) -> dict[str, pd.Series]:
    """Compute bootstrap and W75 control signals with no input fallback."""

    if source_format != CANONICAL_PARQUET:
        raise FactorGovernanceError("bootstrap signals require canonical Parquet provenance")
    if not isinstance(frames, Mapping) or not frames:
        raise FactorGovernanceError("bootstrap signals require in-memory symbol frames")

    low: dict[str, float] = {}
    stability: dict[str, float] = {}
    momentum: dict[str, float] = {}
    amihud: dict[str, float] = {}
    for symbol in sorted((str(value) for value in frames), key=lambda value: value.encode()):
        components = _strict_components(frames[symbol])
        low[symbol], stability[symbol], momentum[symbol], amihud[symbol] = components

    low_series = pd.Series(low, dtype=float)
    stability_rank = pd.Series(stability, dtype=float).rank(pct=True)
    momentum_rank = pd.Series(momentum, dtype=float).rank(pct=True)
    amihud_rank = pd.Series(amihud, dtype=float).rank(pct=True)
    inner_rank = momentum_rank.mul(0.60).add(amihud_rank.mul(0.40)).rank(pct=True)
    return {
        LOW_DOLLAR_VOLUME: low_series,
        BLEND_W80: stability_rank.mul(0.80).add(inner_rank.mul(0.20)),
        BLEND_W75_CONTROL: stability_rank.mul(0.75).add(inner_rank.mul(0.25)),
    }


__all__ = [
    "BLEND_W75_CONTROL",
    "BLEND_W80",
    "BOOTSTRAP_LANE",
    "BOOTSTRAP_REQUIRED_SOURCE_ROLES",
    "BOOTSTRAP_SET_KIND",
    "CANONICAL_PARQUET",
    "FactorSourceRole",
    "LOW_DOLLAR_VOLUME",
    "NOT_CLAIMED",
    "PROSPECTIVE_LANE",
    "bootstrap_factor_definitions",
    "build_bootstrap_factor_set",
    "compute_bootstrap_signals",
    "derive_active_required_source_roles",
    "required_source_roles_for_factor",
    "validate_bootstrap_factor_set",
]
