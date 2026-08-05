"""Pinned residualization baselines for production factor definitions.

Formulaic mining builds `<primitive>_resid_existing` by regressing a primitive
on `context.existing_composite`, which `compute_existing_composite` derives from
`registry.selectable_factors()`. That couples the factor's value to the
production set it is joining: promoting anything rewrites the definition, the
recorded gate evidence stops describing what production computes, and the v4
canonical replay cannot be reproduced from market data alone.
`runtime.production_set_dependent_primitives` refuses that shape outright.

This module supplies the replayable alternative. The baseline factors are named
explicitly and bound by a content hash, so a residual is a pure function of
market data plus the spec. Pinning a factor's baseline to what it actually was
when the eight gates were evaluated keeps that evidence valid, because the
arithmetic is unchanged -- only the provenance of the baseline becomes fixed.

The residual itself is cross-sectional: production scores one date at a time, so
a baseline is a vector over symbols rather than the mining path's date-by-symbol
matrix. The 20-observation floor and the constant-baseline fallback are kept
identical to `_residualize_against_existing` so pinned values match mined ones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from quant_investor.factors.runtime import is_production_allowlisted_implementation

RESIDUAL_BASELINE_SCHEMA_VERSION: Final = "factor-residual-baseline.v1"

# Mirrors `_residualize_against_existing`: below this many jointly observed
# symbols the cross-sectional fit is not trustworthy and the date is skipped.
MIN_RESIDUAL_OBSERVATIONS: Final = 20

_BASELINE_FIELDS: Final = ("name", "implementation", "weight", "direction")


class ResidualBaselineError(ValueError):
    """Raised when a residual baseline spec is missing, malformed or unpinned."""


def _normalized_factor(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(raw.get("name") or "").strip(),
        "implementation": str(raw.get("implementation") or "").strip(),
        "weight": float(raw.get("weight") or 0.0),
        "direction": float(raw.get("direction", 1.0) or 0.0),
    }


def baseline_sha256(factors: Sequence[Mapping[str, Any]]) -> str:
    """Content hash of a baseline factor list.

    Sorted by name and normalized to floats so declaration order and int/float
    spelling cannot produce two hashes for the same baseline.
    """

    canonical = sorted(
        (_normalized_factor(item) for item in factors),
        key=lambda item: item["name"],
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_residual_baseline(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized baseline, or raise if it is not a valid pin.

    Never mutates ``spec``.
    """

    if not isinstance(spec, Mapping):
        raise ResidualBaselineError("residual baseline spec must be a mapping")
    if spec.get("schema_version") != RESIDUAL_BASELINE_SCHEMA_VERSION:
        raise ResidualBaselineError(
            f"unsupported residual baseline schema_version: "
            f"{spec.get('schema_version')!r}"
        )

    raw_factors = spec.get("factors")
    if not isinstance(raw_factors, Sequence) or isinstance(raw_factors, (str, bytes)):
        raise ResidualBaselineError("residual baseline factors must be a list")
    if not raw_factors:
        raise ResidualBaselineError("residual baseline needs at least one factor")

    factors: list[dict[str, Any]] = []
    for raw in raw_factors:
        if not isinstance(raw, Mapping):
            raise ResidualBaselineError("each residual baseline factor must be a mapping")
        item = _normalized_factor(raw)
        if not item["name"]:
            raise ResidualBaselineError("residual baseline factor is missing a name")
        if not is_production_allowlisted_implementation(item["implementation"]):
            raise ResidualBaselineError(
                f"residual baseline implementation is not allowlisted: "
                f"{item['name']}:{item['implementation']}"
            )
        if not np.isfinite(item["weight"]) or abs(item["weight"]) <= 1e-15:
            raise ResidualBaselineError(
                f"residual baseline factor needs a non-zero weight: {item['name']}"
            )
        factors.append(item)

    names = [item["name"] for item in factors]
    if len(set(names)) != len(names):
        raise ResidualBaselineError("residual baseline contains duplicate factors")

    expected = baseline_sha256(factors)
    if spec.get("baseline_sha256") != expected:
        raise ResidualBaselineError(
            f"residual baseline_sha256 does not match its factors: "
            f"expected {expected}, got {spec.get('baseline_sha256')!r}"
        )

    return {
        "schema_version": RESIDUAL_BASELINE_SCHEMA_VERSION,
        "factors": sorted(factors, key=lambda item: item["name"]),
        "baseline_sha256": expected,
    }


def rank_normalize(values: pd.Series) -> pd.Series:
    """Percentile-rank a cross-section onto [-1, 1], preserving NaN.

    Matches `compute_existing_composite`'s convention so a pinned baseline and a
    mined one agree.
    """

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.rank(pct=True).mul(2.0).sub(1.0)


def cross_sectional_residual(
    signal: pd.Series,
    baseline: pd.Series,
    *,
    min_observations: int = MIN_RESIDUAL_OBSERVATIONS,
) -> pd.Series:
    """Regress ``signal`` on ``baseline`` across symbols and return the residual.

    Reindexed to ``signal``'s symbols, so a caller never silently loses names.
    Symbols missing from either side stay NaN, and a cross-section thinner than
    ``min_observations`` yields an all-NaN result rather than an overfitted one.
    """

    y_all = pd.to_numeric(signal, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_all = pd.to_numeric(baseline, errors="coerce").replace([np.inf, -np.inf], np.nan)
    residual = pd.Series(np.nan, index=y_all.index, dtype=float)

    shared = y_all.index.intersection(x_all.index)
    y = y_all.reindex(shared)
    x = x_all.reindex(shared)
    valid = y.notna() & x.notna()
    if int(valid.sum()) < min_observations:
        return residual

    y_values = y[valid].to_numpy(dtype=float)
    x_values = x[valid].to_numpy(dtype=float)
    variance = float(np.var(x_values))
    if variance <= 1e-18:
        # A constant baseline explains only the mean; demean rather than divide
        # by zero. `_residualize_against_existing` does the same.
        fitted: Any = float(np.mean(y_values))
    else:
        beta = float(np.cov(x_values, y_values, ddof=0)[0, 1] / variance)
        alpha = float(np.mean(y_values) - beta * np.mean(x_values))
        fitted = alpha + beta * x_values

    residual.loc[valid.index[valid]] = y_values - fitted
    return residual
