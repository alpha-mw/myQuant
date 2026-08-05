"""Combinatorial purged cross-validation for overlapping-label factor tests.

A daily RankIC measured against a 30-session forward return shares its label
window with the 29 observations around it.  Contiguous folds leak across that
window in both directions, which is why the previous out-of-sample diagnostic
could report a positive ratio of 1.0 for factors that do not survive an honest
overlap correction.

This module builds the standard treatment: split the calendar into blocks, test
on every combination of blocks, purge training sessions whose label window
reaches into a test block, and embargo a further stretch immediately after each
test block.  Testing on combinations rather than one contiguous tail yields many
backtest paths instead of a single one, which is what a deflated-Sharpe or
overfitting-probability statistic needs downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

CPCV_EVIDENCE_SCHEMA = "factor-cpcv-evidence.v1"
DEFAULT_BLOCK_COUNT = 10
DEFAULT_TEST_BLOCK_COUNT = 2
# Gate 7 requires at least a 30-session purge and exactly a 30-session embargo.
DEFAULT_PURGE_DAYS = 30
DEFAULT_EMBARGO_DAYS = 30
# Below this share of paths with positive test IC, the factor's edge is carried
# by a stretch of the calendar rather than by the calendar as a whole.
DATE_RANGE_ROBUSTNESS_FLOOR = 0.55


@dataclass(frozen=True)
class CPCVSplit:
    """One train/test path with its purge and embargo already applied."""

    test_dates: tuple[pd.Timestamp, ...]
    train_dates: tuple[pd.Timestamp, ...]
    test_block_ids: tuple[int, ...]
    purged_date_count: int
    embargoed_date_count: int


def _block_bounds(
    total: int,
    block_count: int,
) -> tuple[tuple[int, int], ...]:
    edges = np.linspace(0, total, block_count + 1).astype(int)
    return tuple(
        (int(edges[index]), int(edges[index + 1]) - 1)
        for index in range(block_count)
    )


def build_cpcv_splits(
    dates: Sequence[pd.Timestamp],
    *,
    block_count: int = DEFAULT_BLOCK_COUNT,
    test_block_count: int = DEFAULT_TEST_BLOCK_COUNT,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[CPCVSplit, ...]:
    """Return every combinatorial purged split of ``dates``."""

    ordered = [pd.Timestamp(date).normalize() for date in dates]
    ordered = sorted(set(ordered))
    total = len(ordered)
    if block_count < 2 or test_block_count < 1:
        raise ValueError("cpcv needs at least two blocks and one test block")
    if test_block_count >= block_count:
        raise ValueError("cpcv needs at least one training block")
    if total < block_count * max(purge_days, 1):
        raise ValueError(
            f"cpcv calendar too short: {total} sessions for {block_count} blocks"
        )

    bounds = _block_bounds(total, block_count)
    splits: list[CPCVSplit] = []
    for test_blocks in combinations(range(block_count), test_block_count):
        test_positions: set[int] = set()
        excluded: set[int] = set()
        for block_id in test_blocks:
            first, last = bounds[block_id]
            test_positions.update(range(first, last + 1))
            # A label opened up to ``purge_days`` before the block still
            # resolves inside it, and the sessions just after the block are
            # still correlated with it.
            excluded.update(range(first - purge_days, last + 1))
            excluded.update(range(last + 1, last + 1 + embargo_days))
        train_positions = [
            position
            for position in range(total)
            if position not in excluded
        ]
        purged = len(
            {
                position
                for position in excluded
                if position not in test_positions
                and any(
                    bounds[block][0] - purge_days
                    <= position
                    < bounds[block][0]
                    for block in test_blocks
                )
            }
        )
        embargoed = len(
            {
                position
                for position in excluded
                if position not in test_positions
                and any(
                    bounds[block][1]
                    < position
                    <= bounds[block][1] + embargo_days
                    for block in test_blocks
                )
            }
        )
        splits.append(
            CPCVSplit(
                test_dates=tuple(
                    ordered[position] for position in sorted(test_positions)
                ),
                train_dates=tuple(
                    ordered[position] for position in train_positions
                ),
                test_block_ids=tuple(test_blocks),
                purged_date_count=purged,
                embargoed_date_count=embargoed,
            )
        )
    return tuple(splits)


def cpcv_path_evidence(
    ic_by_date: pd.Series,
    splits: Sequence[CPCVSplit],
    *,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> dict[str, Any]:
    """Summarise one factor's IC across every CPCV path."""

    empty = {
        "schema": CPCV_EVIDENCE_SCHEMA,
        "path_count": 0,
        "positive_path_ratio": 0.0,
        "mean_path_ic": 0.0,
        "path_ic_std": 0.0,
        "min_path_ic": 0.0,
        "max_path_ic": 0.0,
        "purge_days": int(purge_days),
        "embargo_days": int(embargo_days),
        "date_range_robustness": False,
        "evidence_hash": "",
    }
    series = pd.Series(ic_by_date, dtype=float).dropna()
    if series.empty or not splits:
        return empty
    series.index = pd.DatetimeIndex(series.index).normalize()

    path_ics: list[float] = []
    for split in splits:
        observed = series.reindex(pd.DatetimeIndex(split.test_dates)).dropna()
        if observed.empty:
            continue
        path_ics.append(float(observed.mean()))
    if not path_ics:
        return empty

    values = np.asarray(path_ics, dtype=float)
    positive_path_ratio = float((values > 0.0).mean())
    payload = {
        "schema": CPCV_EVIDENCE_SCHEMA,
        "purge_days": int(purge_days),
        "embargo_days": int(embargo_days),
        "block_ids": [list(split.test_block_ids) for split in splits],
        "path_ics": [round(value, 12) for value in path_ics],
    }
    evidence_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema": CPCV_EVIDENCE_SCHEMA,
        "path_count": len(path_ics),
        "positive_path_ratio": positive_path_ratio,
        "mean_path_ic": float(values.mean()),
        "path_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min_path_ic": float(values.min()),
        "max_path_ic": float(values.max()),
        "purge_days": int(purge_days),
        "embargo_days": int(embargo_days),
        "date_range_robustness": bool(
            positive_path_ratio >= DATE_RANGE_ROBUSTNESS_FLOOR
        ),
        "evidence_hash": evidence_hash,
    }
