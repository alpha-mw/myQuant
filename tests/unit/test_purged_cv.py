from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.purged_cv import (
    CPCV_EVIDENCE_SCHEMA,
    build_cpcv_splits,
    cpcv_path_evidence,
)

DATES = tuple(pd.bdate_range("2021-01-04", periods=600))


def test_cpcv_builds_every_test_block_pair() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)

    # C(10, 2) = 45 paths.
    assert len(splits) == 45
    assert all(split.test_dates for split in splits)
    assert len({split.test_dates for split in splits}) == 45


def test_cpcv_purges_the_label_horizon_around_every_test_block() -> None:
    splits = build_cpcv_splits(
        DATES, block_count=10, test_block_count=1, purge_days=30, embargo_days=0
    )
    split = splits[5]
    test_positions = {DATES.index(date) for date in split.test_dates}
    train_positions = {DATES.index(date) for date in split.train_dates}
    first_test = min(test_positions)
    last_test = max(test_positions)

    # A training label starting within 30 sessions before the test block still
    # resolves inside it, so it must be gone.
    assert not train_positions & set(range(first_test - 30, last_test + 1))
    assert min(train_positions) < first_test - 30 or max(train_positions) > last_test


def test_cpcv_embargoes_sessions_after_every_test_block() -> None:
    splits = build_cpcv_splits(
        DATES, block_count=10, test_block_count=1, purge_days=30, embargo_days=30
    )
    split = splits[3]
    test_positions = {DATES.index(date) for date in split.test_dates}
    train_positions = {DATES.index(date) for date in split.train_dates}
    last_test = max(test_positions)

    assert not train_positions & set(range(last_test + 1, last_test + 31))
    assert split.embargoed_date_count > 0
    assert split.purged_date_count > 0


def test_cpcv_train_and_test_never_overlap() -> None:
    splits = build_cpcv_splits(DATES, block_count=8, test_block_count=2)

    for split in splits:
        assert not set(split.test_dates) & set(split.train_dates)


def test_cpcv_rejects_a_calendar_too_short_to_split() -> None:
    with pytest.raises(ValueError):
        build_cpcv_splits(DATES[:20], block_count=10, test_block_count=2)


def test_cpcv_path_evidence_counts_positive_paths() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)
    ic = pd.Series(0.05, index=pd.DatetimeIndex(DATES))

    evidence = cpcv_path_evidence(ic, splits)

    assert evidence["schema"] == CPCV_EVIDENCE_SCHEMA
    assert evidence["path_count"] == 45
    assert evidence["positive_path_ratio"] == pytest.approx(1.0)
    assert evidence["mean_path_ic"] == pytest.approx(0.05)
    assert len(evidence["evidence_hash"]) == 64


def test_cpcv_path_evidence_is_negative_for_a_negative_factor() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)
    ic = pd.Series(-0.05, index=pd.DatetimeIndex(DATES))

    evidence = cpcv_path_evidence(ic, splits)

    assert evidence["positive_path_ratio"] == pytest.approx(0.0)
    assert evidence["mean_path_ic"] == pytest.approx(-0.05)


def test_cpcv_path_evidence_flags_a_single_block_carrying_the_factor() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)
    # All the edge sits in the first sixth of the calendar.
    values = np.where(np.arange(len(DATES)) < 100, 0.30, -0.01)
    ic = pd.Series(values, index=pd.DatetimeIndex(DATES))

    evidence = cpcv_path_evidence(ic, splits)

    assert evidence["positive_path_ratio"] < 0.55
    assert evidence["date_range_robustness"] is False


def test_cpcv_path_evidence_hash_is_stable_and_content_bound() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)
    ic = pd.Series(0.05, index=pd.DatetimeIndex(DATES))
    other = pd.Series(0.06, index=pd.DatetimeIndex(DATES))

    first = cpcv_path_evidence(ic, splits)
    again = cpcv_path_evidence(ic, splits)
    different = cpcv_path_evidence(other, splits)

    assert first["evidence_hash"] == again["evidence_hash"]
    assert first["evidence_hash"] != different["evidence_hash"]


def test_cpcv_path_evidence_is_empty_without_observations() -> None:
    splits = build_cpcv_splits(DATES, block_count=10, test_block_count=2)

    evidence = cpcv_path_evidence(pd.Series(dtype=float), splits)

    assert evidence["path_count"] == 0
    assert evidence["positive_path_ratio"] == 0.0
    assert evidence["evidence_hash"] == ""
