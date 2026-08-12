"""
Guard for bars that arrive without a usable adj_factor.

``upsert_bars`` requires adj_factor to be present and positive on every row and
rejects the whole batch otherwise. Tushare serves a small number of daily bars
with no matching adj_factor row — across 2005 exactly one code (600018.SH),
which was enough to block the entire year from being backfilled.

Such a row cannot produce an adjusted price, so dropping it costs nothing. The
property that matters is that it is *not* dropped silently: the affected codes
have to reach the batch manifest, or the store's audit trail would claim a
completeness it does not have.

Scenarios:
  A01  a row with a null adj_factor is dropped and reported
  A02  a non-positive adj_factor is treated the same as a missing one
  A03  a clean frame is returned untouched, with no allocation of blame
  A04  usable rows survive alongside dropped ones
  A05  an empty frame is handled without error
  A06  a frame with no adj_factor column is left alone
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backfill_cn_history.py"
_spec = importlib.util.spec_from_file_location("backfill_cn_history", _SCRIPT)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
sys.modules["backfill_cn_history"] = _module
_spec.loader.exec_module(_module)

_drop_unusable_adj_factor = _module._drop_unusable_adj_factor


def _frame(adj_factors: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": [f"60000{i}.SH" for i in range(len(adj_factors))],
        "trade_date": ["20050104"] * len(adj_factors),
        "close": [10.0] * len(adj_factors),
        "adj_factor": adj_factors,
    })


def test_A01_null_adj_factor_is_dropped_and_reported():
    kept, dropped = _drop_unusable_adj_factor(_frame([1.0, None]))

    assert len(kept) == 1
    assert dropped == ["600001.SH"], "the code must be named, not merely counted"


def test_A02_non_positive_adj_factor_is_also_unusable():
    kept, dropped = _drop_unusable_adj_factor(_frame([1.0, 0.0, -1.0]))

    assert len(kept) == 1
    assert dropped == ["600001.SH", "600002.SH"]


def test_A03_a_clean_frame_is_untouched():
    frame = _frame([1.0, 2.0, 3.0])
    kept, dropped = _drop_unusable_adj_factor(frame)

    assert dropped == []
    pd.testing.assert_frame_equal(kept, frame)


def test_A04_usable_rows_survive_alongside_dropped_ones():
    kept, dropped = _drop_unusable_adj_factor(_frame([1.5, None, 2.5]))

    assert list(kept["ts_code"]) == ["600000.SH", "600002.SH"]
    assert list(kept["adj_factor"]) == [1.5, 2.5]
    assert dropped == ["600001.SH"]


def test_A05_empty_frame_is_handled():
    kept, dropped = _drop_unusable_adj_factor(pd.DataFrame())

    assert kept.empty
    assert dropped == []


def test_A06_frame_without_the_column_is_left_alone():
    frame = pd.DataFrame({"ts_code": ["600000.SH"], "close": [10.0]})
    kept, dropped = _drop_unusable_adj_factor(frame)

    assert dropped == []
    pd.testing.assert_frame_equal(kept, frame)
