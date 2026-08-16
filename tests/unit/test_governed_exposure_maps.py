from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_investor.factors import exposure_maps as exposure_module
from quant_investor.factors.exposure_maps import (
    GOVERNED_EXPOSURE_SOURCE,
    GOVERNED_INDUSTRY_POLICY,
    GOVERNED_SIZE_POLICY,
    clamp_analysis_start_to_exposure,
    governed_exposure_date_bounds,
    load_governed_exposure_maps,
)
from quant_investor.market.fundamental_generation import (
    publish_fundamental_generation,
)

SYMBOLS = ("000001.SZ", "000002.SZ", "000004.SZ", "600000.SH", "600004.SH", "600006.SH")
SECTORS = ("银行", "全国地产", "区域地产", "银行", "运输设备", "运输设备")
TRADE_DATES = ("20240102", "20240103", "20240104")


@pytest.fixture(autouse=True)
def _bind_manual_fixture_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give manual unit fixtures the binding a verified primary loader emits."""

    original = exposure_module.load_fundamental_pointer

    def load_with_binding(root: str | Path):
        pointer = original(root)
        if pointer is not None and "derivation_binding" not in pointer:
            pointer["derivation_binding"] = {
                "schema_version": "cn-fundamental-derivation-binding.v1",
                "generation_id": str(pointer.get("generation_id") or ""),
                "binding_sha256": "b" * 64,
                "mixed": False,
                "original_seam": "",
                "append_parent_cutoff": "",
                "target_cutoff": "20240104",
                "legacy_direct_reader_provenance": "verified_homogeneous",
                "binding_aware_research_ready": True,
                "homogeneous_history_ready": True,
            }
        return pointer

    monkeypatch.setattr(
        exposure_module,
        "load_fundamental_pointer",
        load_with_binding,
    )


def _daily_frame(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    trade_dates: tuple[str, ...] = TRADE_DATES,
    availability_date: str | None = None,
    market_caps: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    caps = market_caps or tuple(
        1.0e9 * (index + 1) for index in range(len(symbols))
    )
    rows = []
    for trade_date in trade_dates:
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "end_date": "20231231",
                    "availability_date": availability_date or trade_date,
                    "sector": SECTORS[index % len(SECTORS)],
                    "size_bucket": "large",
                    "total_mv_rmb": caps[index],
                }
            )
    return pd.DataFrame(rows)


def _publish(root: Path, daily: pd.DataFrame) -> None:
    publish_fundamental_generation(
        root=root,
        run_id="exposure-fixture",
        tables={
            "fundamental_period": pd.DataFrame(
                [
                    {
                        "ts_code": SYMBOLS[0],
                        "end_date": "20231231",
                        "availability_date": "20240101",
                    }
                ]
            ),
            "fundamental_daily": daily,
            "fundamental_quarantine": pd.DataFrame(
                columns=["ts_code", "quarantine_reason"]
            ),
        },
        metadata={
            "run_id": "exposure-fixture",
            "source_priority": "manual_offline_snapshot",
            "gate2_passed": True,
        },
    )


def _close_matrix(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [[10.0] * len(SYMBOLS)] * len(dates),
        index=dates,
        columns=list(SYMBOLS),
    )


def _load(root: Path, *, dates: pd.DatetimeIndex, as_of: pd.Timestamp):
    return load_governed_exposure_maps(
        mart_root=root,
        symbols=list(SYMBOLS),
        as_of=as_of,
        evaluation_dates=list(dates),
        close_by_date=_close_matrix(dates),
    )


def test_governed_exposure_reads_the_generation_without_legacy_raw_tables(
    tmp_path: Path,
) -> None:
    _publish(tmp_path, _daily_frame())
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    sectors, sizes, by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "ready"
    assert evidence["blocker"] == ""
    assert evidence["source"] == GOVERNED_EXPOSURE_SOURCE
    assert evidence["size_policy"] == GOVERNED_SIZE_POLICY
    assert evidence["industry_policy"] == GOVERNED_INDUSTRY_POLICY
    assert evidence["generation_id"] == "exposure-fixture"
    assert evidence["catalog_validated"] is True
    assert evidence["mixed_generation"] is False
    assert evidence["homogeneous_history_ready"] is True
    assert evidence["binding_aware_research_ready"] is True
    assert evidence["legacy_direct_reader_provenance"] == (
        "verified_homogeneous"
    )
    assert len(evidence["derivation_binding_sha256"]) == 64
    assert evidence["methodology_boundary_preserved"] is True
    assert sectors["000001.SZ"] == "银行"
    assert set(sizes.values()) == {"small", "mid", "large"}
    assert list(by_date.index) == list(dates)
    assert by_date.notna().all().all()
    # No raw ``daily_basic``/``dag_core_raw`` table exists in this root at all.
    assert not (tmp_path / "daily_basic").exists()
    assert not (tmp_path / "dag_core_raw").exists()


def test_governed_exposure_preserves_mixed_successor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(tmp_path, _daily_frame())
    pointer = exposure_module.load_fundamental_pointer(tmp_path)
    assert pointer is not None
    pointer["derivation_binding"] = {
        "schema_version": "cn-fundamental-derivation-binding.v1",
        "generation_id": "exposure-fixture",
        "binding_sha256": "a" * 64,
        "mixed": True,
        "original_seam": "20260806",
        "append_parent_cutoff": "20260806",
        "target_cutoff": "20260814",
        "legacy_direct_reader_provenance": "binding_required",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }
    monkeypatch.setattr(
        exposure_module,
        "load_fundamental_pointer",
        lambda _root: pointer,
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "ready"
    assert evidence["mixed_generation"] is True
    assert evidence["original_seam"] == "20260806"
    assert evidence["append_parent_cutoff"] == "20260806"
    assert evidence["fundamental_target_cutoff"] == "20260814"
    assert evidence["legacy_direct_reader_provenance"] == "binding_required"
    assert evidence["binding_aware_research_ready"] is True
    assert evidence["homogeneous_history_ready"] is False
    assert evidence["derivation_binding"] == pointer["derivation_binding"]


def test_governed_exposure_blocks_non_binding_aware_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(tmp_path, _daily_frame())
    pointer = exposure_module.load_fundamental_pointer(tmp_path)
    assert pointer is not None
    binding = dict(pointer["derivation_binding"])
    binding["mixed"] = True
    binding["binding_aware_research_ready"] = False
    binding["homogeneous_history_ready"] = False
    pointer["derivation_binding"] = binding
    monkeypatch.setattr(
        exposure_module,
        "load_fundamental_pointer",
        lambda _root: pointer,
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "blocked"
    assert evidence["blocker"] == "governed_exposure_incomplete"
    assert evidence["binding_aware_research_ready"] is False


def test_governed_exposure_size_buckets_are_cross_sectional_terciles(
    tmp_path: Path,
) -> None:
    # Absolute market caps rise tenfold, but the cross-section is unchanged, so
    # every bucket assignment must stay put.  An absolute-threshold bucketing
    # would sweep the whole cross-section into ``large``.
    daily = _daily_frame()
    late = daily["trade_date"] == TRADE_DATES[-1]
    daily.loc[late, "total_mv_rmb"] = daily.loc[late, "total_mv_rmb"] * 10.0
    _publish(tmp_path, daily)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "ready"
    first = by_date.loc[dates[0]]
    last = by_date.loc[dates[-1]]
    assert first.to_dict() == last.to_dict()
    assert set(last) == {"small", "mid", "large"}


def test_governed_exposure_is_point_in_time(tmp_path: Path) -> None:
    _publish(
        tmp_path,
        _daily_frame(availability_date="20240401"),
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "blocked"
    assert evidence["pit_violation_row_count"] > 0
    assert by_date.notna().sum().sum() == 0


def test_governed_exposure_blocks_when_the_generation_stops_short(
    tmp_path: Path,
) -> None:
    _publish(tmp_path, _daily_frame(trade_dates=TRADE_DATES[:2]))
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["status"] == "blocked"
    assert evidence["exposure_covers_evaluation_end"] is False
    assert evidence["share_reference_covers_evaluation_end"] is False
    assert evidence["evaluation_date_coverage_ratio"] < 1.0


def test_governed_exposure_never_reconstructs_market_caps(
    tmp_path: Path,
) -> None:
    _publish(tmp_path, _daily_frame())
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert evidence["point_in_time_size"] is True
    assert evidence["reconstructed_size_pair_count"] == 0
    assert evidence["reconstructed_size_pair_ratio"] == 0.0
    assert evidence["pit_size_pair_coverage_ratio"] == pytest.approx(1.0)
    assert evidence["combined_size_pair_coverage_ratio"] == pytest.approx(1.0)


def test_governed_exposure_blocks_when_the_pointer_is_missing(
    tmp_path: Path,
) -> None:
    dates = pd.to_datetime(["2024-01-02"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-02")
    )

    assert evidence["status"] == "blocked"
    assert evidence["blocker"].startswith("governed_exposure_generation_unavailable")
    assert evidence["catalog_validated"] is False


def test_governed_exposure_blocks_when_derivation_binding_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(tmp_path, _daily_frame())
    pointer = exposure_module.load_fundamental_pointer(tmp_path)
    assert pointer is not None
    pointer.pop("derivation_binding")
    monkeypatch.setattr(
        exposure_module,
        "load_fundamental_pointer",
        lambda _root: pointer,
    )
    dates = pd.to_datetime(["2024-01-02"])

    _sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-02")
    )

    assert evidence["status"] == "blocked"
    assert "derivation binding missing" in evidence["blocker"]
    assert evidence["catalog_validated"] is False


def test_governed_exposure_date_bounds_follow_the_generation(
    tmp_path: Path,
) -> None:
    _publish(tmp_path, _daily_frame())

    start, end = governed_exposure_date_bounds(tmp_path)

    assert start == pd.Timestamp("2024-01-02")
    assert end == pd.Timestamp("2024-01-04")


def test_governed_exposure_date_bounds_are_empty_without_a_generation(
    tmp_path: Path,
) -> None:
    assert governed_exposure_date_bounds(tmp_path) == (None, None)


@pytest.mark.parametrize(
    ("resolved", "exposure_start", "expected"),
    [
        ("2021-06-25", pd.Timestamp("2021-08-04"), "2021-08-04"),
        ("2022-01-04", pd.Timestamp("2021-08-04"), "2022-01-04"),
        ("", pd.Timestamp("2021-08-04"), "2021-08-04"),
        ("2021-06-25", None, "2021-06-25"),
        ("not-a-date", pd.Timestamp("2021-08-04"), "2021-08-04"),
    ],
)
def test_clamp_analysis_start_to_exposure(
    resolved: str,
    exposure_start: pd.Timestamp | None,
    expected: str,
) -> None:
    assert clamp_analysis_start_to_exposure(resolved, exposure_start) == expected


def test_governed_exposure_counts_unknown_sectors(tmp_path: Path) -> None:
    daily = _daily_frame()
    daily.loc[daily["ts_code"] == "000004.SZ", "sector"] = "unknown"
    _publish(tmp_path, daily)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    sectors, _sizes, _by_date, evidence = _load(
        tmp_path, dates=dates, as_of=pd.Timestamp("2024-01-04")
    )

    assert sectors["000004.SZ"] == "unknown"
    assert evidence["unknown_sector_count"] == 1
    assert evidence["covered_symbol_count"] == len(SYMBOLS) - 1
