from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.macro.contracts import UTC, canonical_hash
from quant_investor.macro.local_market_observations import (
    LOCAL_MARKET_BREADTH_FORMULA_SHA256,
    LocalMarketObservationError,
    compile_local_market_breadth_observation,
)


def _clock() -> datetime:
    return datetime(2026, 7, 16, 1, 0, tzinfo=UTC)


def _mtime_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1_000


def _scope_sha256(symbols: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(symbols)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _Fixture:
    snapshot_manifest_path: Path
    snapshot_manifest_sha256: str
    coverage_manifest_path: Path
    coverage_manifest_sha256: str
    scope_artifact_path: Path
    scope_artifact_sha256: str
    part_path: Path
    frame: pd.DataFrame
    target_trade_date: str


def _coverage(
    *,
    trade_date: str,
    scope_symbols: list[str],
    observed_bar_count: int,
    absent_symbols: list[str],
    schema: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "coverage_schema_version": schema,
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": len(scope_symbols),
        "expected_scope_count": len(scope_symbols),
        "observed_bar_count": observed_bar_count,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "latest_available_trade_date": trade_date,
        "latest_complete_trade_date": trade_date,
        "upsert_target_trade_date": trade_date,
        "coverage_trade_date": trade_date,
        "expected_scope_sha256": _scope_sha256(scope_symbols),
        "suspended_symbols": [],
        "inactive_symbols": sorted(absent_symbols),
        "inactive_evidence_symbols": sorted(absent_symbols),
        "verified_nontrading_bak_daily_zero_symbols": [],
        "verified_terminal_delisting_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": sorted(absent_symbols),
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
    }
    if schema == "cn-full-a-coverage.v4":
        payload.update(
            {
                "pit_generation_id": "pit-fixture-v4",
                "pit_generation_manifest_path": "/fixture/pit/manifest.json",
                "pit_generation_manifest_sha256": "b" * 64,
                "pit_membership_path": "/fixture/pit/membership.parquet",
                "pit_membership_sha256": "c" * 64,
            }
        )
    return payload


def _write_fixture(
    tmp_path: Path,
    *,
    target_trade_date: str = "20260714",
    data_latest_complete: str = "20260714",
    data_snapshot_id: str = "20260715T042027Z",
    coverage_snapshot_id: str = "20260715T041500Z",
    coverage_schema: str = "cn-full-a-coverage.v3",
    rows_by_date: dict[str, int] | None = None,
    scope_size: int | None = None,
    declared_observed_bar_count: int | None = None,
    absent_symbols: list[str] | None = None,
    outside_scope_target_rows: int = 0,
    frame_mutator=None,
    part_mtime: str = "2026-07-15T04:20:29.774000Z",
    snapshot_manifest_mtime: str = "2026-07-15T04:21:14Z",
    coverage_manifest_mtime: str = "2026-07-15T04:19:00Z",
    scope_mtime: str = "2026-07-15T04:18:00Z",
) -> _Fixture:
    rows_by_date = rows_by_date or {
        "20260710": 100,
        "20260713": 100,
        "20260714": 100,
    }
    target_row_count = rows_by_date[target_trade_date]
    resolved_scope_size = scope_size or max(rows_by_date.values())
    scope_symbols = [
        f"{index:06d}.SZ" for index in range(resolved_scope_size)
    ]
    if target_row_count > resolved_scope_size:
        raise ValueError("target rows cannot exceed scope size")
    resolved_absent = (
        list(absent_symbols)
        if absent_symbols is not None
        else scope_symbols[target_row_count:]
    )
    resolved_observed_count = (
        declared_observed_bar_count
        if declared_observed_bar_count is not None
        else target_row_count
    )

    snapshot_root = tmp_path / "data" / "parquet" / "cn" / "_snapshots"
    table_root = snapshot_root / data_snapshot_id / "table" / "bars"
    part = table_root / "year=2026" / "month=07" / "part.parquet"
    part.parent.mkdir(parents=True)
    positive_ratios = {
        "20260710": 0.2,
        "20260713": 0.5,
        "20260714": 0.8,
        "20260715": 0.6,
    }
    rows: list[dict[str, object]] = []
    for trade_date, row_count in rows_by_date.items():
        positive_count = int(row_count * positive_ratios.get(trade_date, 0.5))
        for index in range(row_count):
            rows.append(
                {
                    "ts_code": scope_symbols[index],
                    "trade_date": trade_date,
                    "pct_chg": 1.0 if index < positive_count else -1.0,
                }
            )
        if trade_date == target_trade_date:
            for index in range(outside_scope_target_rows):
                rows.append(
                    {
                        "ts_code": f"9{index:05d}.SH",
                        "trade_date": trade_date,
                        "pct_chg": 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame_mutator is not None:
        frame = frame_mutator(frame.copy())
    frame.to_parquet(part, index=False)
    part_mtime_ns = _mtime_ns(part_mtime)
    os.utime(part, ns=(part_mtime_ns, part_mtime_ns))

    scope_path = tmp_path / "data" / "cn_universe" / "cn_index_components.json"
    scope_path.parent.mkdir(parents=True)
    scope_path.write_text(
        json.dumps({"full_a": scope_symbols}, sort_keys=True),
        encoding="utf-8",
    )
    scope_mtime_ns = _mtime_ns(scope_mtime)
    os.utime(scope_path, ns=(scope_mtime_ns, scope_mtime_ns))
    scope_file_sha = hashlib.sha256(scope_path.read_bytes()).hexdigest()

    selected_coverage = _coverage(
        trade_date=target_trade_date,
        scope_symbols=scope_symbols,
        observed_bar_count=resolved_observed_count,
        absent_symbols=resolved_absent,
        schema=coverage_schema,
    )
    coverage_root = tmp_path / "coverage" / "_snapshots"
    coverage_path = coverage_root / f"{coverage_snapshot_id}.json"
    coverage_path.parent.mkdir(parents=True)
    coverage_manifest = {
        "snapshot_id": coverage_snapshot_id,
        "market": "CN",
        "status": "OK",
        "latest_trade_date": target_trade_date,
        "latest_complete_trade_date": target_trade_date,
        "table_root": str(table_root),
        "manifest_path": str(coverage_path),
        "readback_validated": True,
        "coverage": selected_coverage,
        "metadata": {"coverage": json.loads(json.dumps(selected_coverage))},
        "blockers": [],
    }
    coverage_path.write_text(
        json.dumps(coverage_manifest, sort_keys=True),
        encoding="utf-8",
    )
    coverage_mtime_ns = _mtime_ns(coverage_manifest_mtime)
    os.utime(coverage_path, ns=(coverage_mtime_ns, coverage_mtime_ns))
    coverage_sha = hashlib.sha256(coverage_path.read_bytes()).hexdigest()

    snapshot_path = snapshot_root / f"{data_snapshot_id}.json"
    data_manifest = {
        "snapshot_id": data_snapshot_id,
        "market": "CN",
        "status": "OK",
        "row_count": len(frame),
        "symbol_count": int(frame["ts_code"].nunique()),
        "latest_trade_date": data_latest_complete,
        "latest_complete_trade_date": data_latest_complete,
        "table_root": str(table_root),
        "manifest_path": str(snapshot_path),
        "readback_validated": True,
        "coverage": {"coverage_schema_version": "cn-full-a-coverage.v4"},
        "metadata": {},
        "blockers": [],
    }
    snapshot_path.write_text(
        json.dumps(data_manifest, sort_keys=True),
        encoding="utf-8",
    )
    snapshot_mtime_ns = _mtime_ns(snapshot_manifest_mtime)
    os.utime(snapshot_path, ns=(snapshot_mtime_ns, snapshot_mtime_ns))
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    return _Fixture(
        snapshot_manifest_path=snapshot_path,
        snapshot_manifest_sha256=snapshot_sha,
        coverage_manifest_path=coverage_path,
        coverage_manifest_sha256=coverage_sha,
        scope_artifact_path=scope_path,
        scope_artifact_sha256=scope_file_sha,
        part_path=part,
        frame=frame,
        target_trade_date=target_trade_date,
    )


def _compile(fixture: _Fixture, *, as_of: str = "20260715"):
    return compile_local_market_breadth_observation(
        snapshot_manifest_path=fixture.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=(
            fixture.snapshot_manifest_sha256
        ),
        coverage_manifest_path=fixture.coverage_manifest_path,
        expected_coverage_manifest_sha256=(
            fixture.coverage_manifest_sha256
        ),
        target_trade_date=fixture.target_trade_date,
        scope_artifact_path=fixture.scope_artifact_path,
        expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
        as_of=as_of,
        clock=_clock,
    )


def _rewrite_manifest(path: Path, mutator) -> str:
    original_mtime_ns = path.stat().st_mtime_ns
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.utime(path, ns=(original_mtime_ns, original_mtime_ns))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compiles_one_hash_bound_breadth_observation_read_only(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, outside_scope_target_rows=3)
    before = {
        item.relative_to(tmp_path).as_posix(): hashlib.sha256(
            item.read_bytes()
        ).hexdigest()
        for item in tmp_path.rglob("*")
        if item.is_file()
    }

    observation, evidence = _compile(fixture)

    after = {
        item.relative_to(tmp_path).as_posix(): hashlib.sha256(
            item.read_bytes()
        ).hexdigest()
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    assert before == after
    assert observation.period_end == "2026-07-14"
    assert observation.value == 80.0
    assert observation.source_system == "local_strict_parquet"
    assert observation.source_url == (
        "local://strict-parquet/cn/snapshots/"
        "20260715T042027Z/bars/20260714"
    )
    assert observation.release_at == "2026-07-14T07:00:00+00:00"
    assert observation.available_at == "2026-07-15T04:21:14+00:00"
    assert observation.fetched_at == "2026-07-16T01:00:00+00:00"
    assert evidence["snapshot_manifest_sha256"] == (
        fixture.snapshot_manifest_sha256
    )
    assert evidence["coverage_manifest_sha256"] == (
        fixture.coverage_manifest_sha256
    )
    assert evidence["coverage_source_location"] == "top"
    assert evidence["coverage_summary"]["expected_scope_count"] == 100
    assert evidence["coverage_summary"]["observed_bar_count"] == 100
    assert canonical_hash(evidence["coverage_summary"]) == (
        evidence["coverage_contract_sha256"]
    )
    assert evidence["scope_artifact"]["file_sha256"] == (
        fixture.scope_artifact_sha256
    )
    assert evidence["part_file"]["target_all_row_count"] == 103
    assert evidence["part_file"]["target_scope_row_count"] == 100
    assert evidence["part_file"]["target_outside_scope_row_count"] == 3
    assert evidence["formula_contract_sha256"] == (
        LOCAL_MARKET_BREADTH_FORMULA_SHA256
    )
    assert evidence["target_trade_date"] == "20260714"
    assert evidence["local_read_only"] is True
    assert evidence["canonical_write"] is False
    semantic_evidence = dict(evidence)
    declared_evidence_sha = semantic_evidence.pop("evidence_sha256")
    assert canonical_hash(semantic_evidence) == declared_evidence_sha
    binding = evidence["observation_binding"]
    assert binding["snapshot_manifest_sha256"] == (
        fixture.snapshot_manifest_sha256
    )
    assert binding["coverage_manifest_sha256"] == (
        fixture.coverage_manifest_sha256
    )
    assert binding["scope_evidence_sha256"] == (
        evidence["scope_evidence_sha256"]
    )
    assert binding["binding_sha256"] in observation.vintage_id
    assert binding["content_hash"] == observation.content_hash


def test_accepts_explicit_v4_coverage_manifest(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        coverage_schema="cn-full-a-coverage.v4",
    )

    observation, evidence = _compile(fixture)

    assert observation.period_end == "2026-07-14"
    assert evidence["coverage_summary"]["coverage_schema_version"] == (
        "cn-full-a-coverage.v4"
    )


def test_rejects_snapshot_manifest_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_manifest_sha256_mismatch",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256="0" * 64,
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


def test_rejects_coverage_manifest_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_coverage_manifest_sha256_mismatch",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256="0" * 64,
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


def test_rejects_scope_artifact_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_scope_artifact_sha256_mismatch",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256="0" * 64,
            as_of="20260715",
            clock=_clock,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "blocker"),
    [
        ("complete", False, "local_breadth_coverage_not_complete"),
        (
            "coverage_ratio",
            0.99,
            "local_breadth_coverage_contract_invalid",
        ),
        (
            "latest_available_trade_date",
            "20260713",
            "local_breadth_coverage_trade_date_mismatch",
        ),
        (
            "true_missing_symbols",
            ["000001.SZ"],
            "local_breadth_coverage_contract_invalid",
        ),
        (
            "coverage_schema_version",
            "cn-full-a-coverage.v2",
            "local_breadth_coverage_schema_invalid",
        ),
    ],
)
def test_rejects_partial_or_invalid_full_a_coverage(
    tmp_path: Path,
    field_name: str,
    value,
    blocker: str,
) -> None:
    fixture = _write_fixture(tmp_path)

    def _mutate(payload):
        payload["coverage"][field_name] = value
        payload["metadata"]["coverage"][field_name] = value

    digest = _rewrite_manifest(fixture.coverage_manifest_path, _mutate)

    with pytest.raises(LocalMarketObservationError, match=blocker):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=digest,
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


def test_rejects_top_and_metadata_coverage_conflict(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def _conflict(payload):
        payload["metadata"]["coverage"]["coverage_ratio"] = 0.5

    digest = _rewrite_manifest(fixture.coverage_manifest_path, _conflict)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_coverage_top_metadata_conflict",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=digest,
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


def test_rejects_legacy_mutable_data_manifest_even_with_valid_coverage(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    legacy_root = tmp_path / "data" / "parquet" / "cn" / "bars"
    legacy_root.mkdir(parents=True)

    def _legacy(payload):
        payload["table_root"] = str(legacy_root)

    digest = _rewrite_manifest(fixture.snapshot_manifest_path, _legacy)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_v4_immutable_table_root_required",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=digest,
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


def test_rejects_legacy_data_coverage_schema(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def _legacy(payload):
        payload["coverage"]["coverage_schema_version"] = (
            "cn-full-a-coverage.v3"
        )

    digest = _rewrite_manifest(fixture.snapshot_manifest_path, _legacy)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_snapshot_manifest_v4_required",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=digest,
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=_clock,
        )


@pytest.mark.parametrize(
    ("data_snapshot_id", "coverage_snapshot_id", "blocker"),
    [
        (
            "20260715T080000Z",
            "20260715T041500Z",
            "local_breadth_snapshot_after_published_cutoff",
        ),
        (
            "20260715T042027Z",
            "20260715T080000Z",
            "local_breadth_coverage_snapshot_after_published_cutoff",
        ),
    ],
)
def test_rejects_snapshot_time_after_cutoff(
    tmp_path: Path,
    data_snapshot_id: str,
    coverage_snapshot_id: str,
    blocker: str,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        data_snapshot_id=data_snapshot_id,
        coverage_snapshot_id=coverage_snapshot_id,
    )

    with pytest.raises(LocalMarketObservationError, match=blocker):
        _compile(fixture)


@pytest.mark.parametrize(
    (
        "part_mtime",
        "snapshot_manifest_mtime",
        "coverage_manifest_mtime",
        "scope_mtime",
    ),
    [
        (
            "2026-07-15T07:00:00.000001Z",
            "2026-07-15T04:21:14Z",
            "2026-07-15T04:19:00Z",
            "2026-07-15T04:18:00Z",
        ),
        (
            "2026-07-15T04:20:29Z",
            "2026-07-15T07:00:00.000001Z",
            "2026-07-15T04:19:00Z",
            "2026-07-15T04:18:00Z",
        ),
        (
            "2026-07-15T04:20:29Z",
            "2026-07-15T04:21:14Z",
            "2026-07-15T07:00:00.000001Z",
            "2026-07-15T04:18:00Z",
        ),
        (
            "2026-07-15T04:20:29Z",
            "2026-07-15T04:21:14Z",
            "2026-07-15T04:19:00Z",
            "2026-07-15T07:00:00.000001Z",
        ),
    ],
)
def test_rejects_any_bound_file_mtime_after_cutoff(
    tmp_path: Path,
    part_mtime: str,
    snapshot_manifest_mtime: str,
    coverage_manifest_mtime: str,
    scope_mtime: str,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        part_mtime=part_mtime,
        snapshot_manifest_mtime=snapshot_manifest_mtime,
        coverage_manifest_mtime=coverage_manifest_mtime,
        scope_mtime=scope_mtime,
    )

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_effective_available_after_published_cutoff",
    ):
        _compile(fixture)


def test_fetched_at_cannot_precede_effective_availability(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_fetched_before_effective_available_at",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=fixture.scope_artifact_sha256,
            as_of="20260715",
            clock=lambda: datetime(2026, 7, 15, 4, 21, 13, tzinfo=UTC),
        )


def _duplicate(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[frame["trade_date"] == "20260714"].iloc[[0]]
    return pd.concat([frame, target], ignore_index=True)


def _non_finite(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index[frame["trade_date"] == "20260714"][0]
    frame.loc[index, "pct_chg"] = np.inf
    return frame


@pytest.mark.parametrize(
    ("rows", "mutator", "blocker"),
    [
        (100, _duplicate, "local_breadth_duplicate_bar"),
        (100, _non_finite, "local_breadth_pct_chg_non_finite"),
        (99, None, "local_breadth_rows_insufficient:20260714:99"),
    ],
)
def test_rejects_unsafe_or_insufficient_observed_rows(
    tmp_path: Path,
    rows: int,
    mutator,
    blocker: str,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        rows_by_date={"20260710": 100, "20260713": 100, "20260714": rows},
        frame_mutator=mutator,
    )

    with pytest.raises(LocalMarketObservationError, match=blocker):
        _compile(fixture)


def test_rejects_symlinked_part_even_when_target_is_valid(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    target = tmp_path / "part-target.parquet"
    target.write_bytes(fixture.part_path.read_bytes())
    fixture.part_path.unlink()
    fixture.part_path.symlink_to(target)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_part_unsafe_or_unreadable",
    ):
        _compile(fixture)


def test_rejects_declared_5502_when_actual_target_has_only_100_rows(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        rows_by_date={"20260710": 100, "20260713": 100, "20260714": 100},
        scope_size=5502,
        declared_observed_bar_count=5502,
        absent_symbols=[],
    )

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_actual_observed_bar_count_mismatch",
    ):
        _compile(fixture)


def test_rejects_wrong_absent_set_even_when_declared_counts_balance(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        rows_by_date={"20260710": 100, "20260713": 100, "20260714": 100},
        scope_size=101,
        declared_observed_bar_count=100,
        absent_symbols=["000099.SZ"],
    )

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_observed_absent_overlap",
    ):
        _compile(fixture)


def test_one_latest_manifest_cannot_emit_prior_sparse_dates(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        rows_by_date={
            "20260710": 100,
            "20260713": 100,
            "20260714": 5502,
        },
        scope_size=5502,
    )

    observation, evidence = _compile(fixture)

    assert observation.period_end == "2026-07-14"
    assert evidence["target_trade_date"] == "20260714"
    assert "target_trade_dates" not in evidence


def test_each_sparse_prior_date_needs_its_own_closing_coverage_manifest(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        target_trade_date="20260713",
        data_latest_complete="20260714",
        rows_by_date={"20260710": 100, "20260713": 100, "20260714": 5502},
        scope_size=5502,
        declared_observed_bar_count=5502,
        absent_symbols=[],
    )

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_actual_observed_bar_count_mismatch",
    ):
        _compile(fixture)


def test_rejects_scope_file_with_correct_raw_hash_but_wrong_semantic_set(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    original_mtime = fixture.scope_artifact_path.stat().st_mtime_ns
    payload = json.loads(
        fixture.scope_artifact_path.read_text(encoding="utf-8")
    )
    payload["full_a"][0] = "999999.SH"
    fixture.scope_artifact_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    os.utime(
        fixture.scope_artifact_path,
        ns=(original_mtime, original_mtime),
    )
    new_sha = hashlib.sha256(
        fixture.scope_artifact_path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_scope_artifact_semantic_sha256_mismatch",
    ):
        compile_local_market_breadth_observation(
            snapshot_manifest_path=fixture.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=(
                fixture.snapshot_manifest_sha256
            ),
            coverage_manifest_path=fixture.coverage_manifest_path,
            expected_coverage_manifest_sha256=(
                fixture.coverage_manifest_sha256
            ),
            target_trade_date=fixture.target_trade_date,
            scope_artifact_path=fixture.scope_artifact_path,
            expected_scope_artifact_sha256=new_sha,
            as_of="20260715",
            clock=_clock,
        )
