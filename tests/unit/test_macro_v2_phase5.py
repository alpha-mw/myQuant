from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.data._tushare_client import TushareClientPool
from quant_investor.macro.coverage import (
    MacroCoverageError,
    build_macro_coverage_audit,
    run_macro_coverage_audit,
)


def _raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "dag_core_raw"
    fixtures = {
        "cn_cpi": {"month": "202404", "nt_yoy": 0.3},
        "cn_ppi": {"month": "202404", "ppi_yoy": -2.5},
        "cn_m": {"month": "202404", "m1_yoy": 2.0, "m2_yoy": 7.2},
        "cn_pmi": {"month": "202404", "PMI010000": 50.4},
        "sf_month": {"month": "202404", "inc_month": 12000.0},
    }
    for endpoint, row in fixtures.items():
        directory = root / f"table={endpoint}"
        directory.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    **row,
                    "source": "tushare_dag_core",
                    "source_snapshot_id": "local_fixture",
                    "fetched_at": "2024-05-10T06:00:00+00:00",
                }
            ]
        ).to_parquet(directory / "part.parquet", index=False)
    return root


def _observation(
    indicator_id: str,
    period: str,
    available: str,
    value: float,
    *,
    industry_chain: str = "",
    unit: str = "%",
) -> dict[str, object]:
    return {
        "indicator_id": indicator_id,
        "dimension_type": "industry" if industry_chain else "national",
        "industry_chain": industry_chain,
        "period_end": period,
        "release_at": available,
        "available_at": available,
        "vintage_id": f"observed:{period}",
        "value": value,
        "unit": unit,
        "frequency": "monthly",
        "source_system": "nbs_official",
        "source_record_id": f"fixture:{indicator_id}:{period}",
        "source_url": "https://www.stats.gov.cn/fixture",
        "fetched_at": available,
        "quality_status": "pass",
    }


def _write_observations(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _cpi_history() -> list[dict[str, object]]:
    return [
        _observation(
            "cn.cpi_yoy", "2024-02-29", "2024-03-10T01:30:00+00:00", 0.7
        ),
        _observation(
            "cn.cpi_yoy", "2024-03-31", "2024-04-10T01:30:00+00:00", 0.1
        ),
        _observation(
            "cn.cpi_yoy", "2024-04-30", "2024-05-10T01:30:00+00:00", 0.3
        ),
    ]


def test_raw_tables_are_visible_but_never_count_as_pit_observations(
    tmp_path: Path,
):
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=tmp_path / "missing_observations",
        raw_root=_raw_root(tmp_path),
    )
    by_id = {row["indicator_id"]: row for row in audit["national"]}
    assert audit["status"] == "blocked"
    assert audit["national_pit_signal_ready_count"] == 0
    assert len(audit["national"]) == 16
    assert len(audit["industry"]) == 96
    assert by_id["cn.cpi_yoy"]["status"] == "raw_present_pit_evidence_missing"
    assert by_id["cn.m1_yoy"]["raw_table_present"] is True
    assert by_id["cn.gdp_yoy"]["status"] == "mapped_raw_missing"
    assert by_id["cn.exports_yoy"]["status"] == "mapping_not_implemented"
    assert audit["raw_inventory"]["cn_cpi"]["pit_promotable"] is False
    assert audit["raw_inventory"]["cn_cpi"][
        "raw_available_by_audit_cutoff"
    ] is True
    assert audit["raw_inventory"]["cn_cpi"]["pit_blocker"] == (
        "timestamp_level_availability_evidence_not_bound"
    )
    assert audit["raw_inventory"]["cn_pmi"]["pit_blocker"] == (
        "timestamp_level_availability_evidence_not_bound"
    )


def test_future_raw_capture_is_not_visible_at_historical_cutoff(
    tmp_path: Path,
):
    raw_root = _raw_root(tmp_path)
    cpi_path = raw_root / "table=cn_cpi" / "part.parquet"
    frame = pd.read_parquet(cpi_path)
    frame["fetched_at"] = "2024-05-10T08:00:00+00:00"
    frame.to_parquet(cpi_path, index=False)
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=tmp_path / "missing",
        raw_root=raw_root,
    )
    cpi = next(
        row for row in audit["national"] if row["indicator_id"] == "cn.cpi_yoy"
    )
    assert cpi["status"] == "mapped_raw_not_usable_as_of"
    assert cpi["raw_available_by_audit_cutoff"] is False
    assert audit["raw_inventory"]["cn_cpi"]["pit_blocker"] == (
        "raw_capture_after_audit_cutoff"
    )


def test_raw_alias_conflict_and_missing_provenance_fail_closed(
    tmp_path: Path,
):
    raw_root = _raw_root(tmp_path)
    pmi_path = raw_root / "table=cn_pmi" / "part.parquet"
    pmi = pd.read_parquet(pmi_path)
    pmi["pmi010000"] = pmi["PMI010000"] + 1.0
    pmi.to_parquet(pmi_path, index=False)
    cpi_path = raw_root / "table=cn_cpi" / "part.parquet"
    cpi = pd.read_parquet(cpi_path)
    cpi["source_snapshot_id"] = ""
    cpi.to_parquet(cpi_path, index=False)

    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=tmp_path / "missing",
        raw_root=raw_root,
    )
    assert audit["raw_inventory"]["cn_pmi"]["pit_blocker"] == (
        "raw_value_alias_conflict:cn.pmi_manufacturing"
    )
    assert audit["raw_inventory"]["cn_cpi"]["pit_blocker"] == (
        "raw_provenance_missing"
    )


def test_real_pit_history_counts_but_does_not_clear_other_gaps(tmp_path: Path):
    observations = _write_observations(
        tmp_path / "observations.jsonl", _cpi_history()
    )
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=observations,
        raw_root=_raw_root(tmp_path),
    )
    by_id = {row["indicator_id"]: row for row in audit["national"]}
    assert by_id["cn.cpi_yoy"]["status"] == "pit_signal_ready"
    assert by_id["cn.cpi_yoy"]["latest_period_end"] == "2024-04-30"
    assert audit["national_pit_signal_ready_count"] == 1
    assert audit["status"] == "blocked"
    assert audit["production_eligible"] is False


def test_one_industry_chain_can_show_partial_real_coverage(tmp_path: Path):
    chain = "semiconductor_electronics"
    rows: list[dict[str, object]] = []
    for component in (
        "output",
        "orders",
        "inventory",
        "price_margin",
        "profits",
        "capex",
    ):
        indicator_id = f"industry.{chain}.{component}"
        rows.extend(
            [
                _observation(
                    indicator_id,
                    "2024-02-29",
                    "2024-03-05T01:00:00+00:00",
                    1.0,
                    industry_chain=chain,
                    unit="index",
                ),
                _observation(
                    indicator_id,
                    "2024-03-31",
                    "2024-04-05T01:00:00+00:00",
                    2.0,
                    industry_chain=chain,
                    unit="index",
                ),
                _observation(
                    indicator_id,
                    "2024-04-30",
                    "2024-05-05T01:00:00+00:00",
                    3.0,
                    industry_chain=chain,
                    unit="index",
                ),
            ]
        )
    observations = _write_observations(tmp_path / "industry.jsonl", rows)
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=observations,
        raw_root=_raw_root(tmp_path),
    )
    assert audit["industry_pit_signal_ready_count"] == 6
    assert audit["industry_chain_70pct_ready_count"] == 1
    chain_rows = [
        row for row in audit["industry"] if row["industry_chain"] == chain
    ]
    assert sum(row["status"] == "pit_signal_ready" for row in chain_rows) == 6


def test_conflicting_observations_are_disclosed_as_blocked(tmp_path: Path):
    rows = _cpi_history()
    rows.append({**rows[-1], "value": 9.9, "vintage_id": "conflict"})
    observations = _write_observations(tmp_path / "conflict.jsonl", rows)
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=observations,
        raw_root=_raw_root(tmp_path),
    )
    assert audit["observation_input_status"] == "blocked"
    assert any(
        item.startswith("snapshot_build_blocked:conflicting_vintage")
        for item in audit["blockers"]
    )
    assert audit["national_pit_signal_ready_count"] == 0


def test_private_artifacts_are_idempotent_and_tamper_fails(tmp_path: Path):
    kwargs = {
        "as_of": "2024-05-10",
        "observations_path": tmp_path / "missing",
        "raw_root": _raw_root(tmp_path),
        "output_root": tmp_path / "coverage",
    }
    first = run_macro_coverage_audit(**kwargs)
    out = Path(first["output_dir"])
    assert first["idempotent"] is False
    for path in out.iterdir():
        assert os.stat(path).st_mode & 0o777 == 0o600
    second = run_macro_coverage_audit(**kwargs)
    assert second["idempotent"] is True

    report = out / "coverage_report.md"
    report.write_bytes(report.read_bytes() + b"tamper")
    with pytest.raises(MacroCoverageError, match="existing_artifact_mismatch"):
        run_macro_coverage_audit(**kwargs)


def test_coverage_audit_never_calls_tushare(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        TushareClientPool,
        "query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network forbidden")
        ),
    )
    audit = build_macro_coverage_audit(
        as_of="2024-05-10",
        observations_path=tmp_path / "missing",
        raw_root=_raw_root(tmp_path),
    )
    assert audit["observer_only"] is True
