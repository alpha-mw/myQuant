from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quant_investor.v17_v4_contract import load_canonical_artifact
from quant_investor.v17_v4_runtime.cli import main as v17_v4_main
from quant_investor.v17_v4_runtime.research_factor_set import (
    CANDIDATE_CATALOG_SHA256,
    CATALOG_RESOURCE_SHA256,
    GATE_ORDER,
    IMPLEMENTATION_RESOURCE_SHA256,
    ResearchFactorSetStore,
    build_research_shadow_factor_set,
    research_factor_catalog_bindings,
)
from quant_investor.v17_v4_runtime.source_builder import (
    FACTOR_FIELDS,
    NEUTRALIZER_FIELDS,
    SourceSnapshotGap,
    build_source_snapshot,
)
from quant_investor.v17_v4_runtime.source_storage import EMPTY_SHA256

STRATEGY_ID = "cn-aggressive-tech-manufacturing"
SESSION = "2026-07-29"
CUTOFF = "2026-07-29T08:00:00Z"
SYMBOLS = ("000001.SZ", "600000.SH", "688001.SH")
FACTOR_NAMES = (
    "cn_fip_continuous_direction_12m",
    "cn_low_market_adjusted_tail_asymmetry_252d",
    "cn_low_total_skewness_20d",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _copy_factor_set(workspace: Path) -> str:
    bindings = research_factor_catalog_bindings()
    selections: list[dict[str, object]] = []
    for name in FACTOR_NAMES:
        factor = bindings["factors"][name]
        selections.append(
            {
                "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
                "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
                "definition_sha256": factor["definition_sha256"],
                "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
                "implementation_sha256": factor["implementation_sha256"],
                "name": name,
                "selection_gates": {gate_id: True for gate_id in GATE_ORDER},
            }
        )
    audit_path = "data/private/v17_v4_sources/audits/monthly-audit-test.json"
    factor_set = build_research_shadow_factor_set(
        factor_set_id="source-builder-test-factors",
        strategy_id=STRATEGY_ID,
        cutoff="2026-07-28T08:00:00Z",
        audit_session="2026-07-28",
        selected_at="2026-07-28T08:00:00Z",
        published_at="2026-07-28T08:00:01Z",
        open_sessions=("2026-07-28", "2026-07-29"),
        monthly_audit_ref={
            "artifact_id": "monthly-audit-test",
            "artifact_version": "myquant.factor-governance.monthly-audit.v4",
            "byte_sha256": hashlib.sha256(audit_path.encode()).hexdigest(),
            "cutoff": "2026-07-28T08:00:00Z",
            "relative_path": audit_path,
            "semantic_sha256": hashlib.sha256(
                f"{audit_path}:semantic".encode()
            ).hexdigest(),
            "strategy_id": STRATEGY_ID,
        },
        previous_factor_set_ref=None,
        selection_rows=selections,
        expected_candidate_catalog_sha256=CANDIDATE_CATALOG_SHA256,
        expected_catalog_resource_sha256=CATALOG_RESOURCE_SHA256,
        expected_implementation_resource_sha256=IMPLEMENTATION_RESOURCE_SHA256,
    )
    publication = ResearchFactorSetStore(str(workspace.resolve())).publish(
        factor_set,
        expected_pointer_sha256=EMPTY_SHA256,
    )
    return str(publication.pointer_ref["byte_sha256"])


def _canonical_inputs(workspace: Path) -> dict[str, str]:
    sessions = pd.bdate_range(end=SESSION, periods=253)
    market_rows: list[dict[str, object]] = []
    for day_index, trade_day in enumerate(sessions):
        for symbol_index, symbol in enumerate(SYMBOLS):
            market_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_day.strftime("%Y%m%d"),
                    "adj_close": (
                        10.0
                        + symbol_index * 3.0
                        + day_index * (0.01 + symbol_index * 0.002)
                        + (day_index % (7 + symbol_index)) * 0.003
                    ),
                    "total_mv": (100_000.0 + symbol_index * 20_000.0 + day_index * 25.0),
                }
            )
    table_root = workspace / "data/parquet/cn/_snapshots/test-snapshot/table/bars"
    market_part = table_root / "part.parquet"
    _write_parquet(market_part, pd.DataFrame(market_rows))

    pit_path = (
        workspace / "data/parquet/cn/reference/_generations/pit-test/"
        "stock_basic_membership.parquet"
    )
    _write_parquet(
        pit_path,
        pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "industry": f"industry-{index}",
                    "source_list_status": "L",
                    "effective_from": "2020-01-01",
                    "effective_to": None,
                    "observed_at": "2026-07-28T02:00:00Z",
                }
                for index, symbol in enumerate(SYMBOLS)
            ]
        ),
    )
    pit_manifest_path = pit_path.parent / "manifest.json"
    _write_json(
        pit_manifest_path,
        {
            "canonical_path": str(pit_path),
            "canonical_sha256": _sha(pit_path),
            "written_at": "2026-07-28T02:00:01Z",
        },
    )

    market_manifest_path = workspace / "data/parquet/cn/_snapshots/test-snapshot.json"
    _write_json(
        market_manifest_path,
        {
            "blockers": [],
            "latest_complete_trade_date": "20260729",
            "readback_validated": True,
            "snapshot_id": "test-snapshot",
            "status": "OK",
            "table_root": str(table_root),
        },
    )
    market_pointer_path = workspace / "data/parquet/cn/_latest.json"
    _write_json(
        market_pointer_path,
        {
            "blockers": [],
            "coverage": {
                "complete": True,
                "pit_generation_manifest_path": str(pit_manifest_path),
                "pit_generation_manifest_sha256": _sha(pit_manifest_path),
                "pit_membership_path": str(pit_path),
                "pit_membership_sha256": _sha(pit_path),
            },
            "latest_complete_trade_date": "20260729",
            "manifest_path": str(market_manifest_path),
            "snapshot_id": "test-snapshot",
            "status": "OK",
            "table_root": str(table_root),
            "updated_at": "2026-07-29T07:10:00Z",
        },
    )

    fundamental_root = workspace / "data/parquet/cn/_fundamental_generations/fundamental-test"
    fundamental_path = fundamental_root / "fundamental_daily.parquet"
    _write_parquet(
        fundamental_path,
        pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "trade_date": pd.Timestamp("2026-07-28"),
                    "end_date": "2026-06-30",
                    "availability_date": pd.Timestamp("2026-07-20"),
                    "fetched_at": "2026-07-20T05:00:00Z",
                    "fin_roe": 8.0 + index,
                    "fin_ocf_to_profit": 0.8 + index * 0.1,
                    "fin_debt_to_assets": 30.0 + index,
                }
                for index, symbol in enumerate(SYMBOLS)
            ]
        ),
    )
    fundamental_manifest_path = fundamental_root / "manifest.json"
    _write_json(
        fundamental_manifest_path,
        {
            "generation_id": "fundamental-test",
            "metadata": {
                "provider_manifest": {
                    "canonical_scope_evidence": {
                        "canonical_bar_first_dates": {
                            f"{index:06d}.SZ": "2020-01-01" for index in range(600)
                        }
                    }
                }
            },
            "status": "OK",
            "tables": {
                "fundamental_daily": {
                    "sha256": _sha(fundamental_path),
                }
            },
        },
    )
    fundamental_pointer_path = workspace / "data/parquet/cn/_fundamental_latest.json"
    _write_json(
        fundamental_pointer_path,
        {
            "generation_id": "fundamental-test",
            "manifest_path": ("_fundamental_generations/fundamental-test/manifest.json"),
            "primary_provenance": {
                "output_parquet_sha256": {
                    "fundamental_daily": _sha(fundamental_path),
                }
            },
            "status": "OK",
            "tables": {
                "fundamental_daily": (
                    "_fundamental_generations/fundamental-test/" "fundamental_daily.parquet"
                )
            },
        },
    )

    universe_path = (
        workspace / "results/strategy_records/CN/aggressive_tech_manufacturing/"
        "_cache/market_metrics/test-snapshot_20260729/full_metrics.parquet"
    )
    _write_parquet(
        universe_path,
        pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "name": f"name-{index}",
                    "category": "test",
                }
                for index, symbol in enumerate(SYMBOLS)
            ]
        ),
    )
    universe_manifest_path = universe_path.parent / "breadth.json"
    _write_json(
        universe_manifest_path,
        {
            "analysis_trade_date": "20260729",
            "data_coverage": {"data_coverage_valid": True},
            "full_metrics_path": str(universe_path),
            "generated_at": "2026-07-29T15:15:00+08:00",
            "schema_validation": {"schema_valid": True},
        },
    )
    return {
        "factor_set_pointer_sha256": _copy_factor_set(workspace),
        "fundamental_pointer_sha256": _sha(fundamental_pointer_path),
        "market_pointer_sha256": _sha(market_pointer_path),
        "strategy_universe_manifest_path": str(universe_manifest_path.relative_to(workspace)),
        "strategy_universe_manifest_sha256": _sha(universe_manifest_path),
        "strategy_universe_path": str(universe_path.relative_to(workspace)),
        "strategy_universe_sha256": _sha(universe_path),
    }


def _build(workspace: Path, inputs: dict[str, str]) -> dict[str, object]:
    return build_source_snapshot(
        str(workspace),
        strategy_id=STRATEGY_ID,
        decision_session=SESSION,
        cutoff=CUTOFF,
        **inputs,
    )


def test_source_builder_publishes_schema_valid_exact_once_snapshot(
    tmp_path: Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)

    first = _build(tmp_path, inputs)
    second = _build(tmp_path, inputs)

    assert first["status"] == "READY"
    assert first["created_artifacts"] == 16
    assert second["created_artifacts"] == 0
    assert second["reused_artifacts"] == 16
    snapshot = tmp_path / "data/private/v17_v4_sources/snapshots" / SESSION
    expected = {
        "source_locator.json",
        "factor_input_bundle.json",
        "universe.parquet",
        "universe.manifest.json",
        "neutralizer.parquet",
        "neutralizer.manifest.json",
        *{f"factor_inputs/{field_name}.parquet" for field_name in FACTOR_FIELDS},
        *{f"factor_inputs/{field_name}.manifest.json" for field_name in FACTOR_FIELDS},
    }
    observed = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()}
    assert observed == expected
    for path in snapshot.rglob("*.json"):
        load_canonical_artifact(path.read_bytes(), label=path.name)
    for path in snapshot.rglob("*.parquet"):
        metadata = pq.read_metadata(path).metadata
        assert metadata is not None
        for field_name in (
            b"available_at",
            b"cutoff",
            b"schema_version",
            b"semantic_sha256",
        ):
            assert metadata[field_name]
    neutralizer = pd.read_parquet(snapshot / "neutralizer.parquet")
    assert set(NEUTRALIZER_FIELDS).issubset(neutralizer.columns)
    assert neutralizer[list(NEUTRALIZER_FIELDS)].notna().all().all()


def test_source_builder_fails_closed_before_writes_on_sha_drift(
    tmp_path: Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    inputs["market_pointer_sha256"] = "0" * 64

    with pytest.raises(
        SourceSnapshotGap,
        match="SOURCE_SHA256_MISMATCH",
    ):
        _build(tmp_path, inputs)

    assert not (tmp_path / "data/private/v17_v4_sources/snapshots" / SESSION).exists()


def test_source_builder_cli_emits_only_canonical_gap_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _canonical_inputs(tmp_path)
    inputs["market_pointer_sha256"] = "0" * 64
    argv = [
        "build-source-snapshot",
        "--workspace-root",
        str(tmp_path),
        "--strategy-id",
        STRATEGY_ID,
        "--decision-session",
        SESSION,
        "--cutoff",
        CUTOFF,
    ]
    for key, value in inputs.items():
        argv.extend(["--" + key.replace("_", "-"), value])

    assert v17_v4_main(argv) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "TRUE_CURRENT_CANONICAL_INPUT_GAP"
    assert payload["authority"] == {
        "broker": False,
        "execution": False,
        "formal_research_publication": False,
        "order": False,
        "research_runtime_default": False,
        "trade": False,
    }
    assert payload["formal_activation_eligible"] is False
