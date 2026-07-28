from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_investor.monitoring import v17_daily_gray as subject
from quant_investor.monitoring.v17_daily_gray import (
    NO_AUTHORITY,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    SCHEMA_VERSION,
    run_daily_gray_comparison,
)


def _write_json(path: Path, payload: dict, *, private: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600 if private else 0o644)
    return hashlib.sha256(raw).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _v15_run(tmp_path: Path, *, session: str = "20260727") -> Path:
    run_dir = tmp_path / "records" / "20260728_1200"
    run_dir.mkdir(parents=True)
    (run_dir / "raw_exports").mkdir()
    _write_json(
        run_dir / "manifest.json",
        {
            "timestamp": "20260728_1200",
            "data_snapshot": {
                "analysis_trade_date": session,
                "completeness": {"snapshot_id": "snapshot-1"},
            },
        },
    )
    _write_json(run_dir / "market_snapshot.json", {"analysis_trade_date": session})
    _write_csv(
        run_dir / "candidate_pool.csv",
        [{"symbol": "000001.SZ"}, {"symbol": "000762.SZ"}],
        ["symbol"],
    )
    _write_csv(
        run_dir / "holdings_review.csv",
        [
            {"symbol": "000001.SZ", "nav_weight": "0.03"},
            {"symbol": "000002.SZ", "nav_weight": "0.02"},
        ],
        ["symbol", "nav_weight"],
    )
    _write_csv(
        run_dir / "pnl_summary.csv",
        [{"total_value_after": "10000000"}],
        ["total_value_after"],
    )
    (run_dir / "analysis_report.md").write_text("# V15 report\n", encoding="utf-8")
    return run_dir


def _pointer(tmp_path: Path, *, session: str = "20260727") -> tuple[Path, str]:
    path = tmp_path / "data/parquet/cn/_latest.json"
    sha = _write_json(
        path,
        {
            "snapshot_id": "snapshot-1",
            "latest_complete_trade_date": session,
        },
    )
    return path, sha


def _v17_workspace(
    tmp_path: Path,
    *,
    pointer_sha: str,
    session: str = "20260727",
    run_id: str = "v17-shadow-1",
) -> Path:
    root = tmp_path / "data/private/v17_v3_workspaces"
    workspace = root / "workspace-1"
    workspace.mkdir(parents=True)
    workspace.chmod(0o700)
    run_dir = workspace / "data/private/v17_v3_runs" / run_id
    top24 = [
        {
            "symbol": f"{index:06d}.SZ",
            "name": f"N{index}",
            "deep_status": "BUY_VETO",
            "final_target": "0",
        }
        for index in range(762, 786)
    ]
    summary = {
        "version": "myquant.v17.v3.current-shadow-run-summary.v1",
        "status": "SHADOW_COMPLETE",
        "strategy_id": "cn-full-a-v17-v3-model",
        "run_id": run_id,
        "cutoff": "2026-07-28T02:50:25Z",
        "decision_session": session,
        "factor_baseline_mode": "PROVISIONAL_RESEARCH",
        "portfolio_basis": "MODEL_ONLY_NO_PRIVATE_HOLDINGS",
        "calibration": "UNCALIBRATED_50_50",
        "gross_weight": "0",
        "cash_weight": "1",
        "top24": top24,
        "source_bindings": {"market_pointer_sha256": pointer_sha},
        "authority": dict(NO_AUTHORITY),
    }
    fusion = {
        "version": "myquant.v17.v3.fusion-output.v1",
        "status": "READY",
        "state": "FUSION_COMPLETE",
        "strategy_id": "cn-full-a-v17-v3-model",
        "run_id": run_id,
        "selected_symbols": [row["symbol"] for row in top24],
        "common_ready_domain": [
            "000001.SZ",
            "000002.SZ",
            *[row["symbol"] for row in top24],
        ],
        "authority": dict(NO_AUTHORITY),
    }
    _write_json(run_dir / "run_summary.json", summary, private=True)
    _write_json(run_dir / "fusion_output.json", fusion, private=True)
    return root


def test_same_session_same_pointer_writes_comparable_gray_sidecar(tmp_path: Path) -> None:
    run_dir = _v15_run(tmp_path)
    pointer_path, pointer_sha = _pointer(tmp_path)
    v17_root = _v17_workspace(tmp_path, pointer_sha=pointer_sha)

    result = run_daily_gray_comparison(
        run_dir=run_dir,
        v17_workspace_root=v17_root,
        market_pointer_path=pointer_path,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )

    assert result["status"] == "GRAY_COMPARISON_COMPLETE"
    assert result["classification"] == "COMPARABLE"
    assert result["authority"] == NO_AUTHORITY
    assert result["metrics"]["candidate_overlap_count"] == 1
    assert result["metrics"]["v15_holdings_in_v17_common_ready_count"] == 2
    assert result["metrics"]["v15_holdings_in_v17_top24_count"] == 0
    assert result["metrics"]["v15_actual_gross_weight"] == 0.05
    assert result["metrics"]["v17_model_gross_weight"] == 0.0
    assert result["metrics"]["v17_deep_buy_veto_count"] == 24
    document = json.loads((run_dir / OUTPUT_JSON).read_text(encoding="utf-8"))
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["effect_evaluation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert document["effect_evaluation"]["observed_comparable_sessions"] == 1
    assert document["side_effect_attestation"]["broker_calls"] == 0
    assert (run_dir / OUTPUT_MARKDOWN).is_file()
    assert (run_dir / "raw_exports" / OUTPUT_JSON).is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["v17_gray_comparison"]["classification"] == "COMPARABLE"
    assert "V15 / V17 日度灰度比较" in (run_dir / "analysis_report.md").read_text(encoding="utf-8")


def test_missing_v17_shadow_is_non_blocking_and_explicit(tmp_path: Path) -> None:
    run_dir = _v15_run(tmp_path)
    pointer_path, pointer_sha = _pointer(tmp_path)
    v17_root = tmp_path / "data/private/v17_v3_workspaces"
    v17_root.mkdir(parents=True)

    result = run_daily_gray_comparison(
        run_dir=run_dir,
        v17_workspace_root=v17_root,
        market_pointer_path=pointer_path,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )

    assert result["status"] == "GRAY_UNAVAILABLE"
    assert result["classification"] == "NON_COMPARABLE"
    assert "same_session_v17_shadow_not_found" in result["blockers"]
    assert (run_dir / OUTPUT_JSON).is_file()


def test_pointer_drift_rejects_comparison_before_v17_admission(tmp_path: Path) -> None:
    run_dir = _v15_run(tmp_path)
    pointer_path, pointer_sha = _pointer(tmp_path)
    v17_root = _v17_workspace(tmp_path, pointer_sha=pointer_sha)

    result = run_daily_gray_comparison(
        run_dir=run_dir,
        v17_workspace_root=v17_root,
        market_pointer_path=pointer_path,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15="f" * 64,
    )

    assert result["classification"] == "NON_COMPARABLE"
    assert "market_pointer_drift_during_dual_run" in result["blockers"]
    assert result["effect_verdict"] == "NO_V15_V17_PERFORMANCE_CONCLUSION"


def test_session_mismatch_never_falls_back_to_stale_v17(tmp_path: Path) -> None:
    run_dir = _v15_run(tmp_path, session="20260727")
    pointer_path, pointer_sha = _pointer(tmp_path, session="20260727")
    v17_root = _v17_workspace(
        tmp_path,
        pointer_sha=pointer_sha,
        session="20260724",
    )

    result = run_daily_gray_comparison(
        run_dir=run_dir,
        v17_workspace_root=v17_root,
        market_pointer_path=pointer_path,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )

    assert result["classification"] == "NON_COMPARABLE"
    assert "same_session_v17_shadow_not_found" in result["blockers"]


def test_prior_comparable_rank_sets_mature_local_forward_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = _v15_run(tmp_path)
    pointer_path, pointer_sha = _pointer(tmp_path)
    v17_root = _v17_workspace(tmp_path, pointer_sha=pointer_sha)
    prior_dir = run_dir.parent / "20260602_1200"
    prior_dir.mkdir()
    _write_json(
        prior_dir / OUTPUT_JSON,
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "COMPARABLE",
            "decision_session": "20260601",
            "selection_sets": {
                "v15_candidates": ["000001.SZ", "000002.SZ"],
                "v17_top24": [f"{index:06d}.SZ" for index in range(762, 786)],
            },
        },
    )
    sessions = ["20260601", *[f"202606{day:02d}" for day in range(2, 22)]]

    class FakeReader:
        def __init__(self, **_kwargs):
            pass

        def read_symbol_frames(self, symbols, **_kwargs):
            results = {}
            for symbol in symbols:
                is_v17 = int(symbol[:6]) >= 762
                terminal = 1.20 if is_v17 else 1.10
                closes = [100.0 * (1.0 + (terminal - 1.0) * index / 20) for index in range(21)]
                results[symbol] = SimpleNamespace(
                    frame=pd.DataFrame(
                        {
                            "symbol": [symbol] * 21,
                            "trade_date": sessions,
                            "close": closes,
                        }
                    )
                )
            return results

    monkeypatch.setattr(subject, "MarketDataReader", FakeReader)

    result = run_daily_gray_comparison(
        run_dir=run_dir,
        v17_workspace_root=v17_root,
        market_pointer_path=pointer_path,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )

    document = json.loads((run_dir / OUTPUT_JSON).read_text(encoding="utf-8"))
    effect = document["effect_evaluation"]
    assert result["classification"] == "COMPARABLE"
    assert effect["status"] == "RANK_DIAGNOSTIC_AVAILABLE"
    assert effect["paired_forward_return_observation_count"] == 3
    assert effect["rank_set_aggregates"]["20"]["paired_origin_count"] == 1
    assert effect["rank_set_aggregates"]["20"]["v17_minus_v15_mean_return"] == 0.1
    assert effect["rank_set_verdict"] == "PENDING_MINIMUM_MATURE_SAMPLE"
    assert effect["verdict"] == "NO_V15_V17_PERFORMANCE_CONCLUSION"
