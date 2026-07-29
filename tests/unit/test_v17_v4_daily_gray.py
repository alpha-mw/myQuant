from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from quant_investor.monitoring import v17_v4_daily_gray as subject
from quant_investor.monitoring.v17_v4_daily_gray import (
    OUTPUT_JSON,
    SCHEMA_VERSION,
    run_daily_v4_gray_comparison,
)
from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.research_quant import (
    RESEARCH_FACTOR_DEFINITION_SHA256,
    RESEARCH_FACTOR_NAMES,
    RESEARCH_FACTOR_POLICY_SHA256,
)
from quant_investor.v17_v4_runtime.source_storage import GovernedStore

NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
GRAY_NO_AUTHORITY = {
    "broker_authority": False,
    "execution_authority": False,
    "formal_research_publication_authority": False,
    "order_authority": False,
    "production_default": False,
    "trade_authority": False,
}
STRATEGY = "quant-first"
CUTOFF = "2026-07-28T07:00:00Z"
SESSION = "2026-07-28"
RUN_ID = "shadow-gray-run"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ref(
    document: dict[str, Any],
    *,
    path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(document)
    identity = artifact_identity_field(document["version"])
    return {
        "artifact_id": document[identity],
        "artifact_version": document["version"],
        "byte_sha256": _sha(raw),
        "cutoff": document["cutoff"],
        "relative_path": path,
        "semantic_sha256": document["semantic_sha256"],
        "strategy_id": document["strategy_id"],
    }


def _fake_ref(
    identity: str,
    version: str,
    path: str,
) -> dict[str, str]:
    return {
        "artifact_id": identity,
        "artifact_version": version,
        "byte_sha256": _sha(identity.encode()),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": _sha(version.encode()),
        "strategy_id": STRATEGY,
    }


def _store(
    store: GovernedStore,
    document: dict[str, Any],
    *,
    path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(document)
    store.write_exact_once(path, raw)
    return _ref(document, path=path)


def _shadow_session(
    root: Path,
) -> tuple[
    str,
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    store = GovernedStore(root)
    store.initialize()
    pool = [f"{index:06d}.SZ" for index in range(1, 25)]
    initial_ref = _fake_ref(
        "initial-gray",
        "myquant.v17.v4.initial-pool-output.v1",
        "data/private/v17_v4_runs/shadow-gray-run/initial.json",
    )
    quant = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "branch_kind": "QUANT",
            "canary_evidence_eligible": False,
            "cutoff": CUTOFF,
            "factor_definition_sha256": (
                RESEARCH_FACTOR_DEFINITION_SHA256
            ),
            "factor_mode": "LITERATURE_INCUBATOR_RESEARCH",
            "factor_names": list(RESEARCH_FACTOR_NAMES),
            "factor_policy_sha256": RESEARCH_FACTOR_POLICY_SHA256,
            "factor_rows": [
                {
                    "factor_values": [
                        {
                            "factor_name": factor_name,
                            "value": str(index),
                        }
                        for factor_name in RESEARCH_FACTOR_NAMES
                    ],
                    "symbol": symbol,
                }
                for index, symbol in enumerate(pool, start=1)
            ],
            "formal_activation_eligible": False,
            "incubator_version": "v4-literature-incubator.v10",
            "initial_pool_ref": initial_ref,
            "market_slice_ref": _fake_ref(
                "market-slice-gray",
                "myquant.v17.v4.dataset.quant-factor-input.v1",
                (
                    "data/private/v17_v4_sources/research-quant/"
                    "market.parquet"
                ),
            ),
            "origin": SESSION,
            "output_id": "quant-gray",
            "protocol_version": "myquant.v17.v4",
            "score_rows": [
                {"score": str(index), "symbol": symbol}
                for index, symbol in enumerate(pool, start=1)
            ],
            "shadow_only": True,
            "strategy_id": STRATEGY,
            "version": (
                "myquant.v17.v4.research-quant-branch-output.v1"
            ),
        }
    )
    branch_refs: dict[str, dict[str, str]] = {
        "quant": _store(
            store,
            quant,
            path=(
                "data/private/v17_v4_runs/shadow-gray-run/quant.json"
            ),
        )
    }
    fundamental = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "branch_kind": "FUNDAMENTAL",
            "cutoff": CUTOFF,
            "initial_pool_ref": initial_ref,
            "origin": SESSION,
            "output_id": "fundamental-gray",
            "protocol_version": "myquant.v17.v4",
            "score_rows": [
                {"score": str(index), "symbol": symbol}
                for index, symbol in enumerate(pool, start=1)
            ],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.branch-output.v1",
        }
    )
    branch_refs["fundamental"] = _store(
        store,
        fundamental,
        path=(
            "data/private/v17_v4_runs/shadow-gray-run/"
            "fundamental.json"
        ),
    )
    promotion_ref = _fake_ref(
        "promotion-gray",
        "myquant.v17.v4.fusion-promotion-receipt.v1",
        "data/private/v17_v4_runs/shadow-gray-run/promotion.json",
    )
    fusion = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "output_id": "fusion-gray",
            "promotion_receipt_ref": promotion_ref,
            "protocol_version": "myquant.v17.v4",
            "rows": [
                {
                    "base_target": "0.03",
                    "fused_score": str(25 - index),
                    "rank": index,
                    "symbol": symbol,
                }
                for index, symbol in enumerate(
                    reversed(pool),
                    start=1,
                )
            ],
            "run_id": RUN_ID,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.fusion-top24.v1",
        }
    )
    fusion_ref = _store(
        store,
        fusion,
        path=("data/private/v17_v4_runs/shadow-gray-run/" "fusion.json"),
    )
    deep = seal_semantic(
        {
            "assessment_manifest_ref": _fake_ref(
                "assessment-gray",
                "myquant.v17.v4.deep-assessment-manifest.v1",
                ("data/private/v17_v4_runs/shadow-gray-run/" "assessment.json"),
            ),
            "authority": dict(NO_AUTHORITY),
            "bundle_id": "deep-gray",
            "canary_evidence_eligible": False,
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "formal_activation_eligible": False,
            "fusion_top24_ref": fusion_ref,
            "protocol_version": "myquant.v17.v4",
            "rows": [
                {
                    "blocker_codes": ["official_evidence_unavailable"],
                    "buy_veto": True,
                    "event_scan_ref": None,
                    "issuer_dossier_ref": None,
                    "official_evidence_refs": [],
                    "signal": None,
                    "status": "UNAVAILABLE",
                    "symbol": row["symbol"],
                    "target_after_deep": "0",
                }
                for row in fusion["rows"]
            ],
            "run_id": RUN_ID,
            "scoring_policy_ref": {
                "byte_sha256": "1" * 64,
                "relative_path": ("resources/deep_scoring_policy.v1.json"),
                "semantic_sha256": "2" * 64,
                "version": ("myquant.v17.v4.deep-scoring-policy.v1"),
            },
            "shadow_only": True,
            "state": "DEEP_COMPLETE_SHADOW",
            "strategy_id": STRATEGY,
            "version": ("myquant.v17.v4.deep-evidence-bundle.v2"),
        }
    )
    deep_ref = _store(
        store,
        deep,
        path=("data/private/v17_v4_runs/shadow-gray-run/deep.json"),
    )
    comparison_inputs = {
        "calendar_ref": _fake_ref(
            "calendar-gray",
            "myquant.v17.v4.dataset.cn_open_day_calendar.v1",
            "data/private/v17_v4_sources/calendar.json",
        ),
        "holdings_ref": _fake_ref(
            "holdings-gray",
            "myquant.v17.v4.holdings-snapshot.v1",
            "data/private/v17_v4_runs/shadow-gray-run/holdings.json",
        ),
        "market_bars_ref": _fake_ref(
            "bars-gray",
            "myquant.v17.v4.dataset.market_bars.v1",
            "data/private/v17_v4_sources/bars.json",
        ),
        "source_closure_ref": _fake_ref(
            "catalog-gray",
            "myquant.v17.v4.pit-generation-catalog.v1",
            "data/private/v17_v4_runs/shadow-gray-run/catalog.json",
        ),
    }
    assertion = seal_semantic(
        {
            "assertion_scope": "ONE_RUN_V17_V4_SHADOW_RESEARCH_TRIO",
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "factor_evidence_mode": "RESEARCH_TRIO_SHADOW_ONLY",
            "factor_names": list(RESEARCH_FACTOR_NAMES),
            "factor_policy_sha256": RESEARCH_FACTOR_POLICY_SHA256,
            "formal_activation_eligible": False,
            "operator_asserted": True,
            "override_id": "operator-shadow-trio-20260728",
            "protocol_version": "myquant.v17.v4",
            "shadow_only": True,
            "shadow_run_id": RUN_ID,
            "strategy_id": STRATEGY,
            "version": (
                "myquant.v17.v4.research-factor-shadow-assertion.v1"
            ),
        }
    )
    validate_artifact(assertion)
    assertion_ref = _store(
        store,
        assertion,
        path=(
            "results/v17_v4_shadow/strategies/quant-first/"
            "assertions/operator-shadow-trio-20260728.json"
        ),
    )
    run = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "comparison_inputs": comparison_inputs,
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "deep_bundle_ref": deep_ref,
            "factor_evidence_mode": "RESEARCH_TRIO_SHADOW_ONLY",
            "formal_activation_eligible": False,
            "fundamental_branch_ref": branch_refs["fundamental"],
            "fusion_top24_ref": fusion_ref,
            "initial_pool_ref": initial_ref,
            "model_output_present": True,
            "protocol_version": "myquant.v17.v4",
            "quant_branch_ref": branch_refs["quant"],
            "research_factor_shadow_assertion_ref": assertion_ref,
            "research_quant_factor_names": list(RESEARCH_FACTOR_NAMES),
            "research_quant_factor_policy_sha256": (
                RESEARCH_FACTOR_POLICY_SHA256
            ),
            "shadow_only": True,
            "shadow_run_id": RUN_ID,
            "source_locator_ref": _fake_ref(
                "locator-gray",
                "myquant.v17.v4.preselect-locator.v1",
                ("data/private/v17_v4_runs/shadow-gray-run/" "locator.json"),
            ),
            "state": "SHADOW_COMPLETE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.shadow-run.v2",
        }
    )
    validate_artifact(run)
    run_path = "results/v17_v4_shadow/strategies/quant-first/" "runs/shadow-gray-run.json"
    run_ref = _store(store, run, path=run_path)
    session = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "factor_evidence_mode": "RESEARCH_TRIO_SHADOW_ONLY",
            "formal_activation_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "research_factor_shadow_assertion_ref": assertion_ref,
            "session_ref_id": "shadow-session-20260728",
            "shadow_only": True,
            "shadow_run_ref": run_ref,
            "state": "SHADOW_COMPLETE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.shadow-session-ref.v2",
        }
    )
    validate_artifact(session)
    session_path = "results/v17_v4_shadow/strategies/quant-first/" "sessions/2026-07-28.json"
    session_ref = _store(store, session, path=session_path)
    return (
        session_path,
        session_ref["byte_sha256"],
        comparison_inputs,
        session,
        run,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    path.write_bytes(raw)
    return _sha(raw)


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _v15_run(
    root: Path,
    *,
    comparison_inputs: dict[str, Any] | None,
) -> Path:
    run_dir = root / "records/20260728-1200"
    (run_dir / "raw_exports").mkdir(parents=True)
    manifest: dict[str, Any] = {
        "data_snapshot": {
            "analysis_trade_date": "20260728",
        },
        "timestamp": "20260728-1200",
    }
    if comparison_inputs is not None:
        manifest["v17_v4_comparison_inputs"] = {
            "decision_session": SESSION,
            **comparison_inputs,
        }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "market_snapshot.json",
        {"analysis_trade_date": "20260728"},
    )
    _write_csv(
        run_dir / "candidate_pool.csv",
        [
            {"symbol": "000024.SZ"},
            {"symbol": "000001.SZ"},
        ],
    )
    _write_csv(
        run_dir / "holdings_review.csv",
        [{"symbol": "000024.SZ"}],
    )
    (run_dir / "analysis_report.md").write_text(
        "# V15 report\n",
        encoding="utf-8",
    )
    return run_dir


def _pointer(root: Path) -> tuple[Path, str]:
    path = root / "data/parquet/cn/_latest.json"
    sha = _write_json(
        path,
        {
            "latest_complete_trade_date": "20260728",
            "manifest_path": ("data/parquet/cn/_snapshots/current.json"),
            "table_root": ("data/parquet/cn/_snapshots/current/table/bars"),
        },
    )
    return path, sha


def test_v4_gray_uses_explicit_session_ref_and_exact_bindings(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    root = tmp_path.resolve()
    session_path, session_sha, bindings, session, run = _shadow_session(root)
    monkeypatch.setattr(
        subject,
        "read_shadow_session",
        lambda *_args, **_kwargs: {
            "session": session,
            "session_path": session_path,
            "session_sha256": session_sha,
            "shadow_run": run,
        },
    )
    run_dir = _v15_run(root, comparison_inputs=bindings)
    pointer, pointer_sha = _pointer(root)
    result = run_daily_v4_gray_comparison(
        run_dir=run_dir,
        workspace_root=root,
        shadow_session_ref_path=session_path,
        expected_shadow_session_ref_sha256=session_sha,
        market_pointer_path=pointer,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )
    assert result["status"] == "GRAY_COMPARISON_COMPLETE"
    assert result["classification"] == "COMPARABLE"
    assert result["metrics"]["candidate_overlap_count"] == 2
    assert result["metrics"]["v17_deep_buy_veto_count"] == 24
    document = json.loads((run_dir / OUTPUT_JSON).read_text())
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["observation_only"] is True
    assert document["canary_evidence_eligible"] is False
    assert document["historical_policy_eligible"] is False
    assert document["effect_evaluation"]["verdict"] == "NO_V15_V17_V4_PERFORMANCE_CONCLUSION"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "v17_v4_gray_comparison" in manifest
    assert "v17_gray_comparison" not in manifest


def test_v4_gray_missing_explicit_ref_is_non_comparable(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    run_dir = _v15_run(root, comparison_inputs=None)
    pointer, pointer_sha = _pointer(root)
    result = run_daily_v4_gray_comparison(
        run_dir=run_dir,
        workspace_root=root,
        shadow_session_ref_path=None,
        expected_shadow_session_ref_sha256=None,
        market_pointer_path=pointer,
        pointer_sha256_before_v15=pointer_sha,
        pointer_sha256_after_v15=pointer_sha,
    )
    assert result["classification"] == "NON_COMPARABLE"
    assert "explicit_v4_session_ref_pair_missing" in result["blockers"]


def _business_sessions(start: date, count: int) -> list[str]:
    sessions: list[str] = []
    cursor = start
    while len(sessions) < count:
        if cursor.weekday() < 5:
            sessions.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return sessions


def test_close_return_labels_are_append_only_and_evidence_bound(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    root = tmp_path.resolve()
    base_dir = root / "records"
    prior = base_dir / "20260701-1200"
    prior.mkdir(parents=True)
    candidates = ["000001.SZ", "000002.SZ"]
    top24 = [f"{index:06d}.SZ" for index in range(1, 25)]
    _write_json(
        prior / OUTPUT_JSON,
        {
            "classification": "COMPARABLE",
            "decision_session": "20260701",
            "schema_version": SCHEMA_VERSION,
            "strategy_id": STRATEGY,
            "selection_sets": {
                "v15_candidates": candidates,
                "v17_top24": top24,
            },
        },
    )
    pointer_path = root / "data/parquet/cn/_latest.json"
    _write_json(
        pointer_path,
        {
            "manifest_path": ("data/parquet/cn/_snapshots/current.json"),
            "table_root": ("data/parquet/cn/_snapshots/current/table/bars"),
        },
    )
    manifest_path = root / "data/parquet/cn/_snapshots/current.json"
    _write_json(manifest_path, {"snapshot_id": "current"})
    part = root / "data/parquet/cn/_snapshots/current/table/bars/" "year=2026/month=07/part.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"sealed parquet bytes")
    sessions = _business_sessions(date(2026, 7, 1), 21)

    class FakeReader:
        def __init__(self, **_: Any) -> None:
            pass

        def read_symbol_frames(
            self,
            symbols: list[str],
            **_: Any,
        ) -> dict[str, Any]:
            return {
                symbol: SimpleNamespace(
                    frame=pd.DataFrame(
                        {
                            "trade_date": sessions,
                            "close": [100 + index for index in range(len(sessions))],
                        }
                    )
                )
                for symbol in symbols
            }

    monkeypatch.setattr(subject, "MarketDataReader", FakeReader)
    first = subject._write_mature_labels(
        workspace_root=root,
        base_dir=base_dir,
        current_session=sessions[-1],
        market_pointer_path=pointer_path,
        strategy_id=STRATEGY,
    )
    second = subject._write_mature_labels(
        workspace_root=root,
        base_dir=base_dir,
        current_session=sessions[-1],
        market_pointer_path=pointer_path,
        strategy_id=STRATEGY,
    )
    assert len(first) == len(second) == 3
    assert all(row["created"] for row in first)
    assert not any(row["created"] for row in second)
    assert [row["byte_sha256"] for row in first] == [row["byte_sha256"] for row in second]
    label_path = root / first[-1]["relative_path"]
    label = json.loads(label_path.read_text())
    assert label["horizon_sessions"] == 20
    assert label["total_return"] is False
    assert label["performance_conclusion_eligible"] is False
    assert label["market_evidence"]["part_parquet_refs"][0]["sha256"] == _sha(
        b"sealed parquet bytes"
    )
