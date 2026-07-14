from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_v14_retirement_replay_gate as replay_gate
from scripts.run_v14_retirement_replay_gate import (
    RetirementReplayError,
    build_replay_report,
    evaluate_no_new_buy,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    context = {
        "rebalance_date": "20260605",
        "latest_trade_date": "20260605",
        "industry_map": {"A": "bank"},
        "liquidity_filter": {"liquidity_scores": {"A": 1.0}},
        "macro_regime": "neutral",
        "cross_section_quant": {"breadth": 0.2},
        "risk_budget": {"sector_bucket_limit": 2},
        "metadata": {
            "selection_profile": {
                "funnel_profile": "momentum_leader",
                "trend_windows": [20, 60, 120],
                "volume_spike_threshold": 1.35,
                "breakout_distance_pct": 0.06,
                "max_candidates": 20,
                "sector_bucket_limit": 2,
            },
            "symbol_market_state": {
                "A": {
                    "momentum_strength": 0.4,
                    "fake_breakout_risk": 0.2,
                }
            },
            "candidate_sector_counts": {"bank": 1},
        },
    }
    summary = {
        "dag": {
            "candidate_symbols": ["A"],
            "portfolio_decision": {
                "shortlist": [{"symbol": "A", "action": "buy"}],
                "execution_trace": {
                    "steps": [
                        {},
                        {"metadata": {"global_context": context}},
                    ]
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    actions_path = tmp_path / "actions.csv"
    with actions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "name",
                "quant_score",
                "quant_confidence",
                "fundamental_score",
                "fundamental_confidence",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "symbol": "A",
                "name": "Alpha",
                "quant_score": 0.2,
                "quant_confidence": 0.6,
                "fundamental_score": 0.1,
                "fundamental_confidence": 0.6,
            }
        )
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    actions_sha = hashlib.sha256(actions_path.read_bytes()).hexdigest()
    return summary_path, actions_path, summary_sha, actions_sha


def test_replay_gate_uses_only_two_likelihoods_and_adds_no_buy(
    tmp_path,
    monkeypatch,
):
    summary_path, actions_path, summary_sha, actions_sha = _write_fixture(
        tmp_path
    )

    monkeypatch.setattr(
        replay_gate,
        "_verify_candidate_state",
        lambda commit: {
            "repository_root": "/verified",
            "head_commit": commit,
            "worktree_clean": True,
        },
    )
    report = build_replay_report(
        summary_path=summary_path,
        actions_path=actions_path,
        candidate_commit="a" * 40,
        expected_summary_sha256=summary_sha,
        expected_actions_sha256=actions_sha,
    )

    assert report["status"] == "passed"
    assert report["canonical_branch_order"] == [
        "quant",
        "fundamental",
        "macro",
    ]
    assert report["canonical_likelihood_order"] == [
        "quant",
        "fundamental",
    ]
    assert set(report["replay_rows"][0]["likelihoods"]) == {
        "schema_version",
        "quant_likelihood",
        "fundamental_likelihood",
        "correlation_matrix",
    }
    assert report["comparison"]["new_buy_symbols"] == []
    assert report["gates"]["candidate_commit_bound"] is True
    assert report["replay_scope"] == {
        "name": "frozen_candidate_set_pre_control",
        "full_universe_funnel_replayed": False,
        "control_chain_replayed": False,
        "interpretation": (
            "no new BUY inside the hash-bound frozen candidate set"
        ),
    }


def test_replay_gate_rejects_drifted_frozen_evidence(
    tmp_path,
    monkeypatch,
):
    summary_path, actions_path, summary_sha, actions_sha = _write_fixture(
        tmp_path
    )
    summary_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        replay_gate,
        "_verify_candidate_state",
        lambda commit: {
            "repository_root": "/verified",
            "head_commit": commit,
            "worktree_clean": True,
        },
    )

    with pytest.raises(
        RetirementReplayError,
        match="replay_evidence_sha256_mismatch",
    ):
        build_replay_report(
            summary_path=summary_path,
            actions_path=actions_path,
            candidate_commit="b" * 40,
            expected_summary_sha256=summary_sha,
            expected_actions_sha256=actions_sha,
        )


def test_no_new_buy_comparison_fails_for_target_only_buy():
    result = evaluate_no_new_buy(
        baseline_buy_symbols={"A"},
        target_buy_symbols={"A", "B"},
    )

    assert result["passed"] is False
    assert result["new_buy_symbols"] == ["B"]


def test_replay_gate_rejects_unbound_candidate_state(
    tmp_path,
    monkeypatch,
):
    summary_path, actions_path, summary_sha, actions_sha = _write_fixture(
        tmp_path
    )

    def reject(_commit: str):
        raise RetirementReplayError("candidate_worktree_not_clean")

    monkeypatch.setattr(
        replay_gate,
        "_verify_candidate_state",
        reject,
    )
    with pytest.raises(
        RetirementReplayError,
        match="candidate_worktree_not_clean",
    ):
        build_replay_report(
            summary_path=summary_path,
            actions_path=actions_path,
            candidate_commit="c" * 40,
            expected_summary_sha256=summary_sha,
            expected_actions_sha256=actions_sha,
        )
