from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    GlobalContext,
    ICDecision,
    PortfolioPlan,
    RiskDecision,
    ShortlistItem,
)
from quant_investor.market.dag import decision as decision_module
from quant_investor.market.dag.decision import _run_portfolio_construction_phase
from quant_investor.themes.protocol_v2 import evaluate_theme_protocol_v2
from quant_investor.themes.taxonomy import ThemeTaxonomy


SYMBOL = "S001.SZ"
THEME_ID = "tech::ai"


def _membership() -> dict[str, object]:
    return {
        "schema_version": "theme_membership.v2",
        "membership_id": "ai-s001",
        "theme_id": THEME_ID,
        "theme_name": "AI",
        "theme_type": "technology",
        "symbol": SYMBOL,
        "taxonomy_node_id": THEME_ID,
        "supply_chain_role": "application",
        "revenue_exposure": None,
        "effective_from": "2026-01-01",
        "available_at": "2026-07-01",
        "confidence": 0.9,
        "source_type": "local_test",
        "source_ref": "local://theme-membership-test",
    }


def _protocol() -> dict[str, object]:
    return evaluate_theme_protocol_v2(
        theme_scores={
            THEME_ID: {
                "theme_id": THEME_ID,
                "theme_name": "AI",
                "score": 80.0,
                "attention": 0.8,
                "attention_5d": 0.8,
                "attention_20d": 0.8,
                "attention_60d": 0.8,
                "attention_120d": 0.8,
                "attention_history_coverage": 1.0,
                "market_confirmation": 0.8,
                "breadth": 0.8,
                "momentum": 0.8,
                "volume_confirmation": 0.8,
                "acceleration": 0.8,
                "crowding_risk": 0.2,
                "crowding_status": "success",
                "valuation_risk": 0.2,
                "valuation_risk_status": "fresh",
                "member_count": 1,
            }
        },
        taxonomy=ThemeTaxonomy.load(),
        as_of="2026-07-10",
        evidence_events=[
            {
                "schema_version": "theme_evidence_event.v1",
                "event_id": "ai-order",
                "theme_id": THEME_ID,
                "event_type": "order",
                "event_date": "2026-07-01",
                "available_at": "2026-07-01",
                "direction": 1,
                "strength": 0.9,
                "confidence": 0.9,
            }
        ],
        theme_membership_details=[_membership()],
        previous_states={THEME_ID: {"lifecycle": "validated_trend"}},
        markov_regime="趋势上涨",
        formal_enabled=True,
        formal_kill_switch=False,
        valid_trading_dates=[
            "2026-07-06",
            "2026-07-07",
            "2026-07-08",
            "2026-07-09",
            "2026-07-10",
        ],
    )


class _ICCoordinator:
    def run(self, payload):
        del payload
        return ICDecision(action=ActionLabel.BUY)


def _attach(decision, **kwargs):
    del kwargs
    return decision


def _write_manifest_placeholder(tmp_path: Path) -> tuple[Path, str]:
    rendered = b"canonical verifier is explicitly monkeypatched in this test\n"
    path = tmp_path / "private" / "joint_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)
    path.chmod(0o600)
    return path, hashlib.sha256(rendered).hexdigest()


def _run_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: str = "",
    persist_enabled: bool = True,
    producer_available: bool = True,
    verifier_blockers: tuple[str, ...] = (),
    cap_proof: str = "valid",
):
    protocol = _protocol()
    assert protocol["status"] == "prequalified"
    assert protocol["formal_pool"] == []
    rotation = {
        "status": "success",
        "as_of": "2026-07-10",
        "protocol_v2": protocol,
        "symbol_scores": {SYMBOL: 0.8},
        "symbol_primary_theme": {SYMBOL: THEME_ID},
        "symbol_theme_memberships": {SYMBOL: [THEME_ID]},
        "symbol_theme_membership_details": {SYMBOL: [_membership()]},
        "symbol_phase": {SYMBOL: "confirmed_rotation"},
        "symbol_risk_flags": {SYMBOL: []},
        "theme_scores": {THEME_ID: {"theme_name": "AI"}},
    }
    context = GlobalContext(
        risk_budget={
            "target_exposure": 0.5,
            "max_single_weight": 0.2,
            "sector_bucket_limit": 0,
        },
        metadata={"theme_rotation": rotation},
        universe_hash="dag-theme-test",
    )
    shortlist = [
        ShortlistItem(
            symbol=SYMBOL,
            action=ActionLabel.HOLD if missing == "edge" else ActionLabel.BUY,
            rank_score=0.8,
            metadata={
                "posterior_edge_after_costs": 0.0 if missing == "edge" else 0.05,
            },
        )
    ]
    tradability = {
        SYMBOL: {
            "tradable": missing != "tradability",
            "source_path": "" if missing == "data" else "/local/parquet/s001",
            "data_quality_issue_count": 0,
            "liquidity_score": 0.0 if missing == "liquidity" else 0.8,
            "industry": "Technology",
        }
    }

    class _RiskGuard:
        def run(self, payload):
            del payload
            return RiskDecision(
                status=AgentStatus.SUCCESS,
                action_cap=ActionLabel.BUY,
                hard_veto=False,
                veto=False,
                blocked_symbols=[SYMBOL] if missing == "risk" else [],
                max_weight=0.2,
                gross_exposure_cap=0.5,
            )

    class _PortfolioConstructor:
        def run(self, payload):
            del payload
            weight = 0.0 if missing == "portfolio" else 0.05
            valid_lane = {
                "status": "active",
                "applied": True,
                "protocol_hash": str(protocol["protocol_hash"]),
            }
            metadata = {
                "theme_portfolio_cap_enabled": True,
                "theme_portfolio_diagnostic_notes": [],
                "theme_tactical_lane": valid_lane,
            }
            if cap_proof == "missing":
                metadata = {}
            elif cap_proof == "disabled":
                metadata["theme_portfolio_cap_enabled"] = False
            elif cap_proof == "malformed":
                metadata["theme_portfolio_diagnostic_notes"] = [
                    "theme_tactical_lane_malformed"
                ]
                metadata["theme_tactical_lane"] = {
                    **valid_lane,
                    "status": "blocked_malformed",
                }
            elif cap_proof == "protocol_mismatch":
                metadata["theme_tactical_lane"] = {
                    **valid_lane,
                    "protocol_hash": "f" * 64,
                }
            return PortfolioPlan(
                status=AgentStatus.SUCCESS,
                target_exposure=weight,
                target_gross_exposure=weight,
                target_net_exposure=weight,
                cash_ratio=1.0 - weight,
                target_weights={SYMBOL: weight},
                metadata=metadata,
            )

    monkeypatch.setattr(
        decision_module.config,
        "THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED",
        persist_enabled,
    )
    monkeypatch.setattr(
        decision_module.config,
        "THEME_FORMAL_RECONCILIATION_DIR",
        str(tmp_path / "private" / "theme_reconciliation"),
    )
    monkeypatch.setattr(
        decision_module.config,
        "THEME_PORTFOLIO_CAP_ENABLED",
        True,
    )
    monkeypatch.setattr(
        decision_module.replay_v13_1,
        "CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE",
        producer_available,
    )
    manifest_path, manifest_sha = _write_manifest_placeholder(tmp_path)
    monkeypatch.setattr(
        decision_module.config,
        "THEME_V2_JOINT_MANIFEST_PATH",
        str(manifest_path),
    )
    monkeypatch.setattr(
        decision_module.config,
        "THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256",
        manifest_sha,
    )
    if producer_available:
        def _verified_manifest(
            path,
            *,
            expected_artifact_sha256,
            expected_theme_protocol_hash,
        ):
            assert str(path) == str(manifest_path)
            assert expected_artifact_sha256 == manifest_sha
            assert expected_theme_protocol_hash == protocol["protocol_hash"]
            return {
                "ready": not verifier_blockers,
                "status": "ready" if not verifier_blockers else "blocked",
                "blockers": list(verifier_blockers),
            }

        monkeypatch.setattr(
            decision_module.replay_v13_1,
            "verify_joint_replay_manifest",
            _verified_manifest,
        )
    else:
        def _unexpected_verifier(*args, **kwargs):
            del args, kwargs
            raise AssertionError("producer=false must block before verifier")

        monkeypatch.setattr(
            decision_module.replay_v13_1,
            "verify_joint_replay_manifest",
            _unexpected_verifier,
        )
    state = _run_portfolio_construction_phase(
        shortlist=shortlist,
        branch_summaries={},
        macro_verdict=BranchVerdict(),
        global_context=context,
        data_quality_issues=[],
        ic_hints_by_symbol={},
        research_by_symbol={SYMBOL: {}},
        tradability_snapshot=tradability,
        funnel_summary={},
        bayesian_records=[],
        candidate_symbols=[SYMBOL],
        portfolio_master_output=None,
        portfolio_master_meta={},
        risk_guard_cls=_RiskGuard,
        ic_coordinator_cls=_ICCoordinator,
        portfolio_constructor_cls=_PortfolioConstructor,
        attach_symbol_to_ic_decision_fn=_attach,
    )
    state._test_context = context
    state._test_shortlist = shortlist
    state._test_tradability = tradability
    return state


def test_real_portfolio_dag_phase_produces_persisted_formal_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _run_phase(tmp_path, monkeypatch)
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]
    persistence = state.portfolio_decision.metadata[
        "theme_formal_reconciliation_persistence"
    ]

    assert artifact["status"] == "formal"
    assert artifact["formal_pool"] == [THEME_ID]
    assert artifact["formal_symbols"] == [SYMBOL]
    assert persistence["status"] == "persisted"
    assert persistence["readback_verified"] is True
    path = Path(persistence["path"])
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert state.portfolio_decision.target_weights == {SYMBOL: 0.05}


def test_post_control_reconciliation_is_idempotent_without_protocol_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _run_phase(tmp_path, monkeypatch)
    context = state._test_context
    protocol = context.metadata["theme_rotation"]["protocol_v2"]
    protocol_artifact_hash = protocol["artifact_hash"]
    first = state.portfolio_decision.metadata["theme_formal_reconciliation"]

    second = decision_module._post_control_theme_reconciliation(
        global_context=context,
        shortlist=state._test_shortlist,
        tradability_snapshot=state._test_tradability,
        risk_decision=state.risk_decision,
        portfolio_plan=state.portfolio_plan,
    )

    assert second["reconciliation_hash"] == first["reconciliation_hash"]
    assert protocol["artifact_hash"] == protocol_artifact_hash
    assert protocol["status"] == "prequalified"
    assert protocol["formal_pool"] == []
    assert context.metadata["theme_formal_reconciliation_persistence"][
        "status"
    ] == "idempotent_readback"


def test_formal_pool_is_blocked_when_durable_persistence_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _run_phase(
        tmp_path,
        monkeypatch,
        persist_enabled=False,
    )
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]
    persistence = state.portfolio_decision.metadata[
        "theme_formal_reconciliation_persistence"
    ]

    assert artifact["status"] == "blocked"
    assert artifact["formal_pool"] == []
    assert artifact["blockers"] == ["formal_reconciliation_persistence_disabled"]
    assert persistence["status"] == "disabled"
    assert state.portfolio_decision.target_weights == {}


@pytest.mark.parametrize(
    ("missing", "blocker"),
    [
        ("data", "data_gate_blocked"),
        ("tradability", "tradability_gate_blocked"),
        ("liquidity", "liquidity_gate_blocked"),
        ("edge", "positive_edge_buy_gate_blocked"),
        ("risk", "risk_guard_blocked"),
        ("portfolio", "portfolio_constructor_blocked"),
    ],
)
def test_real_portfolio_dag_phase_missing_gate_keeps_formal_pool_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    blocker: str,
) -> None:
    state = _run_phase(tmp_path, monkeypatch, missing=missing)
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]

    assert artifact["status"] == "valid_empty"
    assert artifact["formal_pool"] == []
    assert blocker in artifact["per_symbol"][SYMBOL]["blockers"]
    persistence = state.portfolio_decision.metadata[
        "theme_formal_reconciliation_persistence"
    ]
    assert persistence["readback_verified"] is True
    assert state.portfolio_decision.target_weights == {}


def test_runtime_blocks_forged_ready_manifest_when_canonical_producer_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _run_phase(
        tmp_path,
        monkeypatch,
        producer_available=False,
    )
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]

    assert artifact["status"] == "blocked"
    assert artifact["formal_pool"] == []
    assert artifact["blockers"] == [
        "canonical_joint_replay_producer_not_implemented"
    ]
    assert state.portfolio_decision.target_weights == {}


def test_canonical_manifest_verifier_must_bind_current_theme_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _run_phase(
        tmp_path,
        monkeypatch,
        verifier_blockers=("joint_manifest_protocol_hash_mismatch",),
    )
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]

    assert artifact["status"] == "blocked"
    assert "joint_manifest_protocol_hash_mismatch" in artifact["blockers"]
    assert artifact["formal_pool"] == []
    assert state.portfolio_decision.target_weights == {}


@pytest.mark.parametrize(
    ("cap_proof", "expected_blocker"),
    [
        ("missing", "theme_portfolio_cap_not_applied"),
        ("disabled", "theme_portfolio_cap_not_applied"),
        ("malformed", "theme_portfolio_cap_malformed_diagnostic"),
        ("protocol_mismatch", "theme_tactical_lane_protocol_hash_mismatch"),
    ],
)
def test_missing_or_invalid_theme_cap_execution_proof_clears_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap_proof: str,
    expected_blocker: str,
) -> None:
    state = _run_phase(
        tmp_path,
        monkeypatch,
        cap_proof=cap_proof,
    )
    artifact = state.portfolio_decision.metadata["theme_formal_reconciliation"]

    assert artifact["status"] == "blocked"
    assert expected_blocker in artifact["blockers"]
    assert artifact["formal_pool"] == []
    assert state.portfolio_decision.target_weights == {}
