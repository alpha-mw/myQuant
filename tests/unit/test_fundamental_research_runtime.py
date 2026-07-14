from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
from pydantic import ValidationError

from quant_investor.fundamental_research.ledger import HashChainLedger
from quant_investor.fundamental_research.models import (
    ApplicationEventV1,
    ApplicationState,
    ClaimKind,
    ClaimV1,
    Dimension,
    DimensionAssessmentV1,
    DimensionContributionV1,
    DimensionSignal,
    FundamentalOverlayV1,
    FundamentalResearchDossierV1,
    FundamentalResearchRequestV1,
    FundamentalResearchResponseV1,
    JobEventV1,
    JobState,
    SourceRecordV1,
    SourceTier,
)
from quant_investor.fundamental_research.governance import (
    ActivationGateEvidenceV2,
    HoldingsScopeSnapshotV1,
    LongitudinalObservationV1,
    LongitudinalOutcomeArtifactV1,
    LongitudinalSourceArtifactV1,
    append_longitudinal_observation,
    build_activation_gate_evidence,
    verify_recomputed_evidence,
)
from quant_investor.fundamental_research.runtime import (
    APPLICATION_LEDGER,
    ActivationManifestV1,
    consume_overlay,
)
from quant_investor.fundamental_research.longitudinal_producer import (
    produce_nav_attribution_observation,
    produce_target_weight_observation,
)
from quant_investor.market.report_persistence import write_analysis_run_manifest
from quant_investor.fundamental_research.storage import (
    atomic_write_json_model,
    canonical_json_bytes,
    sha256_bytes,
)

UTC = timezone.utc
# 15:00 Asia/Shanghai, the CN decision boundary used by date-only DAG runs.
DECISION_CUTOFF = datetime(2026, 7, 10, 7, tzinfo=UTC)
GENERATION = "fundamental-generation-1"


def _control_chain_replay(*, variant: str, value: float) -> dict[str, object]:
    risk_decision = {
        "status": "success",
        "action_cap": "buy",
        "gross_exposure_cap": 0.2,
        "target_exposure_cap": 0.2,
        "max_weight": 0.2,
        "position_limits": {"600000.SH": 0.2},
        "blocked_symbols": [],
    }
    shortlist = [
        {
            "symbol": "600000.SH",
            "rank_score": 0.4,
            "action": "buy",
            "confidence": 0.6,
            "expected_upside": 0.1,
        }
    ]
    portfolio_decision = {
        "target_weights": {"600000.SH": value},
        "target_exposure": value,
        "shortlist": shortlist,
        "risk_constraints": {"risk_decision": risk_decision},
        "metadata": {"replay_variant": variant},
    }
    fundamental_metadata: dict[str, object] = {"fundamental_research_variant": variant}
    if variant == "with_dossier":
        fundamental_metadata["fundamental_research_runtime"] = {
            "request_id": "req-600000",
            "dossier_id": "dossier-600000",
            "measurement_only": True,
            "applied": True,
            "counterfactual": False,
        }
    return {
        "schema_version": "fundamental-control-chain-replay.v1",
        "measurement_only": True,
        "variant": variant,
        "branch_summaries": {"fundamental": {"final_score": 0.4, "final_confidence": 0.6}},
        "branch_verdicts_by_symbol": {
            "600000.SH": {
                "fundamental": {
                    "final_score": 0.4,
                    "final_confidence": 0.6,
                    "metadata": fundamental_metadata,
                }
            }
        },
        "bayesian_records": [
            {
                "symbol": "600000.SH",
                "posterior_action_score": 0.4,
                "posterior_expected_alpha": 0.1,
                "posterior_confidence": 0.6,
                "posterior_edge_after_costs": 0.08,
                "metadata": {
                    "fundamental_research_variant": variant,
                    "fundamental_score": 0.4,
                },
            }
        ],
        "shortlist": shortlist,
        "ic_hints_by_symbol": {},
        "risk_decision": risk_decision,
        "ic_decisions": [
            {
                "symbol": "600000.SH",
                "final_score": 0.4,
                "final_confidence": 0.6,
                "action": "buy",
                "metadata": {"llm_hint_applied": False},
            }
        ],
        "portfolio_plan": {
            "target_weights": {"600000.SH": value},
            "target_exposure": value,
            "position_limits": {"600000.SH": 0.2},
            "blocked_symbols": [],
            "rejected_symbols": [],
        },
        "portfolio_decision": portfolio_decision,
    }


def _request(symbol: str = "600000.SH") -> FundamentalResearchRequestV1:
    base = 0.2
    return FundamentalResearchRequestV1(
        request_id="req-600000",
        run_id="CN-20260710",
        symbol=symbol,
        company_name="浦发银行",
        decision_cutoff=DECISION_CUTOFF,
        created_at=DECISION_CUTOFF,
        expires_at=DECISION_CUTOFF + timedelta(days=30),
        base_score=base,
        base_score_sha256=sha256_bytes(canonical_json_bytes({"base_score": base})),
        git_sha="a" * 40,
        data_generation=GENERATION,
        selection_reasons=["funnel_top_k"],
        prompt_version="fundamental-v1",
        policy_version="fundamental-policy-v1",
        source_policy_sha256="b" * 64,
    )


def _overlay(symbol: str = "600000.SH") -> FundamentalOverlayV1:
    return FundamentalOverlayV1(
        request_id="req-600000",
        dossier_id="dossier-600000",
        symbol=symbol,
        base_score=0.2,
        computed_delta=0.08,
        adjusted_score=0.28,
        eligible=True,
        contributions=[
            DimensionContributionV1(
                dimension=dimension,
                signal=DimensionSignal.POSITIVE,
                qualified=True,
                weight=0.1,
                contribution=0.05,
            )
            for dimension in Dimension
        ],
    )


def _response() -> FundamentalResearchResponseV1:
    source = SourceRecordV1(
        source_id="source-1",
        publisher="Shanghai Stock Exchange",
        document_kind="annual_report",
        canonical_url="https://www.sse.com.cn/source-1",
        published_at=DECISION_CUTOFF - timedelta(days=2),
        first_available_at=DECISION_CUTOFF - timedelta(days=2),
        retrieved_at=DECISION_CUTOFF + timedelta(minutes=30),
        source_tier=SourceTier.PRIMARY,
        content_sha256="c" * 64,
        locator="p.1",
        evidence_extract="verified extract",
    )
    claims = [
        ClaimV1(
            claim_id=f"claim-{index}",
            kind=ClaimKind.FACT,
            dimension=dimension,
            statement=f"Evidence for {dimension.value}",
            direction="positive",
            materiality=0.5,
            supporting_source_ids=[source.source_id],
        )
        for index, dimension in enumerate(Dimension)
    ]
    dossier = FundamentalResearchDossierV1(
        dossier_id="dossier-600000",
        request_id="req-600000",
        symbol="600000.SH",
        company_name="浦发银行",
        decision_cutoff=DECISION_CUTOFF,
        produced_at=DECISION_CUTOFF + timedelta(minutes=45),
        model_name="codex",
        prompt_version="fundamental-v1",
        sources=[source],
        claims=claims,
        dimensions=[
            DimensionAssessmentV1(
                dimension=claim.dimension,
                signal=DimensionSignal.POSITIVE,
                claim_ids=[claim.claim_id],
            )
            for claim in claims
        ],
        bull_case=[claims[0].claim_id],
        key_risks=[claims[-1].claim_id],
    )
    request = _request()
    return FundamentalResearchResponseV1(
        request_id=request.request_id,
        request_sha256=sha256_bytes(canonical_json_bytes(request.model_dump(mode="json"))),
        dossier=dossier,
    )


def _validated_artifacts(root: Path, *, validated_at: datetime | None = None) -> None:
    run = root / "CN" / "2026-07-10" / "CN-20260710"
    request_sha = atomic_write_json_model(
        root, run / "requests" / "600000.SH.request.v1.json", _request()
    )
    response = _response()
    response_sha = atomic_write_json_model(
        root, run / "responses" / "600000.SH.response.v1.json", response
    )
    dossier_sha = atomic_write_json_model(
        root, run / "dossiers" / "600000.SH.dossier.v1.json", response.dossier
    )
    overlay_sha = atomic_write_json_model(
        root, run / "overlays" / "600000.SH.overlay.v1.json", _overlay()
    )
    ledger = HashChainLedger(root, root / "state" / "jobs.v1.jsonl")
    event = JobEventV1(
        event_id="job-validated",
        request_id="req-600000",
        state=JobState.VALIDATED,
        occurred_at=validated_at or DECISION_CUTOFF + timedelta(hours=1),
        request_sha256=request_sha,
        response_sha256=response_sha,
        dossier_sha256=dossier_sha,
        overlay_sha256=overlay_sha,
    )
    ledger.append(event, expected_head="")


def _env(root: Path, mode: str = "shadow") -> dict[str, str]:
    return {
        "FUNDAMENTAL_RESEARCH_ROOT": str(root),
        "FUNDAMENTAL_RESEARCH_OVERLAY_MODE": mode,
    }


def _gate_evidence() -> ActivationGateEvidenceV2:
    dates = [(DECISION_CUTOFF + timedelta(days=index)).date() for index in range(10)]
    return ActivationGateEvidenceV2(
        generated_at=datetime(2026, 7, 10, 11, tzinfo=UTC),
        holdings_snapshot_path="state/holdings-scope.v1.json",
        holdings_snapshot_sha256="d" * 64,
        validated_request_ids=[f"req-{index:02d}" for index in range(30)],
        validated_symbols=[f"600{index:03d}.SH" for index in range(10)],
        validated_company_names=[f"Company {index:02d}" for index in range(10)],
        validated_industries=["bank", "industrial", "technology"],
        holdings_symbols=["600000.SH"],
        shadow_trading_dates=dates,
        target_weight_counterfactual_dates=dates,
        recent_validation_success_count=0,
    )


def _write_gate_holdings_snapshot(root: Path, repo_root: Path) -> tuple[str, str]:
    manual = repo_root / "manual"
    manual.mkdir(parents=True, exist_ok=True)
    ledger = manual / "ledger_after_manual_switch.parquet"
    pd.DataFrame([{"symbol": "600000.SH", "shares": 100}]).to_parquet(ledger, index=False)
    ledger_sha = sha256_bytes(ledger.read_bytes())
    manifest = manual / "manual_execution_manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "cn_aggressive_manual_execution.v2",
                "status": "no_action_carry_forward",
                "recorded_at": "2026-07-10T10:00:00+00:00",
                "ledger_after_manual_switch_parquet": ledger.name,
                "ledger_after_manual_switch_parquet_sha256": ledger_sha,
            }
        )
    )
    atomic_write_json_model(
        root,
        root / "state" / "holdings-scope.v1.json",
        HoldingsScopeSnapshotV1(
            generated_at=datetime(2026, 7, 10, 11, tzinfo=UTC),
            symbols=["600000.SH"],
            manual_manifest_repo_path="manual/manual_execution_manifest.json",
            manual_ledger_repo_path="manual/ledger_after_manual_switch.parquet",
            manual_manifest_sha256=sha256_bytes(manifest.read_bytes()),
            manual_ledger_sha256=ledger_sha,
        ),
    )
    return sha256_bytes(manifest.read_bytes()), ledger_sha


def _consume(root: Path, *, cutoff: str, mode: str = "shadow", base: float = 0.2):
    return consume_overlay(
        symbol="600000.SH",
        base_score=base,
        run_cutoff=cutoff,
        run_key=f"CN:full_a:{cutoff}",
        current_data_generation=GENERATION,
        env=_env(root, mode),
    )


def test_shadow_is_prior_run_only_and_consumed_exactly_once(tmp_path: Path) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)

    same_run = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260710",
        run_key="CN:full_a:20260710",
        current_data_generation=GENERATION,
        env=_env(root),
        occurred_at=DECISION_CUTOFF + timedelta(hours=2),
    )
    assert same_run.applied is False
    assert "job_not_validated_at_run_cutoff" in same_run.metadata["blockers"]
    assert not (root / APPLICATION_LEDGER).exists()

    shadow = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260711",
        run_key="CN:full_a:20260711",
        current_data_generation=GENERATION,
        env=_env(root),
        occurred_at=DECISION_CUTOFF + timedelta(days=1),
    )
    assert shadow.effective_mode == "shadow"
    assert shadow.applied is False
    assert shadow.metadata["counterfactual"] is True
    assert shadow.metadata["counterfactual_adjusted_score"] == 0.28
    assert shadow.metadata["dossier_summary"]["bull_case"]
    assert shadow.suppress_generic_fundamental_overlay is False

    replay = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260711",
        run_key="CN:full_a:20260711",
        current_data_generation=GENERATION,
        env=_env(root),
    )
    assert replay.applied is False
    assert replay.metadata["counterfactual"] is True
    assert replay.metadata["idempotent_replay"] is True
    assert replay.metadata["blockers"] == []
    assert len(HashChainLedger(root, root / APPLICATION_LEDGER).read_records()) == 1

    later_run = _consume(root, cutoff="20260712")
    assert later_run.metadata["counterfactual"] is True
    assert len(HashChainLedger(root, root / APPLICATION_LEDGER).read_records()) == 2


def test_non_cn_runtime_is_off_without_ledger_mutation(tmp_path: Path) -> None:
    assert "market" in inspect.signature(consume_overlay).parameters
    root = tmp_path / "private"
    _validated_artifacts(root)

    decision = consume_overlay(
        market="US",
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260711",
        run_key="US:full_market:20260711",
        current_data_generation=GENERATION,
        env=_env(root),
    )

    assert decision.effective_mode == "off"
    assert decision.metadata["blockers"] == ["market_not_supported"]
    assert not (root / APPLICATION_LEDGER).exists()


def test_validated_overlay_bytes_are_hash_bound_and_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    overlay_path = next(root.glob("CN/*/*/overlays/*.overlay.v1.json"))
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload["computed_delta"] = 0.07
    payload["adjusted_score"] = 0.27
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    overlay_path.chmod(0o600)

    decision = _consume(root, cutoff="20260711")

    assert decision.applied is False
    assert decision.metadata["blockers"] == ["validated_overlay_sha256_mismatch"]
    assert not (root / APPLICATION_LEDGER).exists()


def test_concurrent_consumers_append_one_deterministic_application_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(lambda _: _consume(root, cutoff="20260711"), range(2)))
    assert all(bool(item.metadata["counterfactual"]) for item in decisions)
    assert sum(bool(item.metadata.get("idempotent_replay")) for item in decisions) == 1
    records = HashChainLedger(root, root / APPLICATION_LEDGER).read_records()
    assert len(records) == 1
    assert str(records[0]["event"]["event_id"]).startswith("app:")


def test_limited_fails_closed_without_hash_bound_maxwell_activation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260711",
        run_key="CN:full_a:20260711",
        current_data_generation=GENERATION,
        env=_env(root, "limited"),
    )
    assert decision.effective_mode == "off"
    assert decision.applied is False
    assert "activation_path_missing" in decision.metadata["blockers"]
    assert not (root / APPLICATION_LEDGER).exists()


def test_limited_activation_caps_delta_and_suppresses_generic_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    manifest_sha, ledger_sha = _write_gate_holdings_snapshot(root, tmp_path)
    monkeypatch.setattr(
        "quant_investor.fundamental_research.runtime.verify_recomputed_evidence",
        lambda **_kwargs: [],
    )
    evidence = _gate_evidence()
    evidence_path = root / "state" / "activation-evidence.v2.json"
    evidence_digest = atomic_write_json_model(root, evidence_path, evidence)
    activation = ActivationManifestV1(
        mode="limited",
        phase="limited_phase_1",
        effective_from=datetime(2026, 7, 11, tzinfo=UTC),
        approved_by="maxwell",
        approved_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        approval_id="maxwell-confirmation-1",
        separate_confirmation=True,
        shadow_gates_passed=True,
        gate_evidence_path="state/activation-evidence.v2.json",
        gate_evidence_sha256=evidence_digest,
        holdings_manifest_sha256=manifest_sha,
        holdings_ledger_sha256=ledger_sha,
        shadow_trading_days=10,
        validated_dossiers=30,
        distinct_companies=10,
        distinct_industries=3,
        holdings_coverage_passed=True,
        limited_trading_days=0,
        target_weight_counterfactual_days=10,
        nav_attribution_days=0,
        max_abs_delta=0.03,
    )
    activation_path = root / "state" / "activation.v1.json"
    digest = atomic_write_json_model(root, activation_path, activation)
    monkeypatch.setattr(
        "quant_investor.fundamental_research.runtime.build_activation_gate_evidence",
        lambda **_kwargs: _gate_evidence(),
    )
    env = {
        **_env(root, "limited"),
        "FUNDAMENTAL_RESEARCH_ACTIVATION_PATH": "state/activation.v1.json",
        "FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256": digest,
    }
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260712",
        run_key="CN:full_a:20260712",
        current_data_generation=GENERATION,
        env=env,
        occurred_at=datetime(2026, 7, 12, tzinfo=UTC),
    )
    assert decision.applied is True
    assert decision.adjusted_score == 0.23
    assert decision.suppress_generic_fundamental_overlay is True
    assert decision.metadata["computed_delta"] == 0.03
    assert decision.metadata["activation_sha256"] == digest

    monkeypatch.setattr(
        "quant_investor.fundamental_research.runtime.build_activation_gate_evidence",
        lambda **_kwargs: _gate_evidence().model_copy(update={"holdings_symbols": ["999999.SH"]}),
    )
    regressed = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260713",
        run_key="CN:full_a:20260713",
        current_data_generation=GENERATION,
        env=env,
        occurred_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    assert regressed.applied is False
    assert regressed.effective_mode == "off"
    assert "holdings_coverage_incomplete" in regressed.metadata["activation_blockers"]


def test_gate_evidence_is_recomputed_and_low_validation_rate_falls_back_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private"
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    _validated_artifacts(root)
    holdings_snapshot_path = root / "state" / "holdings-scope.v1.json"
    manifest_sha, ledger_sha = _write_gate_holdings_snapshot(root, tmp_path)
    rebuilt = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=holdings_snapshot_path,
        generated_at=datetime(2026, 7, 10, 11, tzinfo=UTC),
    )
    assert rebuilt.validated_request_ids == ["req-600000"]
    assert rebuilt.holdings_coverage_passed is True
    assert verify_recomputed_evidence(root=root, evidence=rebuilt) == []
    assert verify_recomputed_evidence(
        root=root,
        evidence=rebuilt.model_copy(update={"holdings_symbols": []}),
    ) == ["gate_evidence_recomputation_mismatch"]

    monkeypatch.setattr(
        "quant_investor.fundamental_research.runtime.verify_recomputed_evidence",
        lambda **_kwargs: [],
    )
    _write_gate_holdings_snapshot(root, tmp_path)
    evidence = _gate_evidence().model_copy(
        update={
            "recent_received_request_ids": [f"recent-{index:02d}" for index in range(10)],
            "recent_validation_success_count": 7,
        }
    )
    evidence_path = root / "state" / "activation-evidence-low-rate.v2.json"
    evidence_digest = atomic_write_json_model(root, evidence_path, evidence)
    activation = ActivationManifestV1(
        mode="limited",
        phase="limited_phase_1",
        effective_from=datetime(2026, 7, 11, tzinfo=UTC),
        approved_by="maxwell",
        approved_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        approval_id="maxwell-confirmation-low-rate",
        separate_confirmation=True,
        shadow_gates_passed=True,
        gate_evidence_path="state/activation-evidence-low-rate.v2.json",
        gate_evidence_sha256=evidence_digest,
        holdings_manifest_sha256=manifest_sha,
        holdings_ledger_sha256=ledger_sha,
        shadow_trading_days=10,
        validated_dossiers=30,
        distinct_companies=10,
        distinct_industries=3,
        holdings_coverage_passed=True,
        limited_trading_days=0,
        target_weight_counterfactual_days=10,
        nav_attribution_days=0,
        max_abs_delta=0.03,
    )
    activation_path = root / "state" / "activation-low-rate.v1.json"
    activation_digest = atomic_write_json_model(root, activation_path, activation)
    jobs = HashChainLedger(root, root / "state" / "jobs.v1.jsonl")
    for index in range(10):
        request_id = f"post-approval-{index:02d}"
        jobs.append(
            JobEventV1(
                event_id=f"post-approval-received-{index:02d}",
                request_id=request_id,
                state=JobState.RECEIVED,
                occurred_at=datetime(2026, 7, 11, 1, index, tzinfo=UTC),
            ),
            expected_head=jobs.head(),
        )
        if index >= 7:
            jobs.append(
                JobEventV1(
                    event_id=f"post-approval-rejected-{index:02d}",
                    request_id=request_id,
                    state=JobState.REJECTED,
                    occurred_at=datetime(2026, 7, 11, 2, index, tzinfo=UTC),
                    reason="validation rejected",
                ),
                expected_head=jobs.head(),
            )
    monkeypatch.setattr(
        "quant_investor.fundamental_research.runtime.build_activation_gate_evidence",
        lambda **_kwargs: _gate_evidence().model_copy(
            update={
                "recent_received_request_ids": [
                    f"post-approval-{index:02d}" for index in range(10)
                ],
                "recent_validation_success_count": 7,
            }
        ),
    )
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260712",
        run_key="CN:full_a:20260712-low-rate",
        current_data_generation=GENERATION,
        env={
            **_env(root, "limited"),
            "FUNDAMENTAL_RESEARCH_ACTIVATION_PATH": "state/activation-low-rate.v1.json",
            "FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256": activation_digest,
        },
    )
    assert decision.effective_mode == "shadow"
    assert decision.applied is False
    assert decision.metadata["automatic_shadow_fallback"] is True
    assert decision.metadata["activation_blockers"] == [
        "recent_validation_success_below_80pct",
    ]


def test_longitudinal_observation_is_hash_bound_idempotent_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    occurred_at = datetime(2026, 7, 12, 10, tzinfo=UTC)
    run_key = "CN:full_a:20260711"
    application = ApplicationEventV1(
        event_id="application-target-weight-20260711",
        request_id="req-600000",
        dossier_id="dossier-600000",
        run_key=run_key,
        run_cutoff=datetime(2026, 7, 11, 7, tzinfo=UTC),
        state=ApplicationState.SHADOW_EVALUATED,
        occurred_at=datetime(2026, 7, 11, 8, tzinfo=UTC),
        mode="shadow",
        base_score=0.2,
        computed_delta=0.08,
        adjusted_score=0.28,
    )
    applications = HashChainLedger(root, root / APPLICATION_LEDGER)
    applications.append(application, expected_head="")
    source_paths = {
        "actual": root / "outcomes" / "actual-source.v1.json",
        "counterfactual": root / "outcomes" / "counterfactual-source.v1.json",
    }
    canonical_hashes = {}
    canonical_paths = {}
    for variant, value in (("actual", 0.12), ("counterfactual", 0.09)):
        canonical_paths[variant] = root / "canonical" / variant / "analysis_run_manifest.v1.json"
        canonical_payload = {
            "schema_version": "analysis-run-manifest.v1",
            "run_id": f"analysis-{variant}-20260711",
            "generated_at": occurred_at.isoformat(),
            "market": "CN",
            "git_sha": "a" * 40,
            "analysis_meta": {
                "fundamental_research_variant": (
                    "without_dossier" if variant == "actual" else "with_dossier"
                ),
                "portfolio_decision": {"target_weights": {"600000.SH": value}},
                "data_snapshot": {
                    "snapshot_id": "portfolio-generation-1",
                    "local_latest_trade_date": "20260711",
                },
                "global_context": {"universe_key": "full_a"},
            },
        }
        if variant == "counterfactual":
            replay = _control_chain_replay(variant="with_dossier", value=value)
            canonical_payload["analysis_meta"].update(
                {
                    "fundamental_research_source_run_id": "analysis-actual-20260711",
                    "fundamental_research_source_manifest_sha256": canonical_hashes["actual"],
                    "portfolio_decision": replay["portfolio_decision"],
                    "fundamental_research_control_chain": replay,
                }
            )
        canonical_payload["analysis_meta_sha256"] = sha256_bytes(
            canonical_json_bytes(canonical_payload["analysis_meta"])
        )
        canonical_payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(canonical_payload))
        canonical_paths[variant].parent.mkdir(parents=True, exist_ok=True)
        canonical_paths[variant].write_bytes(canonical_json_bytes(canonical_payload))
        canonical_paths[variant].chmod(0o600)
        canonical_hashes[variant] = sha256_bytes(canonical_paths[variant].read_bytes())
    source_hashes = {}
    for variant, value in (("actual", 0.12), ("counterfactual", 0.09)):
        source_hashes[variant] = atomic_write_json_model(
            root,
            source_paths[variant],
            LongitudinalSourceArtifactV1(
                source_kind="analysis_run_manifest",
                variant=variant,
                run_key=run_key,
                trading_date=datetime(2026, 7, 11, tzinfo=UTC).date(),
                generation="portfolio-generation-1",
                produced_at=occurred_at,
                value=value,
                symbol="600000.SH",
                dossier_variant=("without_dossier" if variant == "actual" else "with_dossier"),
                canonical_artifact_path=(
                    f"private/canonical/{variant}/analysis_run_manifest.v1.json"
                ),
                canonical_artifact_sha256=canonical_hashes[variant],
            ),
        )
    actual_path = root / "outcomes" / "actual.v1.json"
    counterfactual_path = root / "outcomes" / "counterfactual.v1.json"
    actual = LongitudinalOutcomeArtifactV1(
        observation_type="target_weight",
        variant="actual",
        run_key=run_key,
        trading_date=datetime(2026, 7, 11, tzinfo=UTC).date(),
        value=0.12,
        produced_at=occurred_at,
        source_generation="portfolio-generation-1",
        source_kind="analysis_run_manifest",
        source_artifact_path="outcomes/actual-source.v1.json",
        source_artifact_sha256=source_hashes["actual"],
    )
    counterfactual = actual.model_copy(
        update={
            "variant": "counterfactual",
            "value": 0.09,
            "source_artifact_path": "outcomes/counterfactual-source.v1.json",
            "source_artifact_sha256": source_hashes["counterfactual"],
        }
    )
    actual_sha = atomic_write_json_model(root, actual_path, actual)
    counterfactual_sha = atomic_write_json_model(root, counterfactual_path, counterfactual)
    observation = LongitudinalObservationV1(
        event_id="longitudinal-target-weight-20260711",
        request_id="req-600000",
        dossier_id="dossier-600000",
        observation_type="target_weight",
        run_key="CN:full_a:20260711",
        trading_date=actual.trading_date,
        application_trading_date=actual.trading_date,
        occurred_at=occurred_at,
        actual_artifact_path="outcomes/actual.v1.json",
        counterfactual_artifact_path="outcomes/counterfactual.v1.json",
        actual_artifact_sha256=actual_sha,
        counterfactual_artifact_sha256=counterfactual_sha,
        actual_value=0.12,
        counterfactual_value=0.09,
    )
    observation_path = root / "outcomes" / "observation.v1.json"
    atomic_write_json_model(root, observation_path, observation)

    first = append_longitudinal_observation(root=root, observation_path=observation_path)
    replay = append_longitudinal_observation(root=root, observation_path=observation_path)

    assert first["appended"] is True
    assert replay["idempotent_replay"] is True

    bypass_application = application.model_copy(
        update={
            "event_id": "application-target-weight-bypass",
            "request_id": "req-bypass",
            "dossier_id": "dossier-bypass",
        }
    )
    applications.append(
        bypass_application,
        expected_head=applications.read_records()[-1]["event_sha256"],
    )
    unlinked_path = root / "canonical" / "unlinked" / "analysis_run_manifest.v1.json"
    unlinked_payload = json.loads(canonical_paths["counterfactual"].read_text(encoding="utf-8"))
    unlinked_payload["analysis_meta"]["fundamental_research_source_manifest_sha256"] = "b" * 64
    unlinked_payload["analysis_meta_sha256"] = sha256_bytes(
        canonical_json_bytes(unlinked_payload["analysis_meta"])
    )
    unlinked_payload.pop("manifest_sha256")
    unlinked_payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unlinked_payload))
    unlinked_path.parent.mkdir(parents=True, exist_ok=True)
    unlinked_path.write_bytes(canonical_json_bytes(unlinked_payload))
    unlinked_path.chmod(0o600)
    unlinked_source = LongitudinalSourceArtifactV1.model_validate(
        {
            **json.loads(source_paths["counterfactual"].read_text(encoding="utf-8")),
            "canonical_artifact_path": ("private/canonical/unlinked/analysis_run_manifest.v1.json"),
            "canonical_artifact_sha256": sha256_bytes(unlinked_path.read_bytes()),
        }
    )
    unlinked_source_path = root / "outcomes" / "unlinked-source.v1.json"
    unlinked_source_sha = atomic_write_json_model(root, unlinked_source_path, unlinked_source)
    unlinked_outcome = counterfactual.model_copy(
        update={
            "source_artifact_path": "outcomes/unlinked-source.v1.json",
            "source_artifact_sha256": unlinked_source_sha,
        }
    )
    unlinked_outcome_path = root / "outcomes" / "unlinked.v1.json"
    unlinked_outcome_sha = atomic_write_json_model(root, unlinked_outcome_path, unlinked_outcome)
    bypass_observation = observation.model_copy(
        update={
            "event_id": "longitudinal-target-weight-bypass",
            "request_id": "req-bypass",
            "dossier_id": "dossier-bypass",
            "counterfactual_artifact_path": "outcomes/unlinked.v1.json",
            "counterfactual_artifact_sha256": unlinked_outcome_sha,
        }
    )
    bypass_path = root / "outcomes" / "observation-bypass.v1.json"
    atomic_write_json_model(root, bypass_path, bypass_observation)
    with pytest.raises(ValueError, match="companion analysis source mismatch"):
        append_longitudinal_observation(root=root, observation_path=bypass_path)

    no_application = observation.model_copy(
        update={
            "event_id": "longitudinal-without-application",
            "request_id": "req-without-application",
        }
    )
    no_application_path = root / "outcomes" / "observation-no-application.v1.json"
    atomic_write_json_model(root, no_application_path, no_application)
    with pytest.raises(ValueError, match="no matching application"):
        append_longitudinal_observation(root=root, observation_path=no_application_path)

    generation_two_canonical_path = (
        root / "canonical" / "counterfactual-2" / "analysis_run_manifest.v1.json"
    )
    generation_two_canonical = json.loads(
        canonical_paths["counterfactual"].read_text(encoding="utf-8")
    )
    generation_two_canonical["analysis_meta"]["data_snapshot"][
        "snapshot_id"
    ] = "portfolio-generation-2"
    generation_two_canonical["analysis_meta_sha256"] = sha256_bytes(
        canonical_json_bytes(generation_two_canonical["analysis_meta"])
    )
    generation_two_canonical.pop("manifest_sha256")
    generation_two_canonical["manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(generation_two_canonical)
    )
    generation_two_canonical_path.parent.mkdir(parents=True, exist_ok=True)
    generation_two_canonical_path.write_bytes(canonical_json_bytes(generation_two_canonical))
    generation_two_canonical_path.chmod(0o600)
    generation_two_source = LongitudinalSourceArtifactV1(
        source_kind="analysis_run_manifest",
        variant="counterfactual",
        run_key=run_key,
        trading_date=actual.trading_date,
        generation="portfolio-generation-2",
        produced_at=occurred_at,
        value=0.09,
        symbol="600000.SH",
        dossier_variant="with_dossier",
        canonical_artifact_path=(
            "private/canonical/counterfactual-2/analysis_run_manifest.v1.json"
        ),
        canonical_artifact_sha256=sha256_bytes(generation_two_canonical_path.read_bytes()),
    )
    generation_two_source_path = root / "outcomes" / "counterfactual-source-2.v1.json"
    generation_two_source_sha = atomic_write_json_model(
        root, generation_two_source_path, generation_two_source
    )
    generation_two_outcome = counterfactual.model_copy(
        update={
            "source_generation": "portfolio-generation-2",
            "source_artifact_path": "outcomes/counterfactual-source-2.v1.json",
            "source_artifact_sha256": generation_two_source_sha,
        }
    )
    generation_two_outcome_path = root / "outcomes" / "counterfactual-2.v1.json"
    generation_two_outcome_sha = atomic_write_json_model(
        root, generation_two_outcome_path, generation_two_outcome
    )
    generation_mismatch = observation.model_copy(
        update={
            "event_id": "longitudinal-generation-mismatch",
            "counterfactual_artifact_path": "outcomes/counterfactual-2.v1.json",
            "counterfactual_artifact_sha256": generation_two_outcome_sha,
        }
    )
    generation_mismatch_path = root / "outcomes" / "observation-generation-mismatch.v1.json"
    atomic_write_json_model(root, generation_mismatch_path, generation_mismatch)
    with pytest.raises(ValueError, match="source generation mismatch"):
        append_longitudinal_observation(root=root, observation_path=generation_mismatch_path)

    holdings_snapshot_path = root / "state" / "holdings-scope.v1.json"
    _write_gate_holdings_snapshot(root, tmp_path)
    evidence = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=holdings_snapshot_path,
        generated_at=occurred_at,
    )
    assert evidence.target_weight_counterfactual_dates == [actual.trading_date]

    duplicate = observation.model_copy(
        update={"event_id": "longitudinal-target-weight-20260711-duplicate"}
    )
    duplicate_path = root / "outcomes" / "observation-duplicate.v1.json"
    atomic_write_json_model(root, duplicate_path, duplicate)
    with pytest.raises(ValueError, match="logical observation already exists"):
        append_longitudinal_observation(root=root, observation_path=duplicate_path)

    source_payload = json.loads(canonical_paths["actual"].read_text(encoding="utf-8"))
    source_payload["analysis_meta"]["portfolio_decision"]["target_weights"]["600000.SH"] = 0.11
    canonical_paths["actual"].write_bytes(canonical_json_bytes(source_payload))
    canonical_paths["actual"].chmod(0o600)
    drifted = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=holdings_snapshot_path,
        generated_at=occurred_at,
    )
    assert drifted.target_weight_counterfactual_dates == []
    assert any(
        item.startswith("longitudinal_artifact_invalid:") for item in drifted.critical_error_codes
    )


def test_real_control_chain_and_strict_parquet_producers_feed_longitudinal_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private"
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    _validated_artifacts(root)
    request_path = next(root.glob("CN/*/*/requests/*.request.v1.json"))
    application = ApplicationEventV1(
        event_id="limited-application-20260711",
        request_id="req-600000",
        dossier_id="dossier-600000",
        run_key="CN:full_a:20260711",
        run_cutoff=datetime(2026, 7, 11, 7, tzinfo=UTC),
        state=ApplicationState.LIMITED_APPLIED,
        occurred_at=datetime(2026, 7, 11, 8, tzinfo=UTC),
        mode="limited",
        base_score=0.2,
        computed_delta=0.03,
        adjusted_score=0.23,
    )
    applications = HashChainLedger(root, root / APPLICATION_LEDGER)
    applications.append(application, expected_head="")
    reports = tmp_path / "reports"
    reports.mkdir()
    trade_report = reports / "trade.md"
    trade_report.write_text("report", encoding="utf-8")
    replay = _control_chain_replay(variant="without_dossier", value=0.09)
    counterfactual_portfolio = replay["portfolio_decision"]
    analysis_meta = {
        "portfolio_decision": {
            "target_weights": {"600000.SH": 0.12},
            "metadata": {"fundamental_research_counterfactual_replay": replay},
        },
        "data_snapshot": {
            "snapshot_id": "snapshot-20260711",
            "local_latest_trade_date": "20260711",
        },
        "global_context": {"universe_key": "full_a"},
    }
    actual_manifest = Path(
        write_analysis_run_manifest(
            market="CN",
            analysis_output_dir=tmp_path,
            report_paths={"trade_report": str(trade_report)},
            analysis_meta=analysis_meta,
        )
    )
    counter_manifest = actual_manifest.parent / "analysis_run_manifest.without_dossier.v1.json"
    produced_at = datetime.now(UTC) + timedelta(seconds=1)

    target = produce_target_weight_observation(
        root=root,
        request_path=request_path,
        dossier_id="dossier-600000",
        actual_analysis_manifest=actual_manifest,
        counterfactual_analysis_manifest=counter_manifest,
        now=produced_at,
    )

    assert target["actual_value"] == 0.12
    assert target["counterfactual_value"] == 0.09

    def _snapshot(self):
        return {"healthy": True, "snapshot_id": "snapshot-20260712"}

    def _read_symbol_frame(self, symbol, **_kwargs):
        return SimpleNamespace(
            frame=pd.DataFrame(
                [
                    {"trade_date": "20260711", "close": 10.0},
                    {"trade_date": "20260712", "close": 11.0},
                ]
            ),
            issues=[],
        )

    monkeypatch.setattr(
        "quant_investor.market.market_data_reader.MarketDataReader.snapshot",
        _snapshot,
    )
    monkeypatch.setattr(
        "quant_investor.market.market_data_reader.MarketDataReader.read_symbol_frame",
        _read_symbol_frame,
    )
    nav = produce_nav_attribution_observation(
        root=root,
        target_weight_observation=root / target["observation_path"],
        attribution_date="2026-07-12",
        data_root=tmp_path / "data",
        now=produced_at + timedelta(seconds=1),
    )

    assert nav["actual_value"] == pytest.approx(0.012)
    assert nav["counterfactual_value"] == pytest.approx(0.009)
    longitudinal_ledger = root / "state" / "longitudinal.v1.jsonl"
    longitudinal_ledger_bytes = longitudinal_ledger.read_bytes()
    longitudinal_ledger.unlink()
    try:
        with pytest.raises(ValueError, match="parent target-weight observation"):
            produce_nav_attribution_observation(
                root=root,
                target_weight_observation=root / target["observation_path"],
                attribution_date="2026-07-12",
                data_root=tmp_path / "data",
                now=produced_at + timedelta(seconds=1),
            )
    finally:
        longitudinal_ledger.write_bytes(longitudinal_ledger_bytes)
        longitudinal_ledger.chmod(0o600)
    _write_gate_holdings_snapshot(root, tmp_path)
    evidence = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=root / "state" / "holdings-scope.v1.json",
        generated_at=produced_at + timedelta(seconds=2),
    )
    assert evidence.target_weight_counterfactual_dates == [date(2026, 7, 11)]
    assert evidence.nav_attribution_dates == [date(2026, 7, 12)]

    target_path = root / target["observation_path"]
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    target_payload["actual_value"] = 0.11
    target_path.write_bytes(canonical_json_bytes(target_payload))
    target_path.chmod(0o600)
    drifted = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=root / "state" / "holdings-scope.v1.json",
        generated_at=produced_at + timedelta(seconds=3),
    )
    assert drifted.target_weight_counterfactual_dates == [date(2026, 7, 11)]
    assert drifted.nav_attribution_dates == []
    assert any(
        item.startswith("longitudinal_artifact_invalid:") for item in drifted.critical_error_codes
    )


def test_runtime_base_score_mismatch_is_zero_effect(tmp_path: Path) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.21,
        run_cutoff="20260711",
        run_key="CN:full_a:20260711",
        current_data_generation=GENERATION,
        env=_env(root),
    )
    assert decision.applied is False
    assert decision.metadata["blockers"] == ["runtime_base_score_mismatch"]
    assert not (root / APPLICATION_LEDGER).exists()


def test_generation_change_invalidates_prior_overlay(tmp_path: Path) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260711",
        run_key="CN:full_a:20260711",
        current_data_generation="fundamental-generation-2",
        env=_env(root),
    )
    assert decision.applied is False
    assert decision.metadata["blockers"] == ["fundamental_data_generation_mismatch"]
    assert not (root / APPLICATION_LEDGER).exists()


def test_activation_rejects_self_asserted_gate_pass_without_counts() -> None:
    with pytest.raises(ValidationError, match="evidence gates"):
        ActivationManifestV1(
            mode="limited",
            phase="limited_phase_1",
            effective_from=datetime(2026, 7, 11, tzinfo=UTC),
            approved_by="maxwell",
            approved_at=datetime(2026, 7, 10, tzinfo=UTC),
            approval_id="maxwell-confirmation-low-evidence",
            separate_confirmation=True,
            shadow_gates_passed=True,
            gate_evidence_path="state/missing-evidence.v1.json",
            gate_evidence_sha256="e" * 64,
            holdings_manifest_sha256="e" * 64,
            holdings_ledger_sha256="f" * 64,
            shadow_trading_days=9,
            validated_dossiers=29,
            distinct_companies=9,
            distinct_industries=2,
            holdings_coverage_passed=True,
            limited_trading_days=0,
            target_weight_counterfactual_days=0,
            nav_attribution_days=0,
            max_abs_delta=0.03,
        )


def test_limited_phase_one_rejects_point_zero_five_cap_at_day_zero() -> None:
    with pytest.raises(ValidationError, match="phase 1"):
        ActivationManifestV1(
            mode="limited",
            phase="limited_phase_1",
            effective_from=datetime(2026, 7, 11, tzinfo=UTC),
            approved_by="maxwell",
            approved_at=datetime(2026, 7, 10, tzinfo=UTC),
            approval_id="maxwell-confirmation-phase-one-overcap",
            separate_confirmation=True,
            shadow_gates_passed=True,
            gate_evidence_path="state/gate-evidence.v1.json",
            gate_evidence_sha256="e" * 64,
            holdings_manifest_sha256="e" * 64,
            holdings_ledger_sha256="f" * 64,
            shadow_trading_days=10,
            validated_dossiers=30,
            distinct_companies=10,
            distinct_industries=3,
            holdings_coverage_passed=True,
            limited_trading_days=0,
            target_weight_counterfactual_days=10,
            nav_attribution_days=0,
            max_abs_delta=0.05,
        )


def test_job_validation_is_point_in_time_at_cn_1500_cutoff(tmp_path: Path) -> None:
    accepted_root = tmp_path / "accepted"
    _validated_artifacts(
        accepted_root,
        validated_at=datetime(2026, 7, 11, 6, 59, tzinfo=UTC),
    )
    accepted = _consume(accepted_root, cutoff="20260711")
    assert accepted.metadata["counterfactual"] is True

    rejected_root = tmp_path / "rejected"
    _validated_artifacts(
        rejected_root,
        validated_at=datetime(2026, 7, 11, 7, 1, tzinfo=UTC),
    )
    rejected = _consume(rejected_root, cutoff="20260711")
    assert rejected.metadata["blockers"] == ["job_not_validated_at_run_cutoff"]
    assert not (rejected_root / APPLICATION_LEDGER).exists()


def test_activation_rejects_wrong_gate_evidence_hash(tmp_path: Path) -> None:
    root = tmp_path / "private"
    _validated_artifacts(root)
    evidence = _gate_evidence()
    atomic_write_json_model(root, root / "state" / "activation-evidence.v2.json", evidence)
    activation = ActivationManifestV1(
        mode="limited",
        phase="limited_phase_1",
        effective_from=datetime(2026, 7, 11, tzinfo=UTC),
        approved_by="maxwell",
        approved_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        approval_id="maxwell-confirmation-bad-evidence-hash",
        separate_confirmation=True,
        shadow_gates_passed=True,
        gate_evidence_path="state/activation-evidence.v2.json",
        gate_evidence_sha256="f" * 64,
        holdings_manifest_sha256="e" * 64,
        holdings_ledger_sha256="f" * 64,
        shadow_trading_days=10,
        validated_dossiers=30,
        distinct_companies=10,
        distinct_industries=3,
        holdings_coverage_passed=True,
        limited_trading_days=0,
        target_weight_counterfactual_days=10,
        nav_attribution_days=0,
        max_abs_delta=0.03,
    )
    activation_path = root / "state" / "activation.v1.json"
    activation_digest = atomic_write_json_model(root, activation_path, activation)
    env = {
        **_env(root, "limited"),
        "FUNDAMENTAL_RESEARCH_ACTIVATION_PATH": "state/activation.v1.json",
        "FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256": activation_digest,
    }
    decision = consume_overlay(
        symbol="600000.SH",
        base_score=0.2,
        run_cutoff="20260712",
        run_key="CN:full_a:20260712",
        current_data_generation=GENERATION,
        env=env,
    )
    assert decision.effective_mode == "off"
    assert decision.applied is False
    assert decision.metadata["blockers"] == ["gate_evidence_invalid:ValueError"]
