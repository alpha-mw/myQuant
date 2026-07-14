from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_investor.fundamental_research import (
    ApplicationEventV1,
    ApplicationState,
    ClaimKind,
    ClaimV1,
    Dimension,
    DimensionAssessmentV1,
    DimensionSignal,
    FundamentalResearchDossierV1,
    FundamentalResearchRequestV1,
    FundamentalResearchResponseV1,
    HashChainLedger,
    JobEventV1,
    JobState,
    LedgerConflictError,
    SourceRecordV1,
    SourceEligibilityPolicyV1,
    SourceTier,
    atomic_write_json_model,
    build_overlay,
    compute_base_score_sha256,
    compute_source_policy_sha256,
    import_response_files,
    load_json_model,
    model_sha256,
    validate_response,
    validate_job_transition,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 14, 8, tzinfo=UTC)
SHA = compute_base_score_sha256(0.2)
PRIMARY_POLICY = SourceEligibilityPolicyV1(primary_hostnames={"example.com", "example.org"})


def request(**changes: object) -> FundamentalResearchRequestV1:
    payload = {
        "request_id": "req-1",
        "run_id": "run-1",
        "symbol": "000001.SZ",
        "company_name": "平安银行",
        "decision_cutoff": NOW,
        "created_at": NOW,
        "expires_at": NOW + timedelta(days=30),
        "base_score": 0.2,
        "base_score_sha256": SHA,
        "git_sha": "abcdef1",
        "data_generation": "gen-1",
        "selection_reasons": ["top-k"],
        "prompt_version": "prompt-v1",
        "policy_version": "policy-v1",
        "source_policy_sha256": compute_source_policy_sha256(PRIMARY_POLICY),
    }
    payload.update(changes)
    if "base_score" in changes and "base_score_sha256" not in changes:
        payload["base_score_sha256"] = compute_base_score_sha256(float(payload["base_score"]))
    return FundamentalResearchRequestV1.model_validate(payload)


def source(
    source_id: str, publisher: str, *, tier: SourceTier = SourceTier.PRIMARY, future: bool = False
) -> SourceRecordV1:
    published = NOW + timedelta(hours=1) if future else NOW - timedelta(days=2)
    return SourceRecordV1(
        source_id=source_id,
        publisher=publisher,
        document_kind="annual_report",
        canonical_url=(
            f"https://disclosure.example.org/{source_id}"
            if source_id.endswith("b")
            else f"https://disclosure.example.com/{source_id}"
        ),
        published_at=published,
        first_available_at=published,
        retrieved_at=NOW + timedelta(days=1),
        source_tier=tier,
        content_sha256=(source_id[-1] if source_id[-1] in "0123456789abcdef" else "b") * 64,
        locator="p.1",
        evidence_extract="verified extract",
    )


def dossier(
    *,
    signals: dict[Dimension, DimensionSignal] | None = None,
    sources: list[SourceRecordV1] | None = None,
    counter_dimension: Dimension | None = None,
) -> FundamentalResearchDossierV1:
    sources = sources or [source("src-a", "Exchange"), source("src-b", "Company")]
    signals = signals or {item: DimensionSignal.POSITIVE for item in Dimension}
    claims = []
    dimensions = []
    for index, dimension in enumerate(Dimension):
        claim_id = f"claim-{index}"
        signal = signals[dimension]
        direction = (
            "negative"
            if signal
            in {
                DimensionSignal.NEGATIVE,
                DimensionSignal.STRONG_NEGATIVE,
            }
            else "positive"
        )
        claims.append(
            ClaimV1(
                claim_id=claim_id,
                kind=ClaimKind.FACT,
                dimension=dimension,
                statement=f"Evidence for {dimension.value}",
                direction=direction,
                materiality=0.5,
                supporting_source_ids=[sources[index % len(sources)].source_id],
                counter_source_ids=(
                    [sources[(index + 1) % len(sources)].source_id]
                    if dimension == counter_dimension
                    else []
                ),
                confidence_rationale="primary disclosure",
            )
        )
        dimensions.append(
            DimensionAssessmentV1(
                dimension=dimension,
                signal=signal,
                claim_ids=[] if signal == DimensionSignal.UNKNOWN else [claim_id],
            )
        )
    return FundamentalResearchDossierV1(
        dossier_id="dossier-1",
        request_id="req-1",
        symbol="000001.SZ",
        company_name="平安银行",
        decision_cutoff=NOW,
        produced_at=NOW + timedelta(days=1),
        model_name="codex",
        prompt_version="prompt-v1",
        sources=sources,
        claims=claims,
        dimensions=dimensions,
    )


def test_strict_contract_rejects_extra_delta_and_invalid_values() -> None:
    req = request()
    response = {
        "request_id": req.request_id,
        "request_sha256": model_sha256(req),
        "dossier": dossier().model_dump(mode="json"),
        "score_delta": 0.1,
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FundamentalResearchResponseV1.model_validate(response)
    with pytest.raises(ValidationError):
        request(base_score=float("nan"), base_score_sha256=SHA)
    with pytest.raises(ValidationError, match="TTL"):
        request(expires_at=NOW + timedelta(days=31))
    with pytest.raises(ValidationError, match="decision_cutoff"):
        request(created_at=NOW - timedelta(seconds=1))
    with pytest.raises(ValidationError, match="base_score_sha256"):
        request(base_score_sha256="b" * 64)


def test_dossier_requires_complete_dimensions_and_valid_graph() -> None:
    payload = dossier().model_dump(mode="python")
    payload["dimensions"] = payload["dimensions"][:-1]
    with pytest.raises(ValidationError):
        FundamentalResearchDossierV1.model_validate(payload)
    payload = dossier().model_dump(mode="python")
    payload["claims"][0]["supporting_source_ids"] = ["missing-source"]
    with pytest.raises(ValidationError, match="unknown sources"):
        FundamentalResearchDossierV1.model_validate(payload)


def test_positive_overlay_uses_local_formula_and_cap() -> None:
    overlay = build_overlay(
        request(base_score=0.95),
        dossier(),
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    assert overlay.eligible is True
    assert overlay.computed_delta == pytest.approx(0.05)
    assert overlay.adjusted_score == 1.0
    assert all(item.qualified for item in overlay.contributions)


def test_positive_gate_and_primary_source_rules_fail_closed() -> None:
    signals = {item: DimensionSignal.UNKNOWN for item in Dimension}
    signals[Dimension.BUSINESS_ECONOMICS] = DimensionSignal.STRONG_POSITIVE
    signals[Dimension.INDUSTRY_VALUE_CHAIN] = DimensionSignal.STRONG_POSITIVE
    overlay = build_overlay(
        request(),
        dossier(signals=signals),
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    assert overlay.eligible is False
    assert overlay.computed_delta == 0.0
    assert "positive_gate_requires_four_dimensions" in overlay.blockers

    secondary = [
        source("src-a", "Media A", tier=SourceTier.SECONDARY),
        source("src-b", "Media B", tier=SourceTier.SECONDARY),
    ]
    secondary_policy = SourceEligibilityPolicyV1(secondary_hostnames={"example.com"})
    overlay = build_overlay(
        request(),
        dossier(sources=secondary),
        imported_at=NOW + timedelta(days=1),
        source_policy=secondary_policy,
    )
    assert overlay.computed_delta == 0.0
    assert "fewer_than_two_independent_publishers" in overlay.blockers
    assert all(not item.qualified for item in overlay.contributions)

    secondary_request = request(source_policy_sha256=compute_source_policy_sha256(secondary_policy))
    response = FundamentalResearchResponseV1(
        request_id="req-1",
        request_sha256=model_sha256(secondary_request),
        dossier=dossier(sources=secondary),
    )
    with pytest.raises(ValueError, match="not eligible for import"):
        validate_response(
            secondary_request,
            response,
            imported_at=NOW + timedelta(days=1),
            source_policy=secondary_policy,
        )


def test_future_primary_and_primary_conflict_do_not_score() -> None:
    future_sources = [
        source("src-a", "Exchange", future=True),
        source("src-b", "Company", future=True),
    ]
    overlay = build_overlay(
        request(),
        dossier(sources=future_sources),
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    assert overlay.computed_delta == 0.0
    assert "fewer_than_two_independent_publishers" in overlay.blockers

    overlay = build_overlay(
        request(),
        dossier(counter_dimension=Dimension.FINANCIAL_QUALITY),
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    financial = next(
        item for item in overlay.contributions if item.dimension == Dimension.FINANCIAL_QUALITY
    )
    assert financial.qualified is False
    assert "unresolved_primary_conflict" in financial.blockers
    assert overlay.computed_delta == 0.0


def test_mixed_future_claim_and_claim_validity_cannot_drive_signal() -> None:
    historical = [source("src-a", "Exchange"), source("src-b", "Company")]
    future = source("src-c", "Future filing", future=True)
    mixed = dossier(sources=[*historical, future])
    financial = next(
        item for item in mixed.dimensions if item.dimension == Dimension.FINANCIAL_QUALITY
    )
    historical_claim = next(item for item in mixed.claims if item.claim_id in financial.claim_ids)
    future_claim = ClaimV1(
        claim_id="claim-future",
        kind=ClaimKind.FACT,
        dimension=Dimension.FINANCIAL_QUALITY,
        statement="Future evidence must not determine the historical signal",
        direction="positive",
        materiality=1.0,
        supporting_source_ids=[future.source_id],
    )
    claims = [
        (
            item.model_copy(update={"direction": "neutral"})
            if item.claim_id == historical_claim.claim_id
            else item
        )
        for item in mixed.claims
    ] + [future_claim]
    dimensions = [
        (
            item.model_copy(
                update={"claim_ids": [historical_claim.claim_id, future_claim.claim_id]}
            )
            if item.dimension == Dimension.FINANCIAL_QUALITY
            else item
        )
        for item in mixed.dimensions
    ]
    mixed = mixed.model_copy(update={"claims": claims, "dimensions": dimensions})
    overlay = build_overlay(
        request(),
        mixed,
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    contribution = next(
        item for item in overlay.contributions if item.dimension == Dimension.FINANCIAL_QUALITY
    )
    assert contribution.qualified is False
    assert "claim_not_pit_eligible:claim-future" in contribution.blockers
    assert "signal_claim_direction_mismatch" in contribution.blockers

    unknown_claim = ClaimV1(
        claim_id="claim-unknown-direction",
        kind=ClaimKind.UNKNOWN,
        dimension=Dimension.FINANCIAL_QUALITY,
        statement="Unknown claim cannot provide a directional scoring signal",
        direction="neutral",
        materiality=0.0,
    )
    unknown_mixed = mixed.model_copy(
        update={
            "claims": [*claims[:-1], unknown_claim],
            "dimensions": [
                (
                    item.model_copy(
                        update={"claim_ids": [historical_claim.claim_id, unknown_claim.claim_id]}
                    )
                    if item.dimension == Dimension.FINANCIAL_QUALITY
                    else item
                )
                for item in mixed.dimensions
            ],
        }
    )
    unknown_overlay = build_overlay(
        request(),
        unknown_mixed,
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    unknown_contribution = next(
        item
        for item in unknown_overlay.contributions
        if item.dimension == Dimension.FINANCIAL_QUALITY
    )
    assert "unknown_scoring_claim:claim-unknown-direction" in unknown_contribution.blockers
    assert "signal_claim_direction_mismatch" in unknown_contribution.blockers

    with pytest.raises(ValidationError, match="directionally neutral"):
        ClaimV1(
            claim_id="claim-invalid-unknown",
            kind=ClaimKind.UNKNOWN,
            dimension=Dimension.FINANCIAL_QUALITY,
            statement="Invalid directional unknown",
            direction="positive",
            materiality=0.0,
        )

    for field, value in (
        ("valid_from", NOW + timedelta(seconds=1)),
        ("valid_until", NOW - timedelta(seconds=1)),
    ):
        baseline = dossier()
        first_claim = baseline.claims[0].model_copy(update={field: value})
        invalid = baseline.model_copy(update={"claims": [first_claim, *baseline.claims[1:]]})
        result = build_overlay(
            request(),
            invalid,
            imported_at=NOW + timedelta(days=1),
            source_policy=PRIMARY_POLICY,
        )
        first = next(
            item for item in result.contributions if item.dimension == first_claim.dimension
        )
        assert f"claim_not_pit_eligible:{first_claim.claim_id}" in first.blockers


def test_response_cannot_forge_primary_tier_and_future_times_block() -> None:
    forged_sources = []
    for item in (source("src-a", "Unknown A"), source("src-b", "Unknown B")):
        payload = item.model_dump(mode="python")
        payload["canonical_url"] = f"https://untrusted.invalid/{item.source_id}"
        forged_sources.append(SourceRecordV1.model_validate(payload))
    overlay = build_overlay(
        request(), dossier(sources=forged_sources), imported_at=NOW + timedelta(days=1)
    )
    assert overlay.computed_delta == 0.0
    assert any(item.startswith("source_tier_mismatch:") for item in overlay.blockers)
    assert all(not item.qualified for item in overlay.contributions)

    early_import = build_overlay(
        request(),
        dossier(),
        imported_at=NOW + timedelta(hours=12),
        source_policy=PRIMARY_POLICY,
    )
    assert early_import.computed_delta == 0.0
    assert "dossier_produced_after_import" in early_import.blockers
    assert any(item.startswith("source_retrieved_after_import:") for item in early_import.blockers)


def test_negative_overlay_does_not_require_four_dimensions() -> None:
    signals = {item: DimensionSignal.UNKNOWN for item in Dimension}
    signals[Dimension.FINANCIAL_QUALITY] = DimensionSignal.STRONG_NEGATIVE
    overlay = build_overlay(
        request(),
        dossier(signals=signals),
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
    )
    assert overlay.eligible is True
    assert overlay.computed_delta == pytest.approx(-0.025)


def test_binding_and_expiry_fail_closed() -> None:
    req = request(expires_at=NOW + timedelta(hours=1))
    overlay = build_overlay(
        req,
        dossier(),
        imported_at=NOW + timedelta(hours=2),
        source_policy=PRIMARY_POLICY,
    )
    assert overlay.eligible is False
    assert overlay.computed_delta == 0.0
    assert "request_expired" in overlay.blockers


def test_private_atomic_json_round_trip_and_symlink_rejection(tmp_path: Path) -> None:
    root = tmp_path / "private"
    path = root / "requests" / "request.json"
    req = request()
    digest = atomic_write_json_model(root, path, req)
    assert digest == model_sha256(req)
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_json_model(root, path, FundamentalResearchRequestV1) == req

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = root / "linked"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        atomic_write_json_model(root, symlink / "escape.json", req)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        atomic_write_json_model(linked_root, linked_root / "request.json", req)
    with pytest.raises(ValueError, match="escapes"):
        atomic_write_json_model(root, tmp_path / "escape.json", req)


def test_json_loader_rejects_nan_depth_and_oversize(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "bad.json"
    path.write_text('{"base_score": NaN}', encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="non-finite"):
        load_json_model(root, path, FundamentalResearchRequestV1)
    value: object = "leaf"
    for _ in range(14):
        value = [value]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="nesting"):
        load_json_model(root, path, FundamentalResearchRequestV1)
    with path.open("wb") as handle:
        handle.truncate(5 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds"):
        load_json_model(root, path, FundamentalResearchRequestV1)


def test_hash_chain_ledger_cas_and_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "private"
    ledger = HashChainLedger(root, root / "state" / "jobs.jsonl")
    first = JobEventV1(
        event_id="event-1", request_id="req-1", state=JobState.PREPARED, occurred_at=NOW
    )
    head = ledger.append(first, expected_head="")
    assert ledger.head() == head
    assert ledger.path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(LedgerConflictError, match="CAS"):
        ledger.append(
            JobEventV1(
                event_id="event-2", request_id="req-1", state=JobState.EXPORTED, occurred_at=NOW
            ),
            expected_head="",
        )
    with pytest.raises(LedgerConflictError, match="duplicate"):
        ledger.append(first, expected_head=head)

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["event"]["reason"] = "tampered"
    ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event hash mismatch"):
        ledger.read_records()


def test_job_state_machine_fails_closed() -> None:
    validate_job_transition(None, JobState.PREPARED)
    validate_job_transition(JobState.PREPARED, JobState.EXPORTED)
    validate_job_transition(JobState.RECEIVED, JobState.VALIDATED)
    with pytest.raises(ValueError, match="invalid job transition"):
        validate_job_transition(JobState.PREPARED, JobState.VALIDATED)


def test_application_event_is_bounded_and_cannot_accept_arbitrary_delta() -> None:
    event = ApplicationEventV1(
        event_id="application-1",
        request_id="req-1",
        dossier_id="dossier-1",
        run_key="CN:20260102",
        run_cutoff=NOW,
        state=ApplicationState.SHADOW_EVALUATED,
        occurred_at=NOW,
        mode="shadow",
        base_score=0.2,
        computed_delta=0.1,
        adjusted_score=0.3,
    )
    assert event.computed_delta == 0.1
    payload = event.model_dump(mode="python")
    payload["computed_delta"] = 0.2
    with pytest.raises(ValidationError):
        ApplicationEventV1.model_validate(payload)


def test_import_service_validates_hash_and_validate_only(tmp_path: Path) -> None:
    root = tmp_path / "private"
    req = request()
    response = FundamentalResearchResponseV1(
        request_id=req.request_id,
        request_sha256=model_sha256(req),
        dossier=dossier(),
    )
    request_path = root / "requests" / "request.json"
    response_path = root / "responses" / "response.json"
    atomic_write_json_model(root, request_path, req)
    atomic_write_json_model(root, response_path, response)
    overlay = import_response_files(
        root=root,
        request_path=request_path,
        response_path=response_path,
        dossier_path=root / "dossiers" / "dossier.json",
        overlay_path=root / "overlays" / "overlay.json",
        imported_at=NOW + timedelta(days=1),
        source_policy=PRIMARY_POLICY,
        validate_only=True,
    )
    assert overlay.eligible is True
    assert not (root / "dossiers" / "dossier.json").exists()

    bad = response.model_copy(update={"request_sha256": "b" * 64})
    with pytest.raises(ValueError, match="request_sha256"):
        validate_response(
            req,
            bad,
            imported_at=NOW + timedelta(days=1),
            source_policy=PRIMARY_POLICY,
        )
