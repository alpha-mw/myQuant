from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping, get_type_hints

import pytest

import quant_investor.intelligence.decision as decision_api
from quant_investor.intelligence._core import ZERO_SHA256
from quant_investor.intelligence.decision import (
    DecisionContractError,
    PaperPortfolioAdapter,
    append_decision_discipline,
    build_context_note,
    build_investment_memo,
    build_paper_intake_proposal,
    collect_investment_decision_context,
    validate_decision_discipline_chain,
    validate_investment_decision_context,
    validate_investment_decision_receipt,
    validate_investment_memo,
    validate_paper_intake_proposal,
    validate_risk_assessment_receipt,
)
from quant_investor.intelligence.evidence import build_ai_draft
from quant_investor.v17_v4_contract.canonical import canonical_bytes
from tests.unit.test_v17_i1_investment_decision import (
    AS_OF,
    REVIEW_DUE_AT,
    _decision_stack,
    _i0_replay_inputs,
    _later,
    _policy,
    _reseal,
)

EXPECTED_PUBLIC_API = {
    "DecisionContractError",
    "PaperPortfolioAdapter",
    "append_decision_discipline",
    "assess_investment_risk",
    "build_context_note",
    "build_decision_policy",
    "build_investment_memo",
    "build_paper_intake_proposal",
    "collect_investment_decision_context",
    "make_investment_decision",
    "validate_context_note",
    "validate_decision_discipline_chain",
    "validate_decision_policy",
    "validate_investment_decision_context",
    "validate_investment_decision_receipt",
    "validate_investment_memo",
    "validate_paper_intake_proposal",
    "validate_risk_assessment_receipt",
}
AUTHORITY_FALSE = {
    "broker",
    "execution",
    "llm",
    "order",
    "portfolio",
    "provider",
    "selector",
    "trade",
}


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _walk_keys(value: Any) -> set[str]:
    if type(value) is dict:
        return set(value) | {key for child in value.values() for key in _walk_keys(child)}
    if type(value) in {list, tuple}:
        return {key for child in value for key in _walk_keys(child)}
    return set()


def _decision_history(stack: Mapping[str, Any]) -> dict[str, Any]:
    return {
        stack["decision"]["decision_receipt_id"]: {
            "decision_receipt": stack["decision"],
            "decision_validation_closure": stack["decision_closure"],
        }
    }


def test_public_api_is_library_only_and_adapter_is_a_protocol() -> None:
    assert set(decision_api.__all__) == EXPECTED_PUBLIC_API
    assert inspect.isclass(PaperPortfolioAdapter)
    assert getattr(PaperPortfolioAdapter, "_is_protocol", False) is True
    method = PaperPortfolioAdapter.submit_for_external_review
    hints = get_type_hints(method)
    assert "proposal" in hints and hints["return"] is type(None)


def test_all_artifacts_keep_every_authority_surface_closed(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    memo = build_investment_memo(
        context=stack["context"],
        context_replay_closure=stack["context_closure"],
        policy=stack["policy"],
        risk_receipt=stack["risk"],
        assessments_by_dimension=stack["assessments"],
        decision_receipt=stack["decision"],
        as_of=AS_OF,
    )
    proposal = build_paper_intake_proposal(
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        proposed_at=_later(AS_OF),
    )
    artifacts = (
        stack["policy"],
        *stack["notes"],
        stack["context"],
        stack["risk"],
        stack["decision"],
        memo,
        proposal,
    )
    for artifact in artifacts:
        assert artifact["research_only"] is True
        assert artifact["production"] is False
        assert artifact["decision_protocol"] == "myquant.v17.v4"
        assert artifact["mainline_authority"] is False
        assert artifact["operational_activation_unchanged"] is True
        assert artifact["broker"] is False
        assert artifact["execution"] is False
        assert artifact["order"] is False
        assert artifact["trade"] is False
        assert all(artifact["authority"][key] is False for key in AUTHORITY_FALSE)


def test_decision_facing_ai_accepts_only_exact_allowlisted_payloads(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    source_ref = next(
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    hypothesis_draft = build_ai_draft(
        kind="HYPOTHESIS_DRAFT",
        payload={"thesis": "Draft only."},
        source_refs=[source_ref],
        generated_at=AS_OF,
        confidence="0.5",
    )
    wrong_summary = build_ai_draft(
        kind="SUMMARY",
        payload={"facts": ["Wrong schema"]},
        source_refs=[source_ref],
        generated_at=AS_OF,
        confidence="0.5",
    )
    for draft in (hypothesis_draft, wrong_summary):
        with pytest.raises(DecisionContractError) as exc_info:
            collect_investment_decision_context(
                i0_replay_inputs=replay,
                policy=_policy(),
                company_code="000001.SZ",
                as_of=AS_OF,
                review_due_at=REVIEW_DUE_AT,
                ai_drafts=(draft,),
            )
        assert exc_info.value.code == "I1_SHAPE_INVALID"


def test_decision_facing_ai_rejects_future_or_unauthorized_sources(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    source_ref = next(
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    future = build_ai_draft(
        kind="SUMMARY",
        payload={"summary": "Future text."},
        source_refs=[source_ref],
        generated_at=_later(AS_OF),
        confidence="0.5",
    )
    foreign_ref = deepcopy(source_ref)
    foreign_ref["artifact_id"] = "foreign-source"
    foreign_ref["relative_path"] = "data/private/v17_v4_sources/foreign-source.json"
    unauthorized = build_ai_draft(
        kind="SUMMARY",
        payload={"summary": "Unbound text."},
        source_refs=[foreign_ref],
        generated_at=AS_OF,
        confidence="0.5",
    )
    for draft, expected in ((future, "I1_FUTURE_INPUT"), (unauthorized, "I1_REF_MISMATCH")):
        with pytest.raises(DecisionContractError) as exc_info:
            collect_investment_decision_context(
                i0_replay_inputs=replay,
                policy=_policy(),
                company_code="000001.SZ",
                as_of=AS_OF,
                review_due_at=REVIEW_DUE_AT,
                ai_drafts=(draft,),
            )
        assert exc_info.value.code == expected


def test_context_note_text_size_duplicate_and_company_bounds(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    source_ref = next(
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    with pytest.raises(DecisionContractError) as exc_info:
        build_context_note(
            kind="WHY_NOW",
            company_code="000001.SZ",
            text="x" * 4001,
            observed_at=AS_OF,
            available_at=AS_OF,
            source_ref=source_ref,
        )
    assert exc_info.value.code == "I1_SHAPE_INVALID"

    note = build_context_note(
        kind="WHY_NOW",
        company_code="000001.SZ",
        text="Bounded.",
        observed_at=AS_OF,
        available_at=AS_OF,
        source_ref=source_ref,
    )
    with pytest.raises(DecisionContractError) as exc_info:
        collect_investment_decision_context(
            i0_replay_inputs=replay,
            policy=_policy(),
            company_code="000001.SZ",
            as_of=AS_OF,
            review_due_at=REVIEW_DUE_AT,
            context_notes=(note, note),
        )
    assert exc_info.value.code == "I1_SHAPE_INVALID"
    with pytest.raises(DecisionContractError) as exc_info:
        collect_investment_decision_context(
            i0_replay_inputs=replay,
            policy=_policy(),
            company_code="600000.SH",
            as_of=AS_OF,
            review_due_at=REVIEW_DUE_AT,
        )
    assert exc_info.value.code == "I1_REF_MISMATCH"


def test_all_validators_full_replay_and_do_not_write(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    memo = build_investment_memo(
        context=stack["context"],
        context_replay_closure=stack["context_closure"],
        policy=stack["policy"],
        risk_receipt=stack["risk"],
        assessments_by_dimension=stack["assessments"],
        decision_receipt=stack["decision"],
        as_of=AS_OF,
    )
    proposal = build_paper_intake_proposal(
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        proposed_at=_later(AS_OF),
    )
    before = _tree_hash(tmp_path)
    assert (
        validate_investment_decision_context(stack["context"], **stack["context_closure"])
        == stack["context"]
    )
    assert (
        validate_risk_assessment_receipt(
            stack["risk"],
            context=stack["context"],
            context_replay_closure=stack["context_closure"],
            policy=stack["policy"],
            assessments_by_dimension=stack["assessments"],
            as_of=AS_OF,
        )
        == stack["risk"]
    )
    assert (
        validate_investment_decision_receipt(stack["decision"], **stack["decision_closure"])
        == stack["decision"]
    )
    assert (
        validate_investment_memo(
            memo,
            context=stack["context"],
            context_replay_closure=stack["context_closure"],
            policy=stack["policy"],
            risk_receipt=stack["risk"],
            assessments_by_dimension=stack["assessments"],
            decision_receipt=stack["decision"],
            as_of=AS_OF,
        )
        == memo
    )
    assert (
        validate_paper_intake_proposal(
            proposal,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            proposed_at=_later(AS_OF),
        )
        == proposal
    )
    assert _tree_hash(tmp_path) == before


def test_decision_builders_never_call_discoverable_external_mutation_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_investor import llm_gateway
    from quant_investor.codex_review import storage as review_storage
    from quant_investor import fetch_cn_index_components
    from quant_investor.factors.registry_store import apply_factor_record_patch
    from quant_investor.factors.store import FactorGovernanceStore
    from quant_investor.v17_mainline.storage import MainlineStore

    stack = _decision_stack(tmp_path)
    called: list[str] = []

    def forbidden(name: str):
        def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
            called.append(name)
            raise AssertionError(f"forbidden external surface called: {name}")

        return fail_if_called

    monkeypatch.setattr(llm_gateway, "complete_json_sync", forbidden("model"))
    monkeypatch.setattr(fetch_cn_index_components, "init_tushare", forbidden("provider"))
    monkeypatch.setattr(review_storage, "write_exact_once", forbidden("persistence"))
    monkeypatch.setattr(MainlineStore, "write_exact_once", forbidden("mainline-write"))
    monkeypatch.setattr(
        FactorGovernanceStore,
        "save_production_library",
        forbidden("portfolio-mutation"),
    )
    monkeypatch.setattr(
        "quant_investor.factors.registry_store.apply_factor_record_patch",
        forbidden("registry-mutation"),
    )
    assert apply_factor_record_patch is not None

    assert (
        validate_investment_decision_context(stack["context"], **stack["context_closure"])
        == stack["context"]
    )
    assert (
        validate_investment_decision_receipt(stack["decision"], **stack["decision_closure"])
        == stack["decision"]
    )
    proposal = build_paper_intake_proposal(
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        proposed_at=_later(AS_OF),
    )
    assert proposal["status"] == "PENDING_EXTERNAL_REVIEW"
    assert called == []


def test_discipline_price_only_review_has_empty_evidence_change_set(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    entries = append_decision_discipline(
        (),
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=None,
        previous_decision_validation_closure=None,
        stage="BEFORE_DECISION",
        event_type="DECISION_CREATED",
        status="ACTIVE",
        summary="Decision created.",
        event_at=_later(AS_OF, seconds=1),
        expected_tip=ZERO_SHA256,
    )
    source_ref = next(
        ref
        for ref in stack["replay"]["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    outcome_ref = {
        **source_ref,
        "artifact_id": "label-a",
        "artifact_version": "myquant.v17.v4.forward-label.v1",
        "relative_path": "results/v17_v4_shadow/forward_labels/label-a.json",
    }
    entries = append_decision_discipline(
        entries,
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=stack["decision"],
        previous_decision_validation_closure=stack["decision_closure"],
        stage="AFTER_OUTCOME",
        event_type="DECISION_REVIEWED",
        status="OUTCOME_AVAILABLE",
        summary="Price observed without evidence change.",
        event_at=_later(AS_OF, seconds=2),
        outcome_refs=(outcome_ref,),
        price_observation_refs=(source_ref,),
        expected_tip=entries[-1]["semantic_sha256"],
    )
    assert entries[-1]["evidence_changes"] == {"added_refs": [], "removed_refs": []}
    assert entries[-1]["price_observation_refs"] == [source_ref]
    assert (
        validate_decision_discipline_chain(entries, decision_history_by_id=_decision_history(stack))
        == entries
    )


def test_discipline_rejects_invalid_root_transition_tip_and_resealed_entry(
    tmp_path: Path,
) -> None:
    stack = _decision_stack(tmp_path)
    kwargs = {
        "decision_receipt": stack["decision"],
        "decision_validation_closure": stack["decision_closure"],
        "previous_decision_receipt": None,
        "previous_decision_validation_closure": None,
        "summary": "Invalid root transition.",
        "event_at": _later(AS_OF),
        "expected_tip": ZERO_SHA256,
    }
    with pytest.raises(DecisionContractError) as exc_info:
        append_decision_discipline(
            (), stage="REVIEW", event_type="LESSON_LEARNED", status="LEARNED", **kwargs
        )
    assert exc_info.value.code == "I1_DISCIPLINE_TRANSITION_INVALID"

    entries = append_decision_discipline(
        (),
        stage="BEFORE_DECISION",
        event_type="DECISION_CREATED",
        status="ACTIVE",
        **kwargs,
    )
    with pytest.raises(DecisionContractError):
        append_decision_discipline(
            entries,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            previous_decision_receipt=stack["decision"],
            previous_decision_validation_closure=stack["decision_closure"],
            stage="AFTER_OUTCOME",
            event_type="DECISION_REVIEWED",
            status="OUTCOME_AVAILABLE",
            summary="Wrong tip.",
            event_at=_later(AS_OF, seconds=2),
            expected_tip="f" * 64,
        )

    with pytest.raises(DecisionContractError) as exc_info:
        append_decision_discipline(
            entries,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            previous_decision_receipt=stack["decision"],
            previous_decision_validation_closure=stack["decision_closure"],
            stage="AFTER_OUTCOME",
            event_type="DECISION_REVIEWED",
            status="OUTCOME_AVAILABLE",
            summary="Missing exact outcome evidence.",
            event_at=_later(AS_OF, seconds=2),
            expected_tip=entries[-1]["semantic_sha256"],
        )
    assert exc_info.value.code == "I1_DISCIPLINE_TRANSITION_INVALID"

    forged = deepcopy(entries[0])
    forged["evidence_changes"] = {"added_refs": [], "removed_refs": []}
    forged = _reseal(forged, identity_field="entry_id")
    with pytest.raises(DecisionContractError):
        validate_decision_discipline_chain(
            (forged,), decision_history_by_id=_decision_history(stack)
        )


def test_decision_discipline_never_mutates_i0_memory_v1(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    original = canonical_bytes(list(stack["replay"]["memory_entries"]))
    entries = append_decision_discipline(
        (),
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=None,
        previous_decision_validation_closure=None,
        stage="BEFORE_DECISION",
        event_type="DECISION_CREATED",
        status="ACTIVE",
        summary="Independent discipline record.",
        event_at=_later(AS_OF),
        expected_tip=ZERO_SHA256,
    )
    assert entries[0]["version"].endswith("decision-discipline-entry.v1")
    assert canonical_bytes(list(stack["replay"]["memory_entries"])) == original
    assert all(
        row["version"] == "myquant.v17.research-intelligence.memory-entry.v1"
        for row in stack["replay"]["memory_entries"]
    )


def test_serialized_artifacts_have_no_trading_control_schema(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    memo = build_investment_memo(
        context=stack["context"],
        context_replay_closure=stack["context_closure"],
        policy=stack["policy"],
        risk_receipt=stack["risk"],
        assessments_by_dimension=stack["assessments"],
        decision_receipt=stack["decision"],
        as_of=AS_OF,
    )
    proposal = build_paper_intake_proposal(
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        proposed_at=_later(AS_OF),
    )
    keys = _walk_keys((stack["context"], stack["risk"], stack["decision"], memo, proposal))
    assert {"action", "cash", "holding", "quantity", "side", "target_price", "weight"}.isdisjoint(
        {key.casefold() for key in keys}
    )
    assert stack["decision"]["state"] not in {"BUY", "HOLD", "SELL"}


def test_decision_sources_have_no_io_provider_model_or_registration_surface() -> None:
    source_root = Path(decision_api.__file__).resolve().parent
    sources = sorted(source_root.glob("*.py"))
    assert {path.name for path in sources} == {
        "__init__.py",
        "decision_engine.py",
        "discipline_engine.py",
        "evidence_collector.py",
        "llm_research_interface.py",
        "memo_generator.py",
        "models.py",
        "paper_adapter.py",
        "receipts.py",
        "risk_assessor.py",
    }
    forbidden_import_roots = {
        "openai",
        "requests",
        "tushare",
        "urllib",
        "yfinance",
    }
    forbidden_calls = {
        "open",
        "write_bytes",
        "write_text",
        "mkdir",
        "makedirs",
        "unlink",
    }
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in forbidden_import_roots for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_import_roots
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                assert name not in forbidden_calls, f"{path.name} contains forbidden call {name}"
