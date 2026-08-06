"""Exact-replay collector for the I1 investment decision context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Final

from .._core import (
    IntelligenceContractError,
    assert_no_authority,
    content_ref,
    sha256,
    timestamp,
    validate_content_addressed,
)
from ..bayesian.engine import validate_bayesian_receipt
from ..evaluator.forward_evaluator import (
    ForwardEvaluationError,
    ImplementationIntegrityError,
    _load_request,
    _replay_origins,
    run_forward_research_evaluation,
)
from ..evaluator.receipts import (
    CALIBRATION_RECEIPT_VERSION,
    ENVELOPE_VERSION,
    FACTOR_RECEIPT_VERSION,
    HYPOTHESIS_RECEIPT_VERSION,
    MAIN_RECEIPT_VERSION,
    MEMORY_PROPOSAL_VERSION,
    REGIME_RECEIPT_VERSION,
    UNIVERSE_INVENTORY_VERSION,
    VARIANT_COMPARISON_VERSION,
    VARIANT_FACTOR_RECEIPT_VERSION,
    validate_closed_receipt,
)
from ..evidence.forward_adapter import ExactArtifactReader
from ..evidence.models import validate_evidence_set
from ..fusion.branches import validate_branch
from ..fusion.engine import validate_fusion_receipt
from ..hypothesis.models import validate_hypothesis
from ..runtime import (
    build_intelligence_runtime_receipt,
    verify_runtime_receipt,
)
from .llm_research_interface import validate_decision_ai_draft
from .models import (
    CONTEXT_VERSION,
    DecisionContractError,
    canonical_timestamp,
    company_code as normalize_company_code,
    ensure_artifact_size,
    fail,
    sorted_content_refs,
    validate_context_note,
    validate_decision_policy,
)
from .receipts import seal_artifact, validate_closed_artifact

I0_REPLAY_INPUT_FIELDS: Final = {
    "bayesian_receipts",
    "branches",
    "closure_refs",
    "evaluation_refs",
    "evidence",
    "expected_memory_tip",
    "fusion_receipt",
    "hypotheses",
    "label_refs",
    "memory_entries",
    "observation_bundle",
    "observation_refs",
    "regime_input",
    "regime_receipt",
    "session_byte_sha256",
    "session_relative_path",
    "workspace_root",
}

CONTEXT_REPLAY_CLOSURE_FIELDS: Final = {
    "ai_drafts",
    "context_notes",
    "i0_replay_inputs",
    "policy",
    "r22_request_path",
    "r22_request_sha256",
}

AVAILABILITY_CLASSES: Final = (
    "INDUSTRY_CONTEXT",
    "THEME_CONTEXT",
    "VALUATION_CONTEXT",
    "WHY_NOW",
    "AI_DRAFT",
    "R22_EVALUATION",
)

_CONTEXT_PAYLOAD_FIELDS: Final = {
    "ai_draft_refs",
    "as_of",
    "availability",
    "bayesian_ref",
    "company_code",
    "company_display_name_ref",
    "evidence_refs",
    "fundamental_branch_ref",
    "fusion_ref",
    "hypothesis_ref",
    "note_refs",
    "observation_bundle_ref",
    "policy_ref",
    "quant_branch_ref",
    "r22_envelope_ref",
    "r22_hypothesis_evaluation_ref",
    "r22_hypothesis_status",
    "r22_main_ref",
    "regime_input_ref",
    "regime_receipt_ref",
    "review_due_at",
    "runtime_receipt_ref",
}

_R22_ENVELOPE_PAYLOAD_FIELDS: Final = {
    "calibration_evidence",
    "factor_evaluations",
    "hypothesis_evaluations",
    "main_receipt",
    "memory_proposal",
    "regime_evaluation",
    "request_ref",
    "universe_inventory",
    "variant_evaluation",
    "variant_factor_evaluations",
}

_R22_MAIN_PAYLOAD_FIELDS: Final = {
    "calibration_ref",
    "evaluation_artifact_refs",
    "evaluation_window",
    "factor_refs",
    "hypothesis_evaluation_refs",
    "hypothesis_refs",
    "implementation_sha",
    "label_refs",
    "limitations",
    "memory_proposal_ref",
    "metrics",
    "observation_refs",
    "policy_ref",
    "regime_ref",
    "request_ref",
    "source_evaluation_refs",
    "universe_ref",
    "variant_ref",
}


def _sequence(value: Any, *, label: str, maximum: int | None = None) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        fail("I1_SHAPE_INVALID", f"{label} must be a sequence")
    rows = list(value)
    if maximum is not None and len(rows) > maximum:
        fail("I1_SHAPE_INVALID", f"{label} exceeds its cardinality limit")
    return rows


def _normalized_i0_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != I0_REPLAY_INPUT_FIELDS:
        fail("I1_SHAPE_INVALID", "i0_replay_inputs shape is not closed")
    result = dict(value)
    for field in (
        "observation_refs",
        "closure_refs",
        "evidence",
        "bayesian_receipts",
        "branches",
        "hypotheses",
        "memory_entries",
        "label_refs",
        "evaluation_refs",
    ):
        result[field] = _sequence(result[field], label=f"i0_replay_inputs.{field}")
    if type(result["workspace_root"]) is not str or not result["workspace_root"]:
        fail("I1_SHAPE_INVALID", "i0_replay_inputs.workspace_root is required")
    if type(result["session_relative_path"]) is not str:
        fail(
            "I1_SHAPE_INVALID",
            "i0_replay_inputs.session_relative_path is required",
        )
    try:
        sha256(result["session_byte_sha256"], label="session_byte_sha256")
        sha256(result["expected_memory_tip"], label="expected_memory_tip")
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_SHAPE_INVALID", str(exc)) from exc
    return result


def validate_context_replay_closure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact closure shape carried between I1 stages."""

    if type(value) is not dict or set(value) != CONTEXT_REPLAY_CLOSURE_FIELDS:
        fail("I1_SHAPE_INVALID", "context_replay_closure shape is not closed")
    result = dict(value)
    result["i0_replay_inputs"] = _normalized_i0_inputs(result["i0_replay_inputs"])
    result["context_notes"] = _sequence(
        result["context_notes"],
        label="context_replay_closure.context_notes",
        maximum=32,
    )
    result["ai_drafts"] = _sequence(
        result["ai_drafts"],
        label="context_replay_closure.ai_drafts",
        maximum=16,
    )
    path = result["r22_request_path"]
    digest = result["r22_request_sha256"]
    if (path is None) != (digest is None):
        fail(
            "I1_R22_CLOSURE_INVALID",
            "R2.2 request path and SHA must be provided together",
        )
    if path is not None:
        if type(path) is not str or not path:
            fail("I1_R22_CLOSURE_INVALID", "R2.2 request path is invalid")
        try:
            sha256(digest, label="r22_request_sha256")
        except IntelligenceContractError as exc:
            raise DecisionContractError("I1_R22_CLOSURE_INVALID", str(exc)) from exc
    return result


def build_context_replay_closure(
    *,
    i0_replay_inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    context_notes: Sequence[Mapping[str, Any]] = (),
    ai_drafts: Sequence[Mapping[str, Any]] = (),
    r22_request_path: str | None = None,
    r22_request_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the exact replay mapping expected by later I1 validators."""

    return validate_context_replay_closure(
        {
            "ai_drafts": list(ai_drafts),
            "context_notes": list(context_notes),
            "i0_replay_inputs": dict(i0_replay_inputs),
            "policy": dict(policy),
            "r22_request_path": r22_request_path,
            "r22_request_sha256": r22_request_sha256,
        }
    )


def _ref_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value.get("artifact_id")),
        str(value.get("artifact_version")),
        str(value.get("byte_sha256")),
        str(value.get("semantic_sha256")),
    )


def _assert_closed_r22_artifact(
    value: Mapping[str, Any], *, version: str, identity_field: str, as_of: str
) -> dict[str, Any]:
    try:
        row = validate_content_addressed(value, identity_field=identity_field)
        if row.get("version") != version:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 embedded artifact version mismatch",
            )
        assert_no_authority(row)
        if any(row.get(field) is not False for field in ("broker", "execution", "order", "trade")):
            fail(
                "I1_AUTHORITY_OPEN",
                "R2.2 embedded artifact action authority is open",
            )
        if (
            row.get("decision_protocol") != "myquant.v17.v4"
            or row.get("mainline_authority") is not False
            or row.get("operational_activation_unchanged") is not True
        ):
            fail(
                "I1_AUTHORITY_OPEN",
                "R2.2 embedded artifact protocol state is open",
            )
        if timestamp(row.get("timestamp"), label="R2.2 artifact.timestamp") > as_of:
            fail("I1_FUTURE_INPUT", "R2.2 artifact is after context as_of")
        return row
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_R22_CLOSURE_INVALID", str(exc)) from exc


def _r22_ref(value: Mapping[str, Any], *, identity_field: str) -> dict[str, str]:
    try:
        return content_ref(value, identity_field=identity_field)
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_R22_CLOSURE_INVALID", str(exc)) from exc


def _validate_r22_envelope(
    envelope: Mapping[str, Any],
    *,
    selected_hypothesis: Mapping[str, Any],
    preregistered: bool,
    request_path: str,
    request_sha256: str,
    as_of: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    try:
        row = validate_closed_receipt(
            envelope,
            version=ENVELOPE_VERSION,
            identity_field="envelope_id",
            payload_fields=_R22_ENVELOPE_PAYLOAD_FIELDS,
        )
        if timestamp(row["timestamp"], label="R2.2 envelope.timestamp") > as_of:
            fail("I1_FUTURE_INPUT", "R2.2 envelope is after context as_of")
        request_ref = row["request_ref"]
        if (
            type(request_ref) is not dict
            or request_ref.get("relative_path") != request_path
            or request_ref.get("byte_sha256") != request_sha256
        ):
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 request ref does not match exact replay",
            )
        main = validate_closed_receipt(
            row["main_receipt"],
            version=MAIN_RECEIPT_VERSION,
            identity_field="evaluation_id",
            payload_fields=_R22_MAIN_PAYLOAD_FIELDS,
        )
        if main["request_ref"] != request_ref:
            fail("I1_R22_CLOSURE_INVALID", "R2.2 main request ref mismatch")
        sha256(main["implementation_sha"], label="R2.2 implementation_sha")

        universe = _assert_closed_r22_artifact(
            row["universe_inventory"],
            version=UNIVERSE_INVENTORY_VERSION,
            identity_field="inventory_id",
            as_of=as_of,
        )
        factor_rows = _sequence(row["factor_evaluations"], label="R2.2 factor_evaluations")
        variant_factor_rows = _sequence(
            row["variant_factor_evaluations"],
            label="R2.2 variant_factor_evaluations",
        )
        hypothesis_rows = _sequence(
            row["hypothesis_evaluations"], label="R2.2 hypothesis_evaluations"
        )
        if not factor_rows or not variant_factor_rows or not hypothesis_rows:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 embedded topology is incomplete",
            )
        factor_rows = [
            _assert_closed_r22_artifact(
                item,
                version=FACTOR_RECEIPT_VERSION,
                identity_field="receipt_id",
                as_of=as_of,
            )
            for item in factor_rows
        ]
        variant_factor_rows = [
            _assert_closed_r22_artifact(
                item,
                version=VARIANT_FACTOR_RECEIPT_VERSION,
                identity_field="receipt_id",
                as_of=as_of,
            )
            for item in variant_factor_rows
        ]
        hypothesis_rows = [
            _assert_closed_r22_artifact(
                item,
                version=HYPOTHESIS_RECEIPT_VERSION,
                identity_field="receipt_id",
                as_of=as_of,
            )
            for item in hypothesis_rows
        ]
        variant = _assert_closed_r22_artifact(
            row["variant_evaluation"],
            version=VARIANT_COMPARISON_VERSION,
            identity_field="receipt_id",
            as_of=as_of,
        )
        calibration = _assert_closed_r22_artifact(
            row["calibration_evidence"],
            version=CALIBRATION_RECEIPT_VERSION,
            identity_field="receipt_id",
            as_of=as_of,
        )
        regime = _assert_closed_r22_artifact(
            row["regime_evaluation"],
            version=REGIME_RECEIPT_VERSION,
            identity_field="receipt_id",
            as_of=as_of,
        )
        memory = _assert_closed_r22_artifact(
            row["memory_proposal"],
            version=MEMORY_PROPOSAL_VERSION,
            identity_field="receipt_id",
            as_of=as_of,
        )

        factor_refs = [_r22_ref(item, identity_field="receipt_id") for item in factor_rows]
        variant_factor_refs = [
            _r22_ref(item, identity_field="receipt_id") for item in variant_factor_rows
        ]
        hypothesis_refs = [_r22_ref(item, identity_field="receipt_id") for item in hypothesis_rows]
        embedded_list_refs = [
            *factor_refs,
            *variant_factor_refs,
            *hypothesis_refs,
        ]
        if len({_ref_key(ref) for ref in embedded_list_refs}) != len(embedded_list_refs):
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 embedded lists contain duplicate content refs",
            )
        if (
            main["factor_refs"] != factor_refs
            or main["hypothesis_evaluation_refs"] != hypothesis_refs
        ):
            fail("I1_R22_CLOSURE_INVALID", "R2.2 main embedded refs mismatch")
        direct_refs = {
            "universe_ref": _r22_ref(universe, identity_field="inventory_id"),
            "variant_ref": _r22_ref(variant, identity_field="receipt_id"),
            "calibration_ref": _r22_ref(calibration, identity_field="receipt_id"),
            "regime_ref": _r22_ref(regime, identity_field="receipt_id"),
            "memory_proposal_ref": _r22_ref(memory, identity_field="receipt_id"),
        }
        if any(main[field] != ref for field, ref in direct_refs.items()):
            fail("I1_R22_CLOSURE_INVALID", "R2.2 main direct ref mismatch")
        expected_evaluation_refs = sorted(
            [
                direct_refs["universe_ref"],
                *factor_refs,
                *variant_factor_refs,
                direct_refs["variant_ref"],
                *hypothesis_refs,
                direct_refs["calibration_ref"],
                direct_refs["regime_ref"],
                direct_refs["memory_proposal_ref"],
            ],
            key=lambda ref: (
                ref["artifact_version"],
                ref["artifact_id"],
                ref["byte_sha256"],
            ),
        )
        if main["evaluation_artifact_refs"] != expected_evaluation_refs:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 evaluation artifact topology mismatch",
            )
        for values, identity_field in (
            (factor_rows, "factor_id"),
            (variant_factor_rows, "variant_id"),
        ):
            identities = [str(item.get(identity_field)) for item in values]
            if len(identities) != len(set(identities)):
                fail(
                    "I1_R22_CLOSURE_INVALID",
                    "R2.2 subject topology contains duplicates",
                )
        origin_rows = universe.get("rows")
        if type(origin_rows) is not list or not origin_rows:
            fail("I1_R22_CLOSURE_INVALID", "R2.2 universe inventory is empty")
        origin_ids = [str(item.get("origin_id")) for item in origin_rows]
        if len(origin_ids) != len(set(origin_ids)):
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 universe topology contains duplicates",
            )

        selected_ref = content_ref(selected_hypothesis, identity_field="hypothesis_id")
        if main.get("hypothesis_refs", []).count(selected_ref) != 1:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 does not uniquely bind selected hypothesis",
            )
        selected = [item for item in hypothesis_rows if item.get("hypothesis_ref") == selected_ref]
        if len(selected) != 1:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 selected hypothesis evaluation is not unique",
            )
        selected_evaluation = selected[0]
        selected_evaluation_ref = _r22_ref(selected_evaluation, identity_field="receipt_id")
        if hypothesis_refs.count(selected_evaluation_ref) != 1:
            fail(
                "I1_R22_CLOSURE_INVALID",
                "R2.2 selected evaluation is not in main refs",
            )
        status = selected_evaluation.get("hypothesis_status")
        if status not in {"SUPPORTED", "FAILED", "UNCERTAIN"}:
            fail("I1_R22_CLOSURE_INVALID", "R2.2 hypothesis status is invalid")
        if status == "FAILED":
            # R2.2 compacts evidence_summary before sealing. I1 therefore
            # replays the exact request origins separately and requires that
            # positive preregistration fact as well as the post-hoc boundary.
            if not preregistered or (
                "POSTHOC_POLICY_CONCLUSIONS_DOWNGRADED" in main.get("limitations", [])
            ):
                fail(
                    "I1_R22_CLOSURE_INVALID",
                    "post-hoc R2.2 result cannot invalidate a thesis",
                )
        return (
            _r22_ref(row, identity_field="envelope_id"),
            _r22_ref(main, identity_field="evaluation_id"),
            selected_evaluation_ref,
            str(status),
        )
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_R22_CLOSURE_INVALID", str(exc)) from exc


def _collect(
    *,
    i0_replay_inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    company_code: str,
    as_of: str,
    review_due_at: str,
    context_notes: Sequence[Mapping[str, Any]],
    ai_drafts: Sequence[Mapping[str, Any]],
    r22_request_path: str | None,
    r22_request_sha256: str | None,
) -> dict[str, Any]:
    cutoff = canonical_timestamp(as_of, label="as_of")
    due = canonical_timestamp(review_due_at, label="review_due_at")
    subject = normalize_company_code(company_code)
    policy_row = validate_decision_policy(policy)
    if policy_row["timestamp"] > cutoff:
        fail("I1_FUTURE_INPUT", "decision policy is after context as_of")
    if not cutoff < due:
        fail("I1_POLICY_INVALID", "review_due_at must be after as_of")
    start = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    end = datetime.fromisoformat(due.replace("Z", "+00:00"))
    if end > start + timedelta(seconds=policy_row["max_review_delay_seconds"]):
        fail("I1_POLICY_INVALID", "review_due_at exceeds decision policy")

    inputs = _normalized_i0_inputs(i0_replay_inputs)
    runtime_cutoff = canonical_timestamp(
        inputs["fusion_receipt"].get("timestamp"),
        label="i0_replay_inputs.fusion_receipt.timestamp",
    )
    if runtime_cutoff > cutoff:
        fail("I1_FUTURE_INPUT", "frozen I0 closure is after context as_of")
    try:
        runtime = build_intelligence_runtime_receipt(as_of=runtime_cutoff, **inputs)
        verify_runtime_receipt(runtime)
        evidence = validate_evidence_set(inputs["evidence"], as_of=runtime_cutoff)
        hypotheses = [
            validate_hypothesis(item, evidence=evidence, as_of=runtime_cutoff)
            for item in inputs["hypotheses"]
        ]
        matches = [item for item in hypotheses if subject in item["related_companies"]]
        if len(matches) != 1:
            fail(
                "I1_REF_MISMATCH",
                "company_code must select exactly one I0 hypothesis",
            )
        hypothesis = matches[0]
        bayesian = [
            validate_bayesian_receipt(item, evidence=evidence, as_of=runtime_cutoff)
            for item in inputs["bayesian_receipts"]
        ]
        selected_bayesian = [
            item for item in bayesian if item.get("hypothesis_id") == hypothesis["hypothesis_id"]
        ]
        if len(selected_bayesian) != 1:
            fail(
                "I1_REF_MISMATCH",
                "selected hypothesis must have exactly one Bayesian receipt",
            )
        branches = [
            validate_branch(item, evidence=evidence, as_of=runtime_cutoff)
            for item in inputs["branches"]
        ]
        branch_by_type: dict[str, dict[str, Any]] = {}
        for item in branches:
            branch_type = str(item["branch_type"])
            if branch_type in branch_by_type:
                fail("I1_REF_MISMATCH", "I1 branch type is not unique")
            branch_by_type[branch_type] = item
        if set(branch_by_type) != {"QUANT", "FUNDAMENTAL"}:
            fail(
                "I1_REF_MISMATCH",
                "I1 requires exactly Quant and Fundamental branches",
            )
        fusion = validate_fusion_receipt(
            inputs["fusion_receipt"],
            branches=branches,
            as_of=runtime_cutoff,
        )
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_REPLAY_MISMATCH", str(exc)) from exc

    bundle = inputs["observation_bundle"]
    authorized_refs = bundle.get("authorized_evidence_refs", [])
    notes = _sequence(context_notes, label="context_notes", maximum=32)
    validated_notes = [
        validate_context_note(
            item,
            as_of=cutoff,
            authorized_source_refs=authorized_refs,
        )
        for item in notes
    ]
    if any(item["company_code"] != subject for item in validated_notes):
        fail("I1_REF_MISMATCH", "context note company_code mismatch")
    note_ids = [str(item["note_id"]) for item in validated_notes]
    if len(note_ids) != len(set(note_ids)):
        fail("I1_SHAPE_INVALID", "context notes contain duplicates")
    notes_by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in validated_notes:
        notes_by_kind.setdefault(str(item["kind"]), []).append(item)
    if any(len(rows) > 1 for rows in notes_by_kind.values()):
        fail("I1_SHAPE_INVALID", "context note company/kind is not unique")
    display_notes = notes_by_kind.get("COMPANY_DISPLAY_NAME", [])

    drafts = _sequence(ai_drafts, label="ai_drafts", maximum=16)
    validated_drafts = [
        validate_decision_ai_draft(
            item,
            as_of=cutoff,
            authorized_source_refs=authorized_refs,
        )
        for item in drafts
    ]
    draft_ids = [str(item["draft_id"]) for item in validated_drafts]
    if len(draft_ids) != len(set(draft_ids)):
        fail("I1_SHAPE_INVALID", "AI drafts contain duplicates")

    r22_values: tuple[
        dict[str, str] | None,
        dict[str, str] | None,
        dict[str, str] | None,
        str | None,
    ]
    if r22_request_path is None and r22_request_sha256 is None:
        r22_values = (None, None, None, None)
    elif r22_request_path is None or r22_request_sha256 is None:
        fail(
            "I1_R22_CLOSURE_INVALID",
            "R2.2 request path and SHA must be provided together",
        )
    else:
        try:
            sha256(r22_request_sha256, label="r22_request_sha256")
            reader = ExactArtifactReader(inputs["workspace_root"])
            request, _ = _load_request(
                reader,
                request_path=r22_request_path,
                request_sha256=r22_request_sha256,
            )
            origins = _replay_origins(
                workspace_root=inputs["workspace_root"],
                request=request,
                reader=reader,
            )
            preregistered_values = {origin.get("preregistered") for origin in origins}
            if preregistered_values not in ({True}, {False}):
                fail(
                    "I1_R22_CLOSURE_INVALID",
                    "R2.2 preregistration proof is not closed",
                )
            preregistered = preregistered_values == {True}
            envelope = run_forward_research_evaluation(
                inputs["workspace_root"],
                request_path=r22_request_path,
                request_sha256=r22_request_sha256,
            )
        except (
            ForwardEvaluationError,
            ImplementationIntegrityError,
            IntelligenceContractError,
        ) as exc:
            raise DecisionContractError("I1_R22_CLOSURE_INVALID", str(exc)) from exc
        r22_values = _validate_r22_envelope(
            envelope,
            selected_hypothesis=hypothesis,
            preregistered=preregistered,
            request_path=r22_request_path,
            request_sha256=r22_request_sha256,
            as_of=cutoff,
        )

    note_refs = sorted_content_refs(
        [content_ref(item, identity_field="note_id") for item in validated_notes],
        label="note_refs",
        maximum=32,
    )
    draft_refs = sorted_content_refs(
        [content_ref(item, identity_field="draft_id") for item in validated_drafts],
        label="ai_draft_refs",
        maximum=16,
    )
    availability: dict[str, dict[str, Any]] = {}
    for kind in AVAILABILITY_CLASSES[:4]:
        refs = sorted_content_refs(
            [
                content_ref(item, identity_field="note_id")
                for item in validated_notes
                if item["kind"] == kind
            ],
            label=f"availability.{kind}.refs",
        )
        availability[kind] = {
            "refs": refs,
            "status": "AVAILABLE" if refs else "UNAVAILABLE",
        }
    availability["AI_DRAFT"] = {
        "refs": draft_refs,
        "status": "AVAILABLE" if draft_refs else "UNAVAILABLE",
    }
    r22_evaluation_refs = [] if r22_values[2] is None else [r22_values[2]]
    availability["R22_EVALUATION"] = {
        "refs": r22_evaluation_refs,
        "status": "AVAILABLE" if r22_evaluation_refs else "UNAVAILABLE",
    }
    document = seal_artifact(
        version=CONTEXT_VERSION,
        identity_field="context_id",
        timestamp_value=cutoff,
        payload={
            "ai_draft_refs": draft_refs,
            "as_of": cutoff,
            "availability": availability,
            "bayesian_ref": content_ref(selected_bayesian[0], identity_field="receipt_id"),
            "company_code": subject,
            "company_display_name_ref": (
                None
                if not display_notes
                else content_ref(display_notes[0], identity_field="note_id")
            ),
            "evidence_refs": sorted_content_refs(
                [content_ref(item, identity_field="evidence_id") for item in evidence],
                label="evidence_refs",
                maximum=256,
            ),
            "fundamental_branch_ref": content_ref(
                branch_by_type["FUNDAMENTAL"], identity_field="branch_id"
            ),
            "fusion_ref": content_ref(fusion, identity_field="receipt_id"),
            "hypothesis_ref": content_ref(hypothesis, identity_field="hypothesis_id"),
            "note_refs": note_refs,
            "observation_bundle_ref": content_ref(bundle, identity_field="bundle_id"),
            "policy_ref": content_ref(policy_row, identity_field="policy_id"),
            "quant_branch_ref": content_ref(branch_by_type["QUANT"], identity_field="branch_id"),
            "r22_envelope_ref": r22_values[0],
            "r22_hypothesis_evaluation_ref": r22_values[2],
            "r22_hypothesis_status": r22_values[3],
            "r22_main_ref": r22_values[1],
            "regime_input_ref": content_ref(inputs["regime_input"], identity_field="input_id"),
            "regime_receipt_ref": content_ref(
                inputs["regime_receipt"], identity_field="receipt_id"
            ),
            "review_due_at": due,
            "runtime_receipt_ref": content_ref(runtime, identity_field="runtime_receipt_id"),
        },
    )
    ensure_artifact_size(document)
    return document


def collect_investment_decision_context(
    *,
    i0_replay_inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    company_code: str,
    as_of: str,
    review_due_at: str,
    context_notes: Sequence[Mapping[str, Any]] = (),
    ai_drafts: Sequence[Mapping[str, Any]] = (),
    r22_request_path: str | None = None,
    r22_request_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay the exact I0/R2.2 closure and build one sealed I1 context."""

    try:
        return _collect(
            i0_replay_inputs=i0_replay_inputs,
            policy=policy,
            company_code=company_code,
            as_of=as_of,
            review_due_at=review_due_at,
            context_notes=context_notes,
            ai_drafts=ai_drafts,
            r22_request_path=r22_request_path,
            r22_request_sha256=r22_request_sha256,
        )
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_REPLAY_MISMATCH", str(exc)) from exc


def validate_investment_decision_context(
    document: Mapping[str, Any],
    *,
    i0_replay_inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    context_notes: Sequence[Mapping[str, Any]] = (),
    ai_drafts: Sequence[Mapping[str, Any]] = (),
    r22_request_path: str | None = None,
    r22_request_sha256: str | None = None,
) -> dict[str, Any]:
    """Fully replay a Decision Context and require byte-for-byte equality."""

    try:
        row = validate_closed_artifact(
            document,
            version=CONTEXT_VERSION,
            identity_field="context_id",
            payload_fields=_CONTEXT_PAYLOAD_FIELDS,
        )
        if row.get("timestamp") != row.get("as_of"):
            fail("I1_SHAPE_INVALID", "Decision Context timestamp/as_of mismatch")
        expected = collect_investment_decision_context(
            i0_replay_inputs=i0_replay_inputs,
            policy=policy,
            company_code=row.get("company_code"),
            as_of=row.get("as_of"),
            review_due_at=row.get("review_due_at"),
            context_notes=context_notes,
            ai_drafts=ai_drafts,
            r22_request_path=r22_request_path,
            r22_request_sha256=r22_request_sha256,
        )
        if expected != row:
            fail("I1_REPLAY_MISMATCH", "Decision Context replay mismatch")
        return row
    except DecisionContractError:
        raise
    except IntelligenceContractError as exc:
        raise DecisionContractError("I1_SHAPE_INVALID", str(exc)) from exc


__all__ = [
    "AVAILABILITY_CLASSES",
    "CONTEXT_REPLAY_CLOSURE_FIELDS",
    "I0_REPLAY_INPUT_FIELDS",
    "build_context_replay_closure",
    "collect_investment_decision_context",
    "validate_context_replay_closure",
    "validate_investment_decision_context",
]
