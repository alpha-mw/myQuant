from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping

import pytest

from quant_investor.intelligence._core import (
    NO_AUTHORITY,
    ZERO_SHA256,
    IntelligenceContractError,
    content_ref,
    seal_content_addressed,
)
from quant_investor.intelligence.evaluator import (
    evaluate_factor,
    evaluate_hypothesis,
    evaluate_regimes,
    evaluate_variants,
)
from quant_investor.intelligence.evaluator.cli import main as evaluator_cli
from quant_investor.intelligence.evaluator.factor_evaluator import METRIC_IDS
from quant_investor.intelligence.evaluator.forward_evaluator import (
    _calibration_receipt,
    _hypothesis_receipts,
    _memory_proposal,
    _origin_identity,
    _rule_shape,
    run_forward_research_evaluation,
)
from quant_investor.intelligence.evaluator.receipts import (
    build_hypothesis_receipt,
    build_memory_inventory,
    build_subject_receipt,
)
from quant_investor.intelligence.evidence import build_evidence
from quant_investor.intelligence.evidence.forward_adapter import ExactArtifactReader
from quant_investor.intelligence.hypothesis import build_hypothesis
from quant_investor.intelligence.memory import validate_memory_chain
from quant_investor.intelligence.regime import infer_multilayer_regime
from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from tests.unit.test_v17_i0_investment_intelligence import (
    _artifact_ref as _v4_artifact_ref,
)
from tests.unit.test_v17_i0_investment_intelligence import _forward_closure
from tests.unit.test_v17_i0_investment_intelligence import _regime_input as _i0_regime_input

AS_OF = "2026-01-02T07:00:00Z"
EVALUATED_AT = "2026-02-27T07:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _source_ref(name: str = "source-a") -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": "myquant.v17.v4.research-source.v1",
        "byte_sha256": SHA_A,
        "cutoff": AS_OF,
        "relative_path": f"data/private/v17_v4_sources/{name}.json",
        "semantic_sha256": SHA_B,
        "strategy_id": "cn-forward-research",
    }


def _evidence(
    *,
    direction: str = "POSITIVE",
    name: str = "source-a",
    source_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    likelihood = "2" if direction == "POSITIVE" else "0.4"
    return build_evidence(
        source_type="QUANT",
        direction=direction,
        likelihood_ratio=likelihood,
        strength="0.8",
        reason=f"Preregistered {direction.lower()} factor evidence.",
        observed_at=AS_OF,
        available_at=AS_OF,
        source_ref=_source_ref(name) if source_ref is None else source_ref,
    )


def _hypothesis() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _evidence()
    contrary = _evidence(direction="CONTRARY", name="source-b")
    hypothesis = build_hypothesis(
        thesis="Forward RankIC remains positive.",
        why_it_may_be_true="The factor has a stable economic mechanism.",
        what_would_make_it_fail="A negative RankIC would falsify the claim.",
        supporting_evidence=[evidence],
        contrary_evidence=[contrary],
        expected_window_start="2026-01-02T07:00:00Z",
        expected_window_end="2026-02-27T23:59:59Z",
        falsification_conditions=[
            {
                "metric_id": "rank_ic",
                "operator": "LT",
                "threshold": "0",
                "window_sessions": 1,
            }
        ],
        related_companies=["000001.SZ"],
        related_industries=["banks"],
        as_of=AS_OF,
    )
    return hypothesis, evidence


def _symbol_rows(offset: str = "0") -> list[dict[str, Any]]:
    shift = Decimal(offset)
    rows: list[dict[str, Any]] = []
    for index in range(5):
        total_return = Decimal(index + 1) / Decimal("100") + shift
        rows.append(
            {
                "cost_adjusted_return": str(total_return - Decimal("0.002")),
                "industry_adjusted_return": str(total_return - Decimal("0.005")),
                "industry_id": "industry-a" if index < 3 else "industry-b",
                "score": str(index + 1),
                "score_status": "AVAILABLE",
                "symbol": f"00000{index + 1}.SZ",
                "total_return": str(total_return),
            }
        )
    return rows


def _origin(
    origin_id: str,
    origin_session: str,
    label_session: str,
    next_open_session: str,
    *,
    offset: str = "0",
) -> dict[str, Any]:
    return {
        "label_session": label_session,
        "next_open_session": next_open_session,
        "origin_id": origin_id,
        "origin_session": origin_session,
        "symbol_rows": _symbol_rows(offset),
    }


def _factor_result() -> dict[str, Any]:
    origins = [
        _origin("origin-a", "2026-01-02", "2026-01-05", "2026-01-05"),
        _origin(
            "origin-b",
            "2026-01-06",
            "2026-01-07",
            "2026-01-07",
            offset="0.001",
        ),
    ]
    return evaluate_factor(
        factor_id="factor-a",
        origins=origins,
        orientation="HIGHER_IS_BETTER",
        horizon_sessions=1,
        min_symbols=5,
        min_available_origins=1,
        min_joint_coverage="1",
        min_industry_mapping_coverage="1",
    )


def _metric_map(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["metric_id"]): row for row in result["metrics"]}


def _variant_metrics(*, rank_ic: str) -> dict[str, dict[str, Any]]:
    values = {
        "cost_adjusted_return": "0.01",
        "drawdown": "0.02",
        "icir": "0.3",
        "joint_coverage": "1",
        "long_short_spread": "0.02",
        "rank_ic": rank_ic,
        "turnover": "0.1",
    }
    return {
        key: {
            "input_origin_ids": ["origin-a", "origin-b"],
            "status": "AVAILABLE",
            "value": value,
        }
        for key, value in values.items()
    }


def _variant_rules() -> list[dict[str, str]]:
    directions = {
        "cost_adjusted_return": "HIGHER_IS_BETTER",
        "drawdown": "LOWER_IS_BETTER",
        "icir": "HIGHER_IS_BETTER",
        "joint_coverage": "HIGHER_IS_BETTER",
        "long_short_spread": "HIGHER_IS_BETTER",
        "rank_ic": "HIGHER_IS_BETTER",
        "turnover": "LOWER_IS_BETTER",
    }
    return [
        {
            "degradation_threshold": "0.01",
            "direction": directions[metric_id],
            "improvement_threshold": "0.01",
            "metric_id": metric_id,
            "tolerance": "0",
        }
        for metric_id in directions
    ]


def _hypothesis_spec(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    rule = {
        "aggregation": "MEAN",
        "factor_id": "factor-a",
        "label_field": "total_return",
        "metric_id": "rank_ic",
        "operator": "GTE",
        "threshold": "0.05",
        "window_end": "2026-02-27",
        "window_start": "2026-01-02",
    }
    return {
        "contrary_rules": [],
        "evidence_refs": [
            *hypothesis["supporting_evidence_refs"],
            *hypothesis["contrary_evidence_refs"],
        ],
        "falsification_bindings": [
            {
                "condition_index": 0,
                "factor_id": "factor-a",
                "label_field": "total_return",
                "metric_id": "rank_ic",
                "window_end": "2026-02-27",
                "window_start": "2026-01-02",
            }
        ],
        "hypothesis_ref": content_ref(hypothesis, identity_field="hypothesis_id"),
        "min_coverage": "0.8",
        "min_mature_origins": 1,
        "spec_id": "hypothesis-spec-a",
        "support_rules": [rule],
    }


def _metric_lookup(value: str | None, status: str = "AVAILABLE") -> dict[Any, Any]:
    return {
        ("factor-a", "rank_ic", "2026-01-02", "2026-02-27", "total_return"): {
            "input_origin_ids": ["origin-a"] if status == "AVAILABLE" else [],
            "status": status,
            "value": value,
        }
    }


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _walk_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def test_factor_metrics_are_deterministic_and_cover_the_fixed_inventory() -> None:
    first = _factor_result()
    second = _factor_result()
    assert first == second
    assert tuple(row["metric_id"] for row in first["metrics"]) == METRIC_IDS
    assert len(METRIC_IDS) == 20
    metrics = _metric_map(first)
    assert metrics["rank_ic"]["value"] == "1.000000000000"
    assert metrics["quantile_return_q1"]["value"] == "0.010500000000"
    assert metrics["quantile_return_q5"]["value"] == "0.050500000000"
    assert metrics["long_short_spread"]["value"] == "0.040000000000"
    assert metrics["cost_adjusted_return"]["value"] == "0.038000000000"
    assert metrics["joint_coverage"]["value"] == "1.000000000000"
    assert metrics["drawdown"]["status"] == "AVAILABLE"


def test_factor_missing_labels_fail_closed_without_imputation() -> None:
    origin = _origin("origin-a", "2026-01-02", "2026-01-05", "2026-01-05")
    for row in origin["symbol_rows"]:
        row["total_return"] = None
        row["cost_adjusted_return"] = None
        row["industry_adjusted_return"] = None
    result = evaluate_factor(
        factor_id="factor-a",
        origins=[origin],
        orientation="HIGHER_IS_BETTER",
        horizon_sessions=1,
        min_symbols=5,
        min_available_origins=1,
        min_joint_coverage="1",
        min_industry_mapping_coverage="1",
    )
    metrics = _metric_map(result)
    assert result["status"] == "UNAVAILABLE"
    assert metrics["rank_ic"]["status"] == "UNAVAILABLE"
    assert metrics["label_coverage"]["value"] == "0.000000000000"
    assert "INSUFFICIENT_JOINT_COVERAGE" in metrics["rank_ic"]["blocker_codes"]
    assert "COMPLETE_DRAWDOWN_PATH_UNAVAILABLE" in metrics["drawdown"]["blocker_codes"]


def test_variant_comparison_keeps_optional_missing_and_partial_explicit() -> None:
    result = evaluate_variants(
        variants={
            "v17-quant-core": {
                "available_origin_ids": ["origin-a", "origin-b"],
                "metrics": _variant_metrics(rank_ic="0.10"),
                "status": "COMPLETE",
            },
            "v17-quant-plus-industry": {
                "available_origin_ids": ["origin-a", "origin-b"],
                "metrics": _variant_metrics(rank_ic="0.12"),
                "status": "PARTIAL",
            },
            "v17-quant-plus-industry-theme": None,
        },
        rules=_variant_rules(),
    )
    assert result["industry_incremental_conclusion"] == "INCREMENTAL_POSITIVE"
    assert result["industry_theme_incremental_conclusion"] == "UNAVAILABLE"
    assert result["comparisons"][1]["blockers"] == ["OPTIONAL_VARIANT_UNAVAILABLE"]
    assert result["limitations"] == ["THEME_INCREMENT_IS_CUMULATIVE_VS_CORE"]


def test_variant_comparison_fails_closed_on_metric_specific_origin_mismatch() -> None:
    candidate_metrics = _variant_metrics(rank_ic="0.12")
    candidate_metrics["rank_ic"]["input_origin_ids"] = ["origin-a"]
    result = evaluate_variants(
        variants={
            "v17-quant-core": {
                "available_origin_ids": ["origin-a", "origin-b"],
                "metrics": _variant_metrics(rank_ic="0.10"),
                "status": "COMPLETE",
            },
            "v17-quant-plus-industry": {
                "available_origin_ids": ["origin-a", "origin-b"],
                "metrics": candidate_metrics,
                "status": "COMPLETE",
            },
            "v17-quant-plus-industry-theme": None,
        },
        rules=_variant_rules(),
    )
    comparison = result["comparisons"][0]
    rank_ic = next(row for row in comparison["metric_comparisons"] if row["metric_id"] == "rank_ic")
    assert comparison["conclusion"] == "INCONCLUSIVE"
    assert "PAIRED_ORIGIN_MISMATCH" in comparison["blockers"]
    assert rank_ic["status"] == "UNAVAILABLE"
    assert rank_ic["input_origin_ids"] == ["origin-a"]
    assert rank_ic["blocker_codes"] == ["PAIRED_ORIGIN_MISMATCH"]


@pytest.mark.parametrize(
    ("value", "preregistered", "expected_status", "expected_falsification"),
    [
        ("0.10", True, "SUPPORTED", "NOT_TRIGGERED"),
        ("-0.10", True, "FAILED", "TRIGGERED"),
        ("-0.10", False, "UNCERTAIN", "TRIGGERED"),
        (None, True, "UNCERTAIN", "INCONCLUSIVE"),
    ],
)
def test_hypothesis_support_failure_and_falsification_precedence(
    value: str | None,
    preregistered: bool,
    expected_status: str,
    expected_falsification: str,
) -> None:
    hypothesis, _ = _hypothesis()
    status = "AVAILABLE" if value is not None else "UNAVAILABLE"
    result = evaluate_hypothesis(
        hypothesis=hypothesis,
        spec=_hypothesis_spec(hypothesis),
        metric_lookup=_metric_lookup(value, status),
        preregistered=preregistered,
        mature_origin_count=1,
        joint_coverage="1",
    )
    assert result["hypothesis_status"] == expected_status
    assert result["falsification_result"] == expected_falsification
    assert result["evidence_summary"]["support_results"]


def test_hypothesis_rejects_label_fields_without_v1_computational_semantics() -> None:
    hypothesis, _ = _hypothesis()
    spec = _hypothesis_spec(hypothesis)
    spec["support_rules"][0]["label_field"] = "market_adjusted_return"
    with pytest.raises(IntelligenceContractError, match="label_field is not supported"):
        evaluate_hypothesis(
            hypothesis=hypothesis,
            spec=spec,
            metric_lookup=_metric_lookup("0.10"),
            preregistered=True,
            mature_origin_count=1,
            joint_coverage="1",
        )


def test_forward_policy_requires_session_date_windows() -> None:
    rule = _hypothesis_spec(_hypothesis()[0])["support_rules"][0]
    rule["window_start"] = "2026-01-02T23:59:59Z"
    with pytest.raises(IntelligenceContractError, match="canonical date"):
        _rule_shape(rule, label="support_rule")


def test_hypothesis_receipt_recomputes_and_caches_the_exact_registered_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hypothesis, evidence = _hypothesis()
    inside = _origin("origin-a", "2026-01-02", "2026-01-05", "2026-01-05")
    outside = _origin("origin-b", "2026-01-06", "2026-01-07", "2026-01-07")
    for index, row in enumerate(outside["symbol_rows"]):
        reverse_return = Decimal(5 - index) / Decimal("100")
        row["total_return"] = str(reverse_return)
        row["cost_adjusted_return"] = str(reverse_return - Decimal("0.002"))
        row["industry_adjusted_return"] = str(reverse_return - Decimal("0.005"))
    full = evaluate_factor(
        factor_id="factor-a",
        origins=[inside, outside],
        orientation="HIGHER_IS_BETTER",
        horizon_sessions=1,
        min_symbols=5,
        min_available_origins=1,
        min_joint_coverage="1",
        min_industry_mapping_coverage="1",
    )
    assert _metric_map(full)["rank_ic"]["value"] == "0.000000000000"
    factor_receipt = build_subject_receipt(
        subject_type="FACTOR",
        subject_id="factor-a",
        subject_ref=content_ref(evidence, identity_field="evidence_id"),
        evaluation_window={"horizon_sessions": 1, "origin_count": 2},
        universe_ref=content_ref(evidence, identity_field="evidence_id"),
        observation_refs=[],
        metrics=full["metrics"],
        origin_metrics=full["origin_metrics"],
        limitations=full["limitations"],
        evaluated_at=EVALUATED_AT,
    )
    spec = _hypothesis_spec(hypothesis)
    spec["support_rules"][0]["window_end"] = "2026-01-02"
    spec["falsification_bindings"][0]["window_end"] = "2026-01-02"
    policy = {
        "factor_specs": [{"direction": "HIGHER_IS_BETTER", "factor_id": "factor-a"}],
        "horizon_sessions": 1,
        "min_available_origins": 1,
        "min_industry_mapping_coverage": "1",
        "min_joint_coverage": "1",
        "min_symbols": 5,
    }
    calls = 0
    actual_evaluate_factor = evaluate_factor

    def counted_evaluate_factor(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return actual_evaluate_factor(**kwargs)

    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.factor_evaluator.evaluate_factor",
        counted_evaluate_factor,
    )
    receipts = _hypothesis_receipts(
        policy=policy,
        origins=[
            {
                "factor_origins": {"factor-a": inside},
                "origin_id": "origin-a",
                "origin_session": "2026-01-02",
            },
            {
                "factor_origins": {"factor-a": outside},
                "origin_id": "origin-b",
                "origin_session": "2026-01-06",
            },
        ],
        factor_receipts=[factor_receipt],
        hypotheses=[hypothesis, hypothesis],
        specs=[spec, spec],
        preregistered=True,
        evaluated_at=EVALUATED_AT,
    )
    assert receipts[0]["hypothesis_status"] == "SUPPORTED"
    assert receipts[1]["hypothesis_status"] == "SUPPORTED"
    assert calls == 1


def test_calibration_is_descriptive_and_has_no_posterior_mutation() -> None:
    _, evidence = _hypothesis()
    factor_receipt = build_subject_receipt(
        subject_type="FACTOR",
        subject_id="factor-a",
        subject_ref=content_ref(evidence, identity_field="evidence_id"),
        evaluation_window={"horizon_sessions": 1, "origin_count": 1},
        universe_ref=content_ref(evidence, identity_field="evidence_id"),
        observation_refs=[],
        metrics=[{"metric_id": "rank_ic", "status": "AVAILABLE", "value": "0.1"}],
        origin_metrics=[],
        limitations=[],
        evaluated_at=EVALUATED_AT,
    )
    policy = {
        "calibration_specs": [
            {
                "evidence_id": evidence["evidence_id"],
                "factor_id": "factor-a",
                "metric_id": "rank_ic",
                "min_mature_count": 1,
                "success_operator": "GTE",
                "success_threshold": "0.05",
            }
        ]
    }
    prior_state = {"posterior": "0.7", "semantic_sha256": SHA_A}
    before = deepcopy(prior_state)
    result = _calibration_receipt(
        policy=policy,
        evidence=[evidence],
        factor_receipts=[factor_receipt],
        preregistered=True,
        evaluated_at=EVALUATED_AT,
    )
    assert prior_state == before
    assert "posterior" not in _walk_keys(result)
    assert result["group_rows"][0]["success_rate"] == "1.000000000000"
    assert "BAYESIAN_CALIBRATION_DIAGNOSTIC_ONLY" in result["limitations"]


def test_regime_uses_selected_states_only_and_rejects_overlap_for_drawdown() -> None:
    subject = {
        "scope": "GLOBAL_BREADTH",
        "subject_id": "factor-a",
        "subject_type": "factor",
    }

    def origin(
        origin_id: str,
        origin_session: str,
        label_session: str,
        next_open_session: str,
        rank_ic: str,
    ) -> dict[str, Any]:
        return {
            "label_session": label_session,
            "next_open_session": next_open_session,
            "origin_id": origin_id,
            "origin_session": origin_session,
            "states": {"industry": "EXPANSION", "market": "BULL", "theme": "EMERGING"},
            "subjects": [
                {
                    "metrics": {
                        "cost_adjusted_return": "0.01",
                        "joint_coverage": "1",
                        "long_short_spread": "0.02",
                        "neutralized_alpha": "0.01",
                        "q5_long_only_cost_adjusted_return": "0.01",
                        "rank_ic": rank_ic,
                    },
                    "q5_weights": {"000001.SZ": "1"},
                    **subject,
                }
            ],
        }

    result = evaluate_regimes(
        origin_rows=[
            origin("origin-a", "2026-01-02", "2026-01-05", "2026-01-05", "0.1"),
            origin("origin-b", "2026-01-05", "2026-01-06", "2026-01-06", "0.2"),
        ],
        subject_ids=[subject],
        horizon_sessions=1,
        min_stratum_origins=1,
    )
    market_bull = result["layer_rows"][0]["state_rows"][0]
    metrics = {row["metric_id"]: row for row in market_bull["factor_metric_rows"]}
    assert metrics["rank_ic"]["value"] == "0.150000000000"
    assert metrics["drawdown"]["status"] == "UNAVAILABLE"
    assert metrics["drawdown"]["blocker_codes"] == ["OVERLAPPING_FORWARD_WINDOWS"]
    assert "SELECTED_STATES_ONLY_NO_POSTERIOR_INPUT" in result["limitations"]
    assert "NO_BACKWARD_SMOOTHING" in result["limitations"]


def test_regime_drawdown_rejects_returns_below_negative_one() -> None:
    subject = {
        "scope": "GLOBAL_BREADTH",
        "subject_id": "factor-a",
        "subject_type": "factor",
    }
    result = evaluate_regimes(
        origin_rows=[
            {
                "label_session": "2026-01-05",
                "next_open_session": "2026-01-05",
                "origin_id": "origin-a",
                "origin_session": "2026-01-02",
                "states": {"industry": "EXPANSION", "market": "BULL", "theme": "EMERGING"},
                "subjects": [
                    {
                        "metrics": {
                            "cost_adjusted_return": "-1.10",
                            "joint_coverage": "1",
                            "long_short_spread": "-1.10",
                            "neutralized_alpha": "-1.10",
                            "q5_long_only_cost_adjusted_return": "-1.10",
                            "rank_ic": "-1",
                        },
                        "q5_weights": {"000001.SZ": "1"},
                        **subject,
                    }
                ],
            }
        ],
        subject_ids=[subject],
        horizon_sessions=1,
        min_stratum_origins=1,
    )
    market_bull = result["layer_rows"][0]["state_rows"][0]
    drawdown = next(
        row for row in market_bull["factor_metric_rows"] if row["metric_id"] == "drawdown"
    )
    assert drawdown["status"] == "UNAVAILABLE"
    assert drawdown["blocker_codes"] == ["RETURN_NOT_COMPOUNDABLE"]


def test_memory_proposal_is_append_only_and_retains_failed_case() -> None:
    hypothesis, _ = _hypothesis()
    receipt = build_hypothesis_receipt(
        evaluated_at=EVALUATED_AT,
        hypothesis_ref=content_ref(hypothesis, identity_field="hypothesis_id"),
        hypothesis_status="FAILED",
    )
    inventory = build_memory_inventory(entries=[], timestamp_value="1970-01-01T00:00:00Z")
    proposal = _memory_proposal(
        inventory=inventory,
        hypotheses=[hypothesis],
        hypothesis_receipts=[receipt],
        evaluated_at=EVALUATED_AT,
    )
    assert inventory["entries"] == []
    assert inventory["tip"] == ZERO_SHA256
    events = [row["event_type"] for row in proposal["proposed_entries"]]
    assert events == ["EVALUATED", "HYPOTHESIS_FALSIFIED", "FAILED_CASE"]
    chain = validate_memory_chain(
        proposal["proposed_entries"], expected_tip=proposal["observed_after_tip"]
    )
    assert chain[-1]["event_type"] == "FAILED_CASE"
    assert chain[-1]["status"] == "FAILED"


def _write_canonical(root: Path, relative_path: str, document: Mapping[str, Any]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_resource_bytes(document))


def _research_input_ref(
    root: Path, document: Mapping[str, Any], *, identity_field: str
) -> dict[str, str]:
    artifact_id = str(document[identity_field])
    relative_path = f"data/private/research_intelligence/evaluation_inputs/{artifact_id}.json"
    raw = canonical_resource_bytes(document)
    _write_canonical(root, relative_path, document)
    return {
        "artifact_id": artifact_id,
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
    }


def _reseal_v4(document: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(document)
    body.pop("semantic_sha256", None)
    return seal_semantic(body)


def _expanded_forward_closure(
    root: Path,
) -> tuple[
    str,
    str,
    dict[str, str],
    dict[str, str],
    dict[str, str],
    list[dict[str, str]],
]:
    session_path, session_sha, _, _, _, closure_refs = _forward_closure(root)
    observation_path = "results/v17_v4_shadow/forward_observations/factor-a.json"
    label_path = "results/v17_v4_shadow/forward_labels/label-a.json"
    origin_inventory_path = "results/v17_v4_shadow/forward_evaluations/origin-a.json"
    factor_inventory_path = "results/v17_v4_shadow/forward_evaluations/inventory-a.json"
    evaluation_path = "results/v17_v4_shadow/forward_evaluations/evaluation-a.json"

    observation = json.loads((root / observation_path).read_text(encoding="utf-8"))
    observation["observations"] = [
        {"status": "AVAILABLE", "symbol": f"00000{index + 1}.SZ", "value": str(index + 1)}
        for index in range(5)
    ]
    observation = _reseal_v4(observation)
    observation_ref = _v4_artifact_ref(
        observation, identity_field="observation_id", relative_path=observation_path
    )

    label = json.loads((root / label_path).read_text(encoding="utf-8"))
    label["label_rows"] = [
        {
            "cost_adjusted_return": str(Decimal(index + 1) / Decimal("100") - Decimal("0.002")),
            "industry_adjusted_return": str(Decimal(index + 1) / Decimal("100") - Decimal("0.005")),
            "industry_id": "banks" if index < 3 else "technology",
            "industry_return": "0.005",
            "market_adjusted_return": str(Decimal(index + 1) / Decimal("100") - Decimal("0.003")),
            "market_return": "0.003",
            "status": "AVAILABLE",
            "symbol": f"00000{index + 1}.SZ",
            "total_return": str(Decimal(index + 1) / Decimal("100")),
        }
        for index in range(5)
    ]
    label["source_lineage_sha256"] = hashlib.sha256(
        canonical_bytes(
            {
                "evidence_refs": label["evidence_refs"],
                "observation_run_ref": label["observation_run_ref"],
                "shanghai_open_sessions": label["shanghai_open_sessions"],
            }
        )
    ).hexdigest()
    label["label_id"] = (
        "forward-label-"
        + hashlib.sha256(
            canonical_bytes(
                {
                    "horizon_sessions": label["horizon_sessions"],
                    "source_lineage_sha256": label["source_lineage_sha256"],
                    "strategy_id": label["strategy_id"],
                }
            )
        ).hexdigest()
    )
    label = _reseal_v4(label)
    label_ref = _v4_artifact_ref(label, identity_field="label_id", relative_path=label_path)

    factor_inventory = json.loads((root / factor_inventory_path).read_text(encoding="utf-8"))
    factor_inventory["factors"][0]["exposure_observation_refs"] = [observation_ref]
    factor_inventory = _reseal_v4(factor_inventory)
    factor_inventory_ref = _v4_artifact_ref(
        factor_inventory,
        identity_field="inventory_id",
        relative_path=factor_inventory_path,
    )

    origin_inventory = json.loads((root / origin_inventory_path).read_text(encoding="utf-8"))
    origin_inventory["origins"][0]["canonical_evidence_ref"] = label_ref
    origin_inventory["origins"][0]["evidence_refs"] = [label_ref]
    origin_inventory = _reseal_v4(origin_inventory)
    origin_inventory_ref = _v4_artifact_ref(
        origin_inventory,
        identity_field="inventory_id",
        relative_path=origin_inventory_path,
    )

    evaluation = json.loads((root / evaluation_path).read_text(encoding="utf-8"))
    evaluation["existing_factor_inventory_ref"] = factor_inventory_ref
    evaluation["evidence_origin_inventory_ref"] = origin_inventory_ref
    evaluation["label_refs"] = [label_ref]
    evaluation = _reseal_v4(evaluation)
    evaluation_ref = _v4_artifact_ref(
        evaluation, identity_field="receipt_id", relative_path=evaluation_path
    )

    for path, document in (
        (observation_path, observation),
        (label_path, label),
        (factor_inventory_path, factor_inventory),
        (origin_inventory_path, origin_inventory),
        (evaluation_path, evaluation),
    ):
        _write_canonical(root, path, document)

    replacements = {
        factor_inventory_path: factor_inventory_ref,
        origin_inventory_path: origin_inventory_ref,
    }
    expanded_closure = [replacements.get(ref["relative_path"], ref) for ref in closure_refs]
    return (
        session_path,
        session_sha,
        observation_ref,
        label_ref,
        evaluation_ref,
        expanded_closure,
    )


def _end_to_end_request(
    root: Path,
    *,
    regime_mode: str = "missing",
) -> tuple[str, str]:
    (
        session_path,
        session_sha,
        observation_ref,
        label_ref,
        evaluation_ref,
        closure_refs,
    ) = _expanded_forward_closure(root)
    source_refs = {
        ref["artifact_id"]: ref
        for ref in closure_refs
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    }
    supporting = _evidence(source_ref=source_refs["source-a"])
    contrary = _evidence(
        direction="CONTRARY",
        name="source-b",
        source_ref=source_refs["source-b"],
    )
    hypothesis = build_hypothesis(
        thesis="Forward RankIC remains positive.",
        why_it_may_be_true="The factor has a stable economic mechanism.",
        what_would_make_it_fail="A negative RankIC would falsify the claim.",
        supporting_evidence=[supporting],
        contrary_evidence=[contrary],
        expected_window_start=AS_OF,
        expected_window_end=EVALUATED_AT,
        falsification_conditions=[
            {
                "metric_id": "rank_ic",
                "operator": "LT",
                "threshold": "0",
                "window_sessions": 1,
            }
        ],
        related_companies=["000001.SZ"],
        related_industries=["banks"],
        as_of=AS_OF,
    )
    supporting_ref = _research_input_ref(root, supporting, identity_field="evidence_id")
    contrary_ref = _research_input_ref(root, contrary, identity_field="evidence_id")
    hypothesis_ref = _research_input_ref(root, hypothesis, identity_field="hypothesis_id")
    memory_inventory = build_memory_inventory(entries=[], timestamp_value="1970-01-01T00:00:00Z")
    memory_ref = _research_input_ref(root, memory_inventory, identity_field="inventory_id")

    regime_binding = None
    if regime_mode != "missing":
        input_source = (
            _source_ref("source-outside")
            if regime_mode == "outside-closure"
            else source_refs["source-a"]
        )
        valid_regime_input = _i0_regime_input(source_ref=input_source)
        receipt_time = "2026-01-03T07:00:00Z" if regime_mode == "mistimed" else AS_OF
        regime_source_receipt = infer_multilayer_regime(
            regime_input=valid_regime_input,
            evidence=[supporting],
            as_of=receipt_time,
        )
        if regime_mode in {"input-ref-mismatch", "evidence-ref-mismatch"}:
            unsealed_receipt = deepcopy(regime_source_receipt)
            unsealed_receipt.pop("receipt_id")
            unsealed_receipt.pop("semantic_sha256")
            target = (
                unsealed_receipt["input_ref"]
                if regime_mode == "input-ref-mismatch"
                else unsealed_receipt["evidence_refs"][0]
            )
            target["artifact_id"] = "deliberate-binding-mismatch"
            regime_source_receipt = seal_content_addressed(
                unsealed_receipt,
                identity_field="receipt_id",
            )
        regime_input = valid_regime_input
        if regime_mode == "malformed":
            regime_input = seal_content_addressed(
                {
                    key: value
                    for key, value in valid_regime_input.items()
                    if key not in {"available_at", "input_id", "semantic_sha256"}
                },
                identity_field="input_id",
            )
        regime_input_ref = _research_input_ref(root, regime_input, identity_field="input_id")
        regime_receipt_ref = _research_input_ref(
            root, regime_source_receipt, identity_field="receipt_id"
        )
        regime_binding = {
            "evidence_refs": [supporting_ref],
            "industry_entity_scope": "GLOBAL_BREADTH",
            "input_ref": regime_input_ref,
            "receipt_ref": regime_receipt_ref,
            "theme_entity_scope": "GLOBAL_BREADTH",
        }

    factor_ref = json.loads((root / observation_ref["relative_path"]).read_text(encoding="utf-8"))[
        "factor_ref"
    ]
    rules = _variant_rules()
    rules.sort(key=lambda row: row["metric_id"])
    hypothesis_spec = {
        "contrary_rules": [],
        "evidence_refs": [supporting_ref, contrary_ref],
        "falsification_bindings": [
            {
                "condition_index": 0,
                "factor_id": "factor-a",
                "label_field": "total_return",
                "metric_id": "rank_ic",
                "window_end": "2026-02-27",
                "window_start": "2026-01-02",
            }
        ],
        "hypothesis_ref": hypothesis_ref,
        "min_coverage": "1",
        "min_mature_origins": 1,
        "spec_id": "hypothesis-spec-a",
        "support_rules": [
            {
                "aggregation": "MEAN",
                "factor_id": "factor-a",
                "label_field": "total_return",
                "metric_id": "rank_ic",
                "operator": "GTE",
                "threshold": "0.05",
                "window_end": "2026-02-27",
                "window_start": "2026-01-02",
            }
        ],
    }
    policy = seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "broker": False,
            "calibration_specs": [
                {
                    "evidence_id": supporting["evidence_id"],
                    "factor_id": "factor-a",
                    "metric_id": "rank_ic",
                    "min_mature_count": 1,
                    "success_operator": "GTE",
                    "success_threshold": "0.05",
                }
            ],
            "created_at": AS_OF,
            "decision_protocol": "myquant.v17.v4",
            "execution": False,
            "factor_specs": [
                {
                    "direction": "HIGHER_IS_BETTER",
                    "expected_rank_ic_sign": "POSITIVE",
                    "factor_id": "factor-a",
                    "factor_ref": factor_ref,
                }
            ],
            "horizon_sessions": 1,
            "hypothesis_specs": [hypothesis_spec],
            "min_available_origins": 1,
            "min_industry_mapping_coverage": "1",
            "min_joint_coverage": "1",
            "min_symbols": 5,
            "mainline_authority": False,
            "order": False,
            "operational_activation_unchanged": True,
            "production": False,
            "regime_policy": {
                "industry_entity_scope": "GLOBAL_BREADTH",
                "min_stratum_origins": 1,
                "theme_entity_scope": "GLOBAL_BREADTH",
            },
            "research_only": True,
            "timestamp": AS_OF,
            "trade": False,
            "variant_policy": {
                "comparison_rules": rules,
                "variants": [
                    {"required": True, "variant_id": "v17-quant-core", "variant_ref": factor_ref},
                    {
                        "required": False,
                        "variant_id": "v17-quant-plus-industry",
                        "variant_ref": factor_ref,
                    },
                    {
                        "required": False,
                        "variant_id": "v17-quant-plus-industry-theme",
                        "variant_ref": factor_ref,
                    },
                ],
            },
            "version": "myquant.v17.research-intelligence.forward-evaluation-policy.v1",
        },
        identity_field="policy_id",
    )
    origin = {
        "closure_refs": closure_refs,
        "evaluation_refs": [evaluation_ref],
        "factor_observation_bindings": [
            {"factor_id": "factor-a", "observation_ref": observation_ref}
        ],
        "label_ref": label_ref,
        "origin_id": "pending",
        "regime_binding": regime_binding,
        "session_byte_sha256": session_sha,
        "session_relative_path": session_path,
        "universe_factor_id": "factor-a",
        "universe_observation_ref": observation_ref,
        "variant_observation_bindings": [
            {"observation_ref": observation_ref, "variant_id": "v17-quant-core"},
            {"observation_ref": None, "variant_id": "v17-quant-plus-industry"},
            {"observation_ref": None, "variant_id": "v17-quant-plus-industry-theme"},
        ],
    }
    origin["origin_id"] = _origin_identity(origin)
    request_body = {
        "as_of": EVALUATED_AT,
        "authority": dict(NO_AUTHORITY),
        "broker": False,
        "decision_protocol": "myquant.v17.v4",
        "evaluated_at": EVALUATED_AT,
        "execution": False,
        "mainline_authority": False,
        "memory_inventory_ref": memory_ref,
        "operational_activation_unchanged": True,
        "order": False,
        "origins": [origin],
        "policy": policy,
        "production": False,
        "research_only": True,
        "trade": False,
        "version": "myquant.v17.research-intelligence.forward-evaluation-request.v1",
    }
    request_id = (
        "forward-evaluation-request-" + hashlib.sha256(canonical_bytes(request_body)).hexdigest()
    )
    request = seal_semantic({**request_body, "request_id": request_id})
    request_path = f"data/private/research_intelligence/evaluation_requests/{request_id}.json"
    raw = canonical_resource_bytes(request)
    _write_canonical(root, request_path, request)
    return request_path, hashlib.sha256(raw).hexdigest()


def test_exact_replay_builds_self_contained_no_authority_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha = _end_to_end_request(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    first = run_forward_research_evaluation(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    second = run_forward_research_evaluation(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    assert first == second
    assert before == {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    main = first["main_receipt"]
    assert main["version"] == ("myquant.v17.research-intelligence.forward-evaluation-receipt.v1")
    assert {
        "evaluation_id",
        "observation_refs",
        "label_refs",
        "factor_refs",
        "hypothesis_refs",
        "universe_ref",
        "evaluation_window",
        "metrics",
        "limitations",
        "implementation_sha",
    }.issubset(main)
    assert main["authority"] == NO_AUTHORITY
    assert len(main["source_evaluation_refs"]) == 1
    assert main["source_evaluation_refs"][0]["artifact_version"] == (
        "myquant.v17.v4.forward-evaluation-receipt.v1"
    )
    assert main["research_only"] is True
    assert main["production"] is False
    assert all(main[key] is False for key in ("broker", "execution", "order", "trade"))
    assert first["hypothesis_evaluations"][0]["hypothesis_status"] == "SUPPORTED"
    assert first["memory_proposal"]["proposed_entries"][-1]["status"] == "SUPPORTED"
    assert "posterior" not in _walk_keys(first)


def test_non_null_regime_binding_replays_selected_states_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha = _end_to_end_request(tmp_path, regime_mode="valid")
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    result = run_forward_research_evaluation(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    regime = result["regime_evaluation"]
    assert "SELECTED_STATES_ONLY_NO_POSTERIOR_INPUT" in regime["limitations"]
    assert "NO_BACKWARD_SMOOTHING" in regime["limitations"]
    assert "posterior" not in _walk_keys(regime)
    market = next(row for row in regime["layer_rows"] if row["layer"] == "market")
    bull = next(row for row in market["state_rows"] if row["state"] == "BULL")
    rank_ic = next(row for row in bull["factor_metric_rows"] if row["metric_id"] == "rank_ic")
    assert rank_ic["status"] == "AVAILABLE"


def test_missing_regime_binding_is_unavailable_not_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha = _end_to_end_request(tmp_path)
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    result = run_forward_research_evaluation(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    regime = result["regime_evaluation"]
    assert "MISSING_SELECTED_STATE_ORIGINS_DISCLOSED" in regime["limitations"]
    assert all(
        metric["status"] == "UNAVAILABLE"
        for layer in regime["layer_rows"]
        for state in layer["state_rows"]
        for metric in state["factor_metric_rows"]
    )


@pytest.mark.parametrize(
    "regime_mode",
    [
        "outside-closure",
        "malformed",
        "mistimed",
        "input-ref-mismatch",
        "evidence-ref-mismatch",
    ],
)
def test_invalid_supplied_regime_binding_is_stable_blocker_and_writes_nothing(
    regime_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    request_path, request_sha = _end_to_end_request(tmp_path, regime_mode=regime_mode)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    assert (
        evaluator_cli(
            [
                "research-evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                request_path,
                "--request-sha256",
                request_sha,
            ]
        )
        == 2
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["blocker_code"] == "evaluation_blocked"
    assert before == {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_cli_success_is_one_canonical_stdout_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    request_path, request_sha = _end_to_end_request(tmp_path)
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    assert (
        evaluator_cli(
            [
                "research-evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                request_path,
                "--request-sha256",
                request_sha,
            ]
        )
        == 0
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["version"] == (
        "myquant.v17.research-intelligence.forward-evaluation-envelope.v1"
    )


def test_cli_requires_exact_request_and_rejects_selector_flags(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert (
        evaluator_cli(
            [
                "research-evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                "latest",
                "--request-sha256",
                SHA_A,
                "--latest",
            ]
        )
        == 2
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["blocker_code"] == "argument_invalid"


def test_public_script_inventory_is_v17_v4_only() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"] == {
        "quant-investor": "quant_investor.cli.main:main",
        "quant-investor-v17-v4": "quant_investor.intelligence.evaluator.cli:main",
    }


def test_non_evaluator_arguments_delegate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    values = ["status"]
    observed: list[list[str]] = []

    def delegated(argv: list[str]) -> int:
        observed.append(argv)
        print("delegated-stdout")
        print("delegated-stderr", file=sys.stderr)
        return 7

    monkeypatch.setattr("quant_investor.v17_v4_runtime.cli.main", delegated)
    assert evaluator_cli(values) == 7
    captured = capfd.readouterr()
    assert observed == [["status"]]
    assert values == ["status"]
    assert captured.out == "delegated-stdout\n"
    assert captured.err == "delegated-stderr\n"


def test_cli_classifies_malformed_canonical_request_as_input_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    relative_path = (
        "data/private/research_intelligence/evaluation_requests/"
        f"forward-evaluation-request-{SHA_A}.json"
    )
    raw = b'{"version":}\n'
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    assert (
        evaluator_cli(
            [
                "research-evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request-path",
                relative_path,
                "--request-sha256",
                hashlib.sha256(raw).hexdigest(),
            ]
        )
        == 2
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["blocker_code"] == "artifact_invalid"


def test_cli_does_not_resolve_away_a_symlink_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_root = tmp_path / "workspace-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluator.forward_evaluator.verify_package",
        lambda: {"semantic_sha256": SHA_A},
    )
    assert (
        evaluator_cli(
            [
                "research-evaluate",
                "--workspace-root",
                str(symlink_root),
                "--request-path",
                (
                    "data/private/research_intelligence/evaluation_requests/"
                    f"forward-evaluation-request-{SHA_A}.json"
                ),
                "--request-sha256",
                SHA_A,
            ]
        )
        == 2
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["blocker_code"] == "artifact_invalid"


def test_shared_reader_rejects_same_path_with_conflicting_sha(tmp_path: Path) -> None:
    relative_path = "data/private/research_intelligence/evaluation_inputs/input.json"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}\n")
    actual = hashlib.sha256(b"{}\n").hexdigest()
    reader = ExactArtifactReader(str(tmp_path))
    assert reader.read(relative_path, actual) == b"{}\n"
    with pytest.raises(IntelligenceContractError, match="same path declares different"):
        reader.read(relative_path, SHA_A)


def test_new_evaluator_sources_have_no_authority_or_provider_surface() -> None:
    root = Path(__file__).parents[2] / "quant_investor" / "intelligence" / "evaluator"
    forbidden_imports = {"openai", "tushare", "yfinance", "quant_investor.providers"}
    forbidden_keys = {
        "factor_weight",
        "governance_mutation",
        "order_request",
        "posterior_update",
        "trade_request",
    }
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"import {name}" in text for name in forbidden_imports)
    result = _factor_result()
    assert not (_walk_keys(result) & forbidden_keys)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
