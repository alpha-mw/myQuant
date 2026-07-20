from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as subject


def _frame(values: list[list[float]], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.date_range("2026-01-05", periods=len(values), freq="D"),
        columns=columns or ["000001.SZ", "000002.SZ"],
        dtype=float,
    )


def _seal(value: dict, field: str) -> dict:
    payload = copy.deepcopy(value)
    payload[field] = subject.semantic_sha256_v4_1(payload)
    return payload


def _synthetic_exact_37() -> tuple[dict, dict, dict, dict]:
    receipt = _seal(
        {
            "schema_version": "factor-governance-aquant-source-receipt.v4.1",
            "protocol_version": "v4",
            "source_system": "A_quant",
            "pinned_commit": subject.PINNED_COMMIT,
            "object_type": "commit",
            "healthy": True,
            "generator_path": subject.PINNED_GENERATOR_PATH,
            "generator_function": subject.PINNED_GENERATOR_FUNCTION,
            "generator_candidate_count": 100,
            "source_files": [
                {"path": path, "raw_sha256": digest}
                for path, digest in sorted(subject.PINNED_SOURCE_FILES.items())
            ],
        },
        "receipt_semantic_sha256",
    )
    expression = "cs_rank(ts_mean(close, 1))"
    tree = subject.normalize_expression_ast_v4_1(expression)
    ast_sha = subject.semantic_sha256_v4_1(tree)
    ideas: list[dict] = []
    mappings: list[dict] = []
    candidates: list[dict] = []
    for index in range(37):
        name = f"alpha_synthetic_{index:02d}"
        candidate_id = f"aquant:{subject.PINNED_COMMIT}:{name}"
        idea = {
            "source_index": index,
            "candidate_id": candidate_id,
            "name": name,
            "expression": expression,
            "factor_type": "alpha",
            "source_family": "synthetic",
            "rationale": f"synthetic {index}",
            "direction": 1.0,
            "direction_origin": "expression_signed_ast",
            "compatibility_status": "compatible",
            "catalog_role": "new_candidate",
            "selected": True,
            "normalized_expression_ast": tree,
            "input_fields": ["close"],
            "lookback": 1,
        }
        source_sha = subject.semantic_sha256_v4_1(
            {
                "version": "aquant-source-definition.v1",
                "pinned_commit": subject.PINNED_COMMIT,
                "name": name,
                "expression": expression,
                "factor_type": "alpha",
                "source_family": "synthetic",
                "rationale": f"synthetic {index}",
                "direction": 1.0,
                "direction_origin": "expression_signed_ast",
            }
        )
        candidate = {
            "name": name,
            "implementation": "aquant_expression_ast.v1",
            "expression": expression,
            "direction": 1.0,
            "params": {},
            "lookback": 1,
            "slot": "primitive:close",
            "input_fields": ["close"],
            "primitive_ids": ["close"],
            "family": "price",
        }
        candidate["definition_sha256"] = subject.semantic_sha256_v4_1(candidate)
        mapping = _seal(
            {
                "candidate_id": candidate_id,
                "name": name,
                "source_definition_sha256": source_sha,
                "catalog_definition_sha256": candidate["definition_sha256"],
                "implementation": "aquant_expression_ast.v1",
                "expression": expression,
                "full_candidate_normalized_ast_sha256": ast_sha,
                "input_fields": ["close"],
                "primitive_ids": ["close"],
                "family": "price",
                "slot": "primitive:close",
                "mapping_status": "complete_unique_occurrence_accounting",
            },
            "mapping_semantic_sha256",
        )
        ideas.append(idea)
        mappings.append(mapping)
        candidates.append(candidate)
    for index in range(37, 100):
        ideas.append(
            {
                "source_index": index,
                "candidate_id": f"unused:{index}",
                "name": f"unused_{index}",
                "compatibility_status": "incompatible",
                "catalog_role": "incompatible",
                "selected": False,
            }
        )
    for index in range(230):
        candidate = {
            "name": f"base_{index:03d}",
            "implementation": "base.v1",
            "expression": "close",
            "direction": 1.0,
            "params": {},
            "lookback": 1,
            "slot": f"base:{index:03d}",
            "input_fields": ["close"],
            "primitive_ids": ["close"],
            "family": "price",
        }
        candidate["definition_sha256"] = subject.semantic_sha256_v4_1(candidate)
        candidates.append(candidate)
    audit = _seal(
        {
            "schema_version": "factor-governance-source-idea-audit.v4.1",
            "source_receipt_sha256": receipt["receipt_semantic_sha256"],
            "ideas": ideas,
        },
        "audit_semantic_sha256",
    )
    proof = _seal(
        {
            "schema_version": "factor-governance-primitive-mapping-proof.v4.1",
            "source_idea_audit_sha256": audit["audit_semantic_sha256"],
            "source_candidate_count": 100,
            "new_candidate_count": 37,
            "structural_alias_count": 6,
            "incompatible_count": 57,
            "new_candidate_mappings": mappings,
        },
        "proof_semantic_sha256",
    )
    catalog = _seal(
        {"schema_version": "factor-candidate-catalog.v4", "candidates": candidates},
        "semantic_sha256",
    )
    return receipt, audit, proof, catalog


def test_exact_ast_whitelist_and_hostile_syntax() -> None:
    assert subject.normalize_expression_ast_v4_1("cs_rank(-ts_mean(close, 20))")
    for expression in (
        "+close",
        "close ** 2",
        "close.shift(-1)",
        "ts_sum(close, 2)",
        "ts_mean(close, 0)",
        "ts_mean(close, 201)",
        "ts_mean(close, True)",
        "unknown",
        "(lambda x: x)(close)",
    ):
        with pytest.raises(subject.FactorGovernanceAquantNoLabelEvalV4_1Error):
            subject.normalize_expression_ast_v4_1(expression)


def test_pinned_operator_semantics_warmup_nan_rank_and_native_division() -> None:
    close = _frame([[1.0, 2.0], [3.0, np.nan]])
    amount = _frame([[1.0, 0.0], [0.0, 1.0]])
    mask = close.notna() | close.isna()
    ranked = subject.evaluate_expression_v4_1(
        expression="cs_rank(ts_mean(close, 2))",
        matrices={"close": close},
        eligibility_mask=mask,
    )
    assert ranked.iloc[0].tolist() == [0.5, 1.0]
    assert ranked.iloc[1].tolist() == [0.75, 0.75]
    divided = subject.evaluate_expression_v4_1(
        expression="amount / close",
        matrices={"amount": amount, "close": close.fillna(0.0)},
        eligibility_mask=mask,
    )
    assert np.isposinf(divided.iloc[1, 1])
    assert divided.iloc[0, 1] == 0.0


def test_pit_mask_is_reapplied_after_rolling_and_cross_sectional_rank() -> None:
    close = _frame([[10.0, 20.0], [1000.0, 30.0], [2000.0, 40.0]])
    mask = pd.DataFrame(
        [[True, True], [False, True], [False, True]],
        index=close.index,
        columns=close.columns,
        dtype=bool,
    )
    result = subject.evaluate_expression_v4_1(
        expression="cs_rank(ts_mean(close, 3))",
        matrices={"close": close.where(mask)},
        eligibility_mask=mask,
    )
    assert result.iloc[1:, 0].isna().all()
    assert result.iloc[1:, 1].tolist() == [1.0, 1.0]


def test_exact_axes_are_required() -> None:
    close = _frame([[1.0, 2.0]])
    mask = pd.DataFrame(True, index=close.index, columns=list(reversed(close.columns)))
    with pytest.raises(subject.FactorGovernanceAquantNoLabelEvalV4_1Error, match="axes"):
        subject.evaluate_expression_v4_1(
            expression="close", matrices={"close": close}, eligibility_mask=mask
        )


def test_matrix_hash_is_copy_order_and_nan_payload_deterministic() -> None:
    first = _frame([[np.nan, np.inf], [-np.inf, 1.0]])
    second = first.copy(deep=True)
    payload = second.to_numpy(copy=True).view(np.uint64)
    payload[0, 0] = np.uint64(0x7FF8000000000001)
    second.iloc[:, :] = payload.view(np.float64)
    assert subject.matrix_sha256_v4_1(first) == subject.matrix_sha256_v4_1(second)
    assert subject.matrix_sha256_v4_1(first) != subject.matrix_sha256_v4_1(
        first.iloc[:, ::-1]
    )
    sign_changed = first.copy()
    sign_changed.iloc[0, 1] = -np.inf
    assert subject.matrix_sha256_v4_1(first) != subject.matrix_sha256_v4_1(
        sign_changed
    )


def test_exact_37_bindings_and_source_or_ast_hash_drift_fail_closed() -> None:
    receipt, audit, proof, catalog = _synthetic_exact_37()
    rows = subject.bind_pinned_source_ideas_v4_1(
        source_receipt=receipt,
        source_idea_audit=audit,
        primitive_mapping_proof=proof,
        formal_catalog=catalog,
    )
    assert len(rows) == 37
    assert all(row["initial_weight"] == 0.0 for row in rows)
    drift = copy.deepcopy(proof)
    drift.pop("proof_semantic_sha256")
    row = drift["new_candidate_mappings"][0]
    row.pop("mapping_semantic_sha256")
    row["full_candidate_normalized_ast_sha256"] = "0" * 64
    row["mapping_semantic_sha256"] = subject.semantic_sha256_v4_1(row)
    drift["proof_semantic_sha256"] = subject.semantic_sha256_v4_1(drift)
    with pytest.raises(subject.FactorGovernanceAquantNoLabelEvalV4_1Error, match="drift"):
        subject.bind_pinned_source_ideas_v4_1(
            source_receipt=receipt,
            source_idea_audit=audit,
            primitive_mapping_proof=drift,
            formal_catalog=catalog,
        )
