from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_aquant_input_resolution_v4_1 as subject
from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator


EXPRESSIONS = {
    "alpha_growth_quality_profit_roa": "cs_rank(cs_rank(fin_net_profit_yoy) + cs_rank(fin_roa))",
    "alpha_quality_low_debt_assets": "cs_rank(-fin_debt_to_assets)",
    "alpha_quality_value_cash_fcf": "cs_rank(cs_rank(fin_ocf_to_profit) + cs_rank(fcf_to_price))",
    "alpha_turnover_low_20d": "cs_rank(-ts_mean(turnover_rate, 20))",
    "alpha_turnover_low_60d": "cs_rank(-ts_mean(turnover_rate, 60))",
    "alpha_vwap_cash_quality_160": "cs_rank(cs_rank(ts_mean((vwap - close) / close, 160)) + cs_rank(fin_ocf_to_profit))",
    "alpha_vwap_growth_profit_160": "cs_rank(cs_rank(ts_mean((vwap - close) / close, 160)) + cs_rank(fin_net_profit_yoy))",
    "alpha_vwap_low_debt_160": "cs_rank(cs_rank(ts_mean((vwap - close) / close, 160)) + cs_rank(-fin_debt_to_assets))",
    "alpha_vwap_quality_roa_160": "cs_rank(cs_rank(ts_mean((vwap - close) / close, 160)) + cs_rank(fin_roa))",
    "alpha_vwap_quality_roe_160": "cs_rank(cs_rank(ts_mean((vwap - close) / close, 160)) + cs_rank(fin_roe))",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reseed(payload: dict[str, object], field: str) -> dict[str, object]:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    value[field] = subject.semantic_sha256_v4_1(value)
    return value


def _fixture() -> dict[str, object]:
    dates = pd.date_range("2026-07-13", periods=5, freq="D")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    mask = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    mask.iloc[0, 2] = False
    base = pd.DataFrame(
        np.arange(1, 16, dtype=float).reshape(5, 3),
        index=dates,
        columns=symbols,
    )
    turnover = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)
    turnover.iloc[1, 0] = 1.2
    turnover.iloc[3, 2] = 0.0
    matrices = {
        "close": base + 10.0,
        "vwap": base + 10.5,
        "turnover_rate": turnover,
        "fin_roe": base / 100.0,
        "fin_roa": base / 120.0,
        "fin_debt_to_assets": base / 30.0,
        "fin_net_profit_yoy": (base - 8.0) / 10.0,
        "fin_ocf_to_profit": (base - 4.0) / 7.0,
        "fcf_to_price": (base - 9.0) / 100.0,
    }
    ideas = []
    for name in subject.RESOLVED_CANDIDATE_NAMES:
        expected = subject.EXPECTED_CANDIDATES[name]
        expression = EXPRESSIONS[name]
        ideas.append(
            {
                "candidate_id": f"aquant:{evaluator.PINNED_COMMIT}:{name}",
                "name": name,
                "expression": expression,
                "normalized_expression_ast": evaluator.normalize_expression_ast_v4_1(
                    expression
                ),
                "input_fields": list(expected["input_fields"]),
                "source_definition_sha256": expected[
                    "source_definition_sha256"
                ],
                "full_candidate_normalized_ast_sha256": expected[
                    "normalized_ast_sha256"
                ],
                "catalog_definition_sha256": expected[
                    "catalog_definition_sha256"
                ],
                "mapping_semantic_sha256": expected[
                    "mapping_semantic_sha256"
                ],
            }
        )
    table_inventory = [
        {
            "relative_path": "2026/part.parquet",
            "byte_sha256": _digest("table"),
            "size_bytes": 12,
            "dataset_member": True,
        }
    ]
    serving_inventory = [
        {
            "relative_path": "2026/part.parquet",
            "byte_sha256": _digest("serving"),
            "size_bytes": 14,
            "dataset_member": True,
        }
    ]
    predecessor = [
        {
            "binding_id": binding_id,
            "path": f"/private/{binding_id}.json",
            "byte_sha256": _digest(f"predecessor-byte-{binding_id}"),
            "semantic_sha256": _digest(f"predecessor-semantic-{binding_id}"),
        }
        for binding_id in subject.PREDECESSOR_BINDING_IDS
    ]
    code = [
        {
            "binding_id": binding_id,
            "path": f"/repo/{binding_id}.py",
            "byte_sha256": _digest(f"code-byte-{binding_id}"),
            "semantic_sha256": _digest(f"code-semantic-{binding_id}"),
        }
        for binding_id in subject.CODE_BINDING_IDS
    ]
    source = []
    for binding_id in subject.SOURCE_BINDING_IDS:
        semantic = _digest(f"source-semantic-{binding_id}")
        if binding_id == "table_inventory":
            semantic = subject.semantic_sha256_v4_1(table_inventory)
        elif binding_id == "serving_inventory":
            semantic = subject.semantic_sha256_v4_1(serving_inventory)
        source.append(
            {
                "binding_id": binding_id,
                "path": f"/data/{binding_id}",
                "byte_sha256": _digest(f"source-byte-{binding_id}"),
                "semantic_sha256": semantic,
            }
        )
    protected = [
        {
            "binding_id": row["binding_id"],
            "path": row["path"],
            "before_sha256": row["byte_sha256"],
            "after_sha256": row["byte_sha256"],
        }
        for row in source
    ]
    return {
        "cycle_id": subject.EXPECTED_CYCLE_ID,
        "predecessor_bindings": predecessor,
        "code_bindings": code,
        "source_bindings": source,
        "table_root": "/data/table",
        "table_inventory": table_inventory,
        "serving_root": "/data/serving",
        "serving_inventory": serving_inventory,
        "snapshot_id": "snapshot-fixture",
        "fundamental_generation_id": "fundamental-fixture",
        "eligibility_mask": mask,
        "field_matrices": matrices,
        "bound_ideas": ideas,
        "protected_stability": protected,
    }


def _artifact() -> dict[str, object]:
    return subject.build_input_resolution_artifact_v4_1(**_fixture())


def test_builds_exact_no_label_fully_resolved_artifact() -> None:
    artifact = _artifact()

    assert artifact["schema_version"] == subject.SCHEMA_VERSION
    assert artifact["semantics_mode"] == subject.SEMANTICS_MODE
    assert artifact["exact_aquant_data_equivalence_claimed"] is False
    assert artifact["resolution_profile"] == "fully_resolved"
    assert artifact["resolution_accounting"] == {
        "predeclared_candidate_count": 10,
        "turnover_resolved_count": 2,
        "fundamental_resolved_count": 8,
        "unresolved_count": 0,
    }
    assert [row["name"] for row in artifact["candidate_rows"]] == list(
        subject.RESOLVED_CANDIDATE_NAMES
    )
    assert all(value is False for value in artifact["authority"].values())
    assert all(value is False for value in artifact["side_effects"].values())


def test_sparse_turnover_is_preserved_without_fallback_or_fill() -> None:
    artifact = _artifact()
    turnover = next(
        row for row in artifact["field_rows"] if row["field_name"] == "turnover_rate"
    )

    assert turnover["finite_count"] == 2
    assert turnover["fallback_policy"] == "none_table_forbidden_no_forward_fill"
    assert turnover["missing_policy"] == (
        "sparse_values_preserved_no_fill_ts_mean_min_periods_one"
    )


def test_rejects_field_axis_drift() -> None:
    fixture = _fixture()
    frame = fixture["field_matrices"]["fin_roe"]
    fixture["field_matrices"]["fin_roe"] = frame.rename(
        columns={"000001.SZ": "000003.SZ"}
    )

    with pytest.raises(subject.FactorGovernanceAquantInputResolutionV4_1Error):
        subject.build_input_resolution_artifact_v4_1(**fixture)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value.__setitem__("semantics_mode", "aquant_exact_semantics"),
            "identity/authority",
        ),
        (
            lambda value: value["field_rows"][3].__setitem__(
                "availability_policy", "announcement_date_ignored"
            ),
            "policy drift",
        ),
        (
            lambda value: value["field_rows"][3].__setitem__(
                "fallback_policy", "legacy_allowed"
            ),
            "policy drift",
        ),
        (
            lambda value: value["authority"].__setitem__(
                "new_risk_authorized", True
            ),
            "identity/authority",
        ),
        (
            lambda value: value["side_effects"].__setitem__("label", True),
            "identity/authority",
        ),
    ],
)
def test_rejects_semantics_policy_authority_and_side_effect_tampering(
    mutator, match: str
) -> None:
    artifact = _artifact()
    mutator(artifact)
    artifact = _reseed(artifact, "resolution_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error, match=match
    ):
        subject.validate_input_resolution_artifact_v4_1(artifact)


def test_rejects_source_hash_substitution_even_when_resealed() -> None:
    artifact = _artifact()
    artifact["source_bindings"][0]["byte_sha256"] = _digest("substitution")
    artifact = _reseed(artifact, "resolution_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error,
        match="protected stability/source mismatch",
    ):
        subject.validate_input_resolution_artifact_v4_1(artifact)


def test_rejects_relative_binding_path_even_when_protected_row_matches() -> None:
    artifact = _artifact()
    artifact["source_bindings"][0]["path"] = "relative/fundamental_daily"
    artifact["protected_stability"][0]["path"] = "relative/fundamental_daily"
    artifact = _reseed(artifact, "resolution_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error,
        match="absolute normalized path",
    ):
        subject.validate_input_resolution_artifact_v4_1(artifact)


def test_readback_rejects_forged_binding_metadata() -> None:
    artifact = _artifact()
    raw = subject.canonical_file_bytes_v4_1(artifact)
    valid_binding = {
        "filename": subject.ARTIFACT_FILENAME,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode": 0o600,
        "uid": os.getuid(),
        "nlink": 1,
    }
    report = subject.build_readback_report_v4_1(
        run_id="fixture_run",
        artifacts={subject.ARTIFACT_FILENAME: artifact},
        artifact_bindings=[valid_binding],
    )
    report["artifact_bindings"][0]["mode"] = 0o644
    report = _reseed(report, "readback_semantic_sha256")

    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error,
        match="binding identity mismatch",
    ):
        subject.validate_readback_report_v4_1(report, artifact=artifact)


def test_explicit_bundle_reader_binds_canonical_bytes_and_semantic_sha(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    path = tmp_path / subject.ARTIFACT_FILENAME
    path.write_bytes(subject.canonical_file_bytes_v4_1(artifact))
    path.chmod(0o600)
    byte_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    result = subject.validate_input_resolution_bundle_v4_1(
        artifact_path=path,
        expected_artifact_sha256=byte_sha,
        expected_semantic_sha256=artifact["resolution_semantic_sha256"],
    )

    assert result["artifact"]["resolution_profile"] == "fully_resolved"
    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error,
        match="byte SHA mismatch",
    ):
        subject.validate_input_resolution_bundle_v4_1(
            artifact_path=path,
            expected_artifact_sha256=_digest("wrong"),
            expected_semantic_sha256=artifact["resolution_semantic_sha256"],
        )


def test_explicit_bundle_reader_rejects_non_private_mode(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / subject.ARTIFACT_FILENAME
    path.write_bytes(subject.canonical_file_bytes_v4_1(artifact))
    path.chmod(0o644)

    with pytest.raises(
        subject.FactorGovernanceAquantInputResolutionV4_1Error,
        match="owner-private 0600",
    ):
        subject.validate_input_resolution_bundle_v4_1(
            artifact_path=path,
            expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_semantic_sha256=artifact["resolution_semantic_sha256"],
        )
