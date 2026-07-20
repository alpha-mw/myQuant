from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import governance_discovery_v4_1 as discovery
from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter
from quant_investor.factors import governance_formal_catalog_bundle_v4_1 as bundle
from quant_investor.factors import (
    governance_formal_catalog_materialization_v4_1 as materializer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = REPO_ROOT / (
    "reports/factor_governance/private/v4_1_cycle/"
    "factor_v4_1_discovery_20260718T170345Z"
)
BASE_ROOT = REPO_ROOT / (
    "reports/factor_governance/private/v4_pre_admission/"
    "factor_v4_pre_admission_20260718_083224"
)
SYNTHETIC_PROTECTED_BINDINGS = {
    f"/synthetic/myQuant{suffix}": hashlib.sha256(suffix.encode("utf-8")).hexdigest()
    for suffix in bundle.PROTECTED_CONTROL_PATH_SUFFIXES
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_bindings(
    artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for filename in bundle.FORMAL_CATALOG_INPUT_FILENAMES:
        raw = bundle.canonical_file_bytes_v4_1(artifacts[filename])
        rows.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    return rows


@pytest.fixture(scope="module")
def real_bundle() -> dict[str, Any]:
    required = [
        *(DISCOVERY_ROOT / filename for filename in discovery.CANONICAL_ARTIFACT_FILENAMES),
        BASE_ROOT / "primitive_ontology.v4.json",
        BASE_ROOT / "candidate_catalog.v4.json",
        *(REPO_ROOT / suffix for suffix in materializer.REQUIRED_CODE_BINDING_SUFFIXES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"real formal-catalog fixture is unavailable: {missing}")

    discovery_values = {
        filename: _read_json(DISCOVERY_ROOT / filename)
        for filename in discovery.CANONICAL_ARTIFACT_FILENAMES
    }
    base_ontology = _read_json(BASE_ROOT / "primitive_ontology.v4.json")
    base_catalog = _read_json(BASE_ROOT / "candidate_catalog.v4.json")
    source_bindings = materializer.build_formal_catalog_source_bindings_v4_1(
        discovery_values=discovery_values,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    code_bindings = []
    for suffix in materializer.REQUIRED_CODE_BINDING_SUFFIXES:
        path = REPO_ROOT / suffix
        raw = path.read_bytes()
        code_bindings.append(
            {
                "absolute_path": str(path),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    materialization_inputs = {
        "discovery_values": discovery_values,
        "base_ontology": base_ontology,
        "base_catalog": base_catalog,
        "source_bindings": source_bindings,
        "code_bindings": code_bindings,
    }
    draft = materializer.build_formal_catalog_materialization_v4_1(
        **materialization_inputs,
        adapter_validation=None,
    )
    adapter_validation = adapter.build_formal_catalog_adapter_validation_v4_1(
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=draft[materializer.FORMAL_ONTOLOGY_FILENAME],
        catalog=draft[materializer.FORMAL_CATALOG_FILENAME],
        mapping_proof=draft[materializer.PRIMITIVE_MAPPING_PROOF_FILENAME],
    )
    materialized = materializer.build_formal_catalog_materialization_v4_1(
        **materialization_inputs,
        adapter_validation=adapter_validation,
    )
    artifacts = {
        **materialized,
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME: adapter_validation,
    }
    raw_bindings = _artifact_bindings(artifacts)
    report = bundle.build_formal_catalog_readback_report_v4_1(
        run_id="formal-catalog-bundle-test",
        artifacts=artifacts,
        artifact_bindings=raw_bindings,
        protected_bindings=SYNTHETIC_PROTECTED_BINDINGS,
    )
    values = {
        **artifacts,
        bundle.FORMAL_CATALOG_READBACK_REPORT_FILENAME: report,
    }
    return {
        "materialization_inputs": materialization_inputs,
        "artifacts": artifacts,
        "raw_bindings": raw_bindings,
        "protected_bindings": SYNTHETIC_PROTECTED_BINDINGS,
        "report": report,
        "values": values,
    }


def test_real_bundle_is_exact_seven_and_recomputes_all_authority_flags(
    real_bundle: dict[str, Any],
) -> None:
    values = real_bundle["values"]
    normalized = bundle.validate_formal_catalog_bundle_values_v4_1(
        values,
        **real_bundle["materialization_inputs"],
        protected_bindings=real_bundle["protected_bindings"],
    )

    assert tuple(values) == bundle.FORMAL_CATALOG_BUNDLE_FILENAMES
    assert len(values) == 7
    assert normalized == values

    report = normalized[bundle.FORMAL_CATALOG_READBACK_REPORT_FILENAME]
    assert report["report_semantic_sha256"] == bundle.semantic_sha256_v4_1(
        report,
        exclude_fields=("report_semantic_sha256",),
    )
    assert report["source_accounting"] == {
        "source_candidate_count": 100,
        "new_candidate_count": 37,
        "structural_alias_count": 6,
        "incompatible_count": 57,
    }
    assert report["catalog_accounting"] == {
        "base_candidate_count": 230,
        "new_candidate_count": 37,
        "candidate_count": 267,
    }
    assert report["ontology_accounting"] == {
        "base_primitive_count": 13,
        "new_primitive_count": 5,
        "primitive_count": 18,
    }
    assert {
        field: report[field]
        for field in (
            "classification_only",
            "runtime_equivalence_verified",
            "signal_computability_proven",
            "screening_eligible",
            "proposal_eligible",
            "registry_entry_created",
            "initial_weight_policy",
            "qualification",
            "formal_admission_authority",
            "production_apply_enabled",
            "new_risk_authorized",
            "source_authenticity_recomputed_by_materializer",
            "adapter_source_authenticity_recomputed",
            "protected_controls_bound_at_build_and_precommit",
            "postcommit_protected_stability_part_of_bundle_acceptance",
            "protected_stability_scope",
        )
    } == {
        "classification_only": True,
        "runtime_equivalence_verified": False,
        "signal_computability_proven": False,
        "screening_eligible": False,
        "proposal_eligible": False,
        "registry_entry_created": False,
        "initial_weight_policy": "zero_only",
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "new_risk_authorized": False,
        "source_authenticity_recomputed_by_materializer": True,
        "adapter_source_authenticity_recomputed": False,
        "protected_controls_bound_at_build_and_precommit": True,
        "postcommit_protected_stability_part_of_bundle_acceptance": False,
        "protected_stability_scope": (
            "build_and_precommit_only_external_controls_are_not_locked"
        ),
    }
    assert report["measurement_status"] == {
        field: "not_run" for field in bundle.MEASUREMENT_STATUS_FIELDS
    }
    assert report["blockers"] == list(bundle.BLOCKERS)
    true_side_effects = {
        field for field, value in report["side_effects"].items() if value is True
    }
    assert true_side_effects == {
        "filesystem_input_read_performed",
        "private_readback_report_created",
        "private_research_bundle_created",
    }


def test_report_binds_exact_canonical_bytes_semantics_and_private_mode(
    real_bundle: dict[str, Any],
) -> None:
    artifacts = real_bundle["artifacts"]
    rows = real_bundle["report"]["artifact_bindings"]
    semantic_field_by_filename = {
        materializer.PRIMITIVE_MAPPING_POLICY_FILENAME: "policy_semantic_sha256",
        materializer.PRIMITIVE_MAPPING_PROOF_FILENAME: "proof_semantic_sha256",
        materializer.FORMAL_ONTOLOGY_FILENAME: "semantic_sha256",
        materializer.FORMAL_CATALOG_FILENAME: "semantic_sha256",
        materializer.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME: (
            "manifest_semantic_sha256"
        ),
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME: (
            "validation_semantic_sha256"
        ),
    }

    assert [row["filename"] for row in rows] == list(
        bundle.FORMAL_CATALOG_INPUT_FILENAMES
    )
    for row in rows:
        raw = bundle.canonical_file_bytes_v4_1(artifacts[row["filename"]])
        assert row["byte_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["size_bytes"] == len(raw)
        assert row["mode"] == 0o600
        assert row["nlink"] == 1
        assert row["semantic_sha256"] == artifacts[row["filename"]][
            semantic_field_by_filename[row["filename"]]
        ]

    protected_rows = real_bundle["report"]["protected_bindings"]
    assert protected_rows == [
        {
            "absolute_path": path,
            "byte_sha256": real_bundle["protected_bindings"][path],
        }
        for path in sorted(real_bundle["protected_bindings"])
    ]
    assert len(protected_rows) == 5
    assert real_bundle["report"]["protected_bindings_semantic_sha256"] == (
        bundle.semantic_sha256_v4_1(protected_rows)
    )

    for field, replacement, error in (
        ("byte_sha256", "0" * 64, "byte SHA mismatch"),
        ("mode", 0o644, "not owner-private"),
        ("nlink", 2, "not owner-private"),
    ):
        bad = copy.deepcopy(real_bundle["raw_bindings"])
        bad[0][field] = replacement
        with pytest.raises(
            bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
            match=error,
        ):
            bundle.build_formal_catalog_readback_report_v4_1(
                run_id="formal-catalog-bundle-test",
                artifacts=artifacts,
                artifact_bindings=bad,
                protected_bindings=real_bundle["protected_bindings"],
            )


def test_unknown_fields_and_noncanonical_inventory_fail_closed(
    real_bundle: dict[str, Any],
) -> None:
    report = copy.deepcopy(real_bundle["report"])
    report["unexpected_authority"] = True
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="fields mismatch",
    ):
        bundle.validate_formal_catalog_readback_report_v4_1(
            report,
            artifacts=real_bundle["artifacts"],
            protected_bindings=real_bundle["protected_bindings"],
        )

    values = copy.deepcopy(real_bundle["values"])
    values["unexpected.json"] = {}
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="exactly seven canonical artifacts",
    ):
        bundle.validate_formal_catalog_bundle_values_v4_1(
            values,
            **real_bundle["materialization_inputs"],
            protected_bindings=real_bundle["protected_bindings"],
        )

    bindings = copy.deepcopy(real_bundle["raw_bindings"])
    bindings[0]["unexpected"] = False
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="fields mismatch",
    ):
        bundle.build_formal_catalog_readback_report_v4_1(
            run_id="formal-catalog-bundle-test",
            artifacts=real_bundle["artifacts"],
            artifact_bindings=bindings,
            protected_bindings=real_bundle["protected_bindings"],
        )

    missing_protected = dict(real_bundle["protected_bindings"])
    missing_protected.pop(next(iter(missing_protected)))
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="exact five control files",
    ):
        bundle.build_formal_catalog_readback_report_v4_1(
            run_id="formal-catalog-bundle-test",
            artifacts=real_bundle["artifacts"],
            artifact_bindings=real_bundle["raw_bindings"],
            protected_bindings=missing_protected,
        )

    substituted_protected = dict(real_bundle["protected_bindings"])
    substituted_protected.pop(next(iter(substituted_protected)))
    substituted_protected["/synthetic/myQuant/data/parquet/cn/other.json"] = (
        "7" * 64
    )
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="outside the exact allowlist",
    ):
        bundle.build_formal_catalog_readback_report_v4_1(
            run_id="formal-catalog-bundle-test",
            artifacts=real_bundle["artifacts"],
            artifact_bindings=real_bundle["raw_bindings"],
            protected_bindings=substituted_protected,
        )


def test_private_contract_closes_over_exact_six_inputs_and_seven_outputs(
    real_bundle: dict[str, Any],
) -> None:
    contract = bundle.build_formal_catalog_bundle_contract_v4_1(
        expected_artifacts=real_bundle["artifacts"],
        **real_bundle["materialization_inputs"],
        protected_bindings=real_bundle["protected_bindings"],
    )
    assert contract.input_filenames == bundle.FORMAL_CATALOG_INPUT_FILENAMES
    assert contract.canonical_filenames == bundle.FORMAL_CATALOG_BUNDLE_FILENAMES
    assert contract.root_suffix == bundle.FORMAL_CATALOG_PRIVATE_ROOT_SUFFIX

    for filename in bundle.FORMAL_CATALOG_INPUT_FILENAMES:
        assert contract.validate_artifact(
            filename,
            real_bundle["artifacts"][filename],
        ) == real_bundle["artifacts"][filename]

    first = bundle.FORMAL_CATALOG_INPUT_FILENAMES[0]
    changed = copy.deepcopy(real_bundle["artifacts"][first])
    changed["unknown"] = True
    with pytest.raises(
        bundle.FactorGovernanceFormalCatalogBundleV4_1Error,
        match="differs from expected recomputation",
    ):
        contract.validate_artifact(first, changed)

    rebuilt_report = contract.build_readback_report(
        run_id="formal-catalog-bundle-test",
        artifacts=real_bundle["artifacts"],
        artifact_bindings=real_bundle["raw_bindings"],
    )
    assert rebuilt_report == real_bundle["report"]
