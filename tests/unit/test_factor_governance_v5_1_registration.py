from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

import pytest

from quant_investor.factors.governance_v5 import FactorGovernanceV5Error
from quant_investor.factors.governance_v5.authority_v5_1 import (
    build_authority_matrix_v5_1,
    validate_authority_matrix_v5_1,
)
from quant_investor.factors.governance_v5.contracts_v5_1 import (
    OWNER_POLICY_FIELDS,
    build_candidate_registration_v5_1,
    validate_candidate_registration_v5_1,
)
from quant_investor.intelligence_v2.quant_producer import validate_initial_pool

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-08-13T02:00:00Z"
OLD_V5_SHA256 = {
    "quant_investor/factors/governance_v5/__init__.py": (
        "27b81dec0e7c855dba6700b4239be2312b0bf0cf587740752629a6db75f0a939"
    ),
    "quant_investor/factors/governance_v5/_core.py": (
        "1b2dd57785ffda69ed00522b2f4b105797d69deace22132bbee25bdd6a4e533f"
    ),
    "quant_investor/factors/governance_v5/contracts.py": (
        "bd0b8d20230595785508b99d78fb87539ee1d6acff39f8ecc739622f049f6001"
    ),
    "quant_investor/factors/governance_v5/prospective.py": (
        "8980923f5441d244be597d462464d4e5730b3cd440fe931ed88fe980b4bc641a"
    ),
    "quant_investor/factors/governance_v5/weights.py": (
        "10cdcb8bb7ed93feb9a73631c63d7060f214df42a524b800748584ceff3bc83a"
    ),
}


def candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "quality_primary",
            "expression": "cs_rank(fin_roe)",
            "family": "quality",
            "input_fields": ["fin_roe"],
            "role": "PRIMARY",
        },
        {
            "candidate_id": "quality_alternate",
            "expression": "cs_rank(fin_roa)",
            "family": "quality",
            "input_fields": ["fin_roa"],
            "role": "ALTERNATE_FOR:quality_primary",
        },
    ]


def registration() -> dict:
    return build_candidate_registration_v5_1(
        registered_at=NOW,
        candidates=candidates(),
        catalog_source_sha256="a" * 64,
        implementation_source_sha256="b" * 64,
        pit_universe_sha256="c" * 64,
        exchange_calendar_sha256="d" * 64,
        missing_owner_policy_fields=OWNER_POLICY_FIELDS,
    )


def test_registration_is_replayable_and_denies_every_downstream_authority() -> None:
    document = registration()
    assert validate_candidate_registration_v5_1(document) == document
    assert document["lifecycle_state"] == "REGISTERED"
    assert document["missing_owner_policy_fields"] == list(OWNER_POLICY_FIELDS)
    for field in (
        "admission_eligible",
        "b0_eligible",
        "decision_eligible",
        "i6_eligible",
        "preregistration_valid",
        "production_active",
        "prospective_observation_authorized",
        "v17_eligible",
    ):
        assert document[field] is False
    assert all(
        value is False for key, value in document["authority"].items() if key != "research_only"
    )
    assert document["authority"]["research_only"] is True

    matrix = build_authority_matrix_v5_1(registered_at=NOW, registration=document)
    assert validate_authority_matrix_v5_1(matrix, registration=document) == matrix
    assert matrix["reachable_lifecycle_states"] == ["REGISTERED"]
    assert matrix["bayesian_posterior_available"] is False
    assert matrix["lifecycle_transition_engine_available"] is False
    assert all(row["stock_pool_builder_available"] is False for row in matrix["lanes"])


def test_registration_rejects_policy_invention_tampering_and_b0_consumption() -> None:
    with pytest.raises(FactorGovernanceV5Error, match="full owner policy gap"):
        build_candidate_registration_v5_1(
            registered_at=NOW,
            candidates=candidates(),
            catalog_source_sha256="a" * 64,
            implementation_source_sha256="b" * 64,
            pit_universe_sha256="c" * 64,
            exchange_calendar_sha256="d" * 64,
            missing_owner_policy_fields=[*OWNER_POLICY_FIELDS[:-1], "unknown_field"],
        )
    tampered = copy.deepcopy(registration())
    tampered["admission_eligible"] = True
    with pytest.raises(FactorGovernanceV5Error):
        validate_candidate_registration_v5_1(tampered)
    with pytest.raises(Exception):
        validate_initial_pool(registration())


def test_additive_v5_1_keeps_old_v5_bytes_and_production_imports_unchanged() -> None:
    for relative_path, expected in OLD_V5_SHA256.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected

    forbidden_modules = {"authority_v5_1", "contracts_v5_1", "research_pool_v5_1"}
    production_paths = [
        ROOT / "quant_investor/intelligence_v2/quant_producer.py",
        *sorted((ROOT / "quant_investor/intelligence_v2/decision_v2").glob("*.py")),
        *sorted((ROOT / "quant_investor/intelligence_v2/portfolio").glob("*.py")),
        *sorted((ROOT / "quant_investor/intelligence_v2/publication").glob("*.py")),
        ROOT / "quant_investor/v17_mainline/intelligence_v2_reader.py",
    ]
    for path in production_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            any(name in module for name in forbidden_modules) for module in imported
        ), path
