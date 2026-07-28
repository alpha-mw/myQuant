from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from quant_investor.v17_v2_contract.validators import (
    PROTOCOL_VERSION,
    validate_source_role_matrix,
)

PACKAGE_ROOT = Path(__file__).parents[2] / "quant_investor" / "v17_v2_contract"
SCHEMA_ROOT = PACKAGE_ROOT / "schemas"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"

EXPECTED_SCHEMA_IDS = {
    "action_failure_receipt.v1.schema.json": ("myquant.v17.v2.action-failure-receipt.schema.v1"),
    "dataset_record_schema_registry.v1.schema.json": (
        "myquant.v17.v2.dataset-record-schema-registry.schema.v1"
    ),
    "dataset_manifest.v1.schema.json": "myquant.v17.v2.dataset-manifest.schema.v1",
    "dataset_summary.v1.schema.json": "myquant.v17.v2.dataset-summary.schema.v1",
    "deep_research_report.v1.schema.json": ("myquant.v17.v2.deep-research-report.schema.v1"),
    "deep_research_request.v1.schema.json": ("myquant.v17.v2.deep-research-request.schema.v1"),
    "deep_research_response.v1.schema.json": ("myquant.v17.v2.deep-research-response.schema.v1"),
    "generation_catalog.v1.schema.json": ("myquant.v17.v2.generation-catalog.schema.v1"),
    "main_suite_runtime_policy.v1.schema.json": (
        "myquant.v17.v2.phase0-main-suite-runtime-policy.schema.v1"
    ),
    "macro_overlay.v1.schema.json": "myquant.v17.v2.macro-overlay.schema.v1",
    "market_pointer.v1.schema.json": "myquant.v17.v2.market-pointer.schema.v1",
    "market_snapshot_manifest.v1.schema.json": (
        "myquant.v17.v2.market-snapshot-manifest.schema.v1"
    ),
    "markov_overlay.v1.schema.json": "myquant.v17.v2.markov-overlay.schema.v1",
    "observation_disposition.v1.schema.json": ("myquant.v17.v2.observation-disposition.schema.v1"),
    "portfolio_output.v1.schema.json": "myquant.v17.v2.portfolio-output.schema.v1",
    "portfolio_required_inputs.v1.schema.json": (
        "myquant.v17.v2.portfolio-required-inputs.schema.v1"
    ),
    "rank_output.v1.schema.json": "myquant.v17.v2.rank-output.schema.v1",
    "risk_policy_snapshot.v1.schema.json": (
        "myquant.v17.v2.risk-policy-snapshot.schema.v1"
    ),
    "shadow_latest_pointer.v1.schema.json": ("myquant.v17.v2.shadow-latest-pointer.schema.v1"),
    "shadow_ledger.v1.schema.json": "myquant.v17.v2.shadow-ledger.schema.v1",
    "shadow_output.v1.schema.json": "myquant.v17.v2.shadow-output.schema.v1",
    "source_binding_set.v1.schema.json": ("myquant.v17.v2.source-binding-set.schema.v1"),
    "source_locator.v1.schema.json": "myquant.v17.v2.source-locator.schema.v1",
    "source_manifest.v1.schema.json": "myquant.v17.v2.source-manifest.schema.v1",
    "source_role_matrix.v1.schema.json": ("myquant.v17.v2.source-role-matrix.schema.v1"),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    assert type(payload) is dict
    return payload


def _walk(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    rows = [(path, value)]
    if type(value) is dict:
        for key, child in value.items():
            rows.extend(_walk(child, f"{path}.{key}"))
    elif type(value) is list:
        for index, child in enumerate(value):
            rows.extend(_walk(child, f"{path}[{index}]"))
    return rows


def test_active_v2_schema_set_and_envelopes_are_exact() -> None:
    paths = sorted(SCHEMA_ROOT.glob("*.json"))
    assert {path.name for path in paths} == set(EXPECTED_SCHEMA_IDS)
    for path in paths:
        schema = _load(path)
        expected_id = EXPECTED_SCHEMA_IDS[path.name]
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == expected_id
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["properties"]["protocol_version"] == {"const": PROTOCOL_VERSION}
        assert schema["properties"]["version"] == {
            "const": expected_id.replace(".schema.v1", ".v1")
        }
        assert {"protocol_version", "version"}.issubset(schema["required"])


def test_all_instance_arrays_declare_their_canonical_order() -> None:
    for schema_path in sorted(SCHEMA_ROOT.glob("*.json")):
        for path, node in _walk(_load(schema_path)):
            if type(node) is dict and node.get("type") == "array":
                assert (
                    "x-canonical-order" in node
                ), f"{schema_path.name}:{path} lacks x-canonical-order"
                marker = node["x-canonical-order"]
                assert type(marker) is str
                assert marker == "order-sensitive" or marker.startswith(
                    ("order-sensitive:", "canonicalized:")
                )


def test_schema_references_are_local_and_path_contracts_are_protocol_isolated() -> None:
    for schema_path in sorted(SCHEMA_ROOT.glob("*.json")):
        schema = _load(schema_path)
        for path, node in _walk(schema):
            if type(node) is not dict:
                continue
            reference = node.get("$ref")
            if reference is not None:
                assert type(reference) is str
                assert reference.startswith("#/") or (
                    schema_path.name == "shadow_output.v1.schema.json"
                    and reference
                    in {
                        "portfolio_output.v1.schema.json",
                        "rank_output.v1.schema.json",
                    }
                ), f"{schema_path.name}:{path}"
            pattern = node.get("pattern")
            if type(pattern) is str:
                re.compile(pattern)
                if "data/private/v17_sources" in pattern:
                    assert "data/private/v17_sources/protocol-v2/" in pattern
                if "results/v17_shadow" in pattern:
                    assert "results/v17_shadow/protocol-v2/" in pattern
            constant = node.get("const")
            if type(constant) is str:
                if constant.startswith("data/private/v17_sources/"):
                    assert constant.startswith("data/private/v17_sources/protocol-v2/")
                if constant.startswith("results/v17_shadow/"):
                    assert constant.startswith("results/v17_shadow/protocol-v2/")


def test_latest_pointer_is_exactly_the_protocol_v2_shadow_pointer() -> None:
    schema = _load(SCHEMA_ROOT / "shadow_latest_pointer.v1.schema.json")
    assert schema["properties"]["pointer_path"]["const"] == (
        "results/v17_shadow/protocol-v2/_latest/shadow.json"
    )
    assert (
        schema["$defs"]["artifactRef"]["properties"]["artifact_version"]["pattern"]
        == r"^myquant\.v17\.v2\.(shadow-ledger|shadow-output)\.v1$"
    )
    assert {
        "ledger_ref",
        "terminal_output_ref",
        "previous_pointer_byte_sha256",
        "publication_mode",
    }.issubset(schema["required"])
    assert {"output_ref", "mode"}.isdisjoint(schema["properties"])


def test_ledger_and_deep_report_freeze_strong_binding_shapes() -> None:
    ledger = _load(SCHEMA_ROOT / "shadow_ledger.v1.schema.json")
    assert ledger["properties"]["artifacts"]["maxItems"] == 128
    assert ledger["properties"]["history"]["maxItems"] == 64
    assert ledger["properties"]["locator_binding"]["$ref"] == ("#/$defs/locatorBinding")
    assert ledger["properties"]["input_bindings"]["items"]["$ref"] == ("#/$defs/inputBinding")
    assert ledger["properties"]["action"]["$ref"] == "#/$defs/action"
    assert ledger["properties"]["checkpoint"]["$ref"] == "#/$defs/checkpoint"
    assert {"contract_bindings", "implementation_bindings"}.issubset(ledger["required"])
    history_required = set(ledger["$defs"]["historyEntry"]["required"])
    assert {
        "attempt_id",
        "acceptance_checkpoint",
        "input_binding_sha256s",
    }.issubset(history_required)
    assert "checkpoint" not in history_required

    report = _load(SCHEMA_ROOT / "deep_research_report.v1.schema.json")
    assert {
        "request_ref",
        "run_id",
        "cutoff",
        "symbol",
        "template_resource_sha256",
        "evidence_refs",
        "coverage",
    }.issubset(report["required"])
    assert set(report["properties"]["signals"]["required"]) == {
        "financial",
        "business_model",
        "industry",
        "competitiveness",
        "management",
        "valuation",
    }
    assert report["properties"]["coverage"]["minItems"] == 16
    assert report["properties"]["layers"]["minItems"] == 5
    assert report["properties"]["severe_red_flags"]["minItems"] == 10


def test_failure_receipt_schema_cannot_represent_preinitialized_zero_write() -> None:
    schema = _load(SCHEMA_ROOT / "action_failure_receipt.v1.schema.json")
    assert schema["$defs"]["acceptanceCheckpoint"] == {"const": "INITIALIZED"}
    assert "acceptance_checkpoint" in schema["required"]
    assert "checkpoint" not in schema["properties"]
    assert set(schema["properties"]["status"]["enum"]) == {
        "UNPUBLISHED_NOT_COMMITTED",
        "POST_COMMIT_UNCERTAIN",
        "TERMINAL_UNPUBLISHED",
    }
    assert "REJECTED_ZERO_WRITE" not in schema["properties"]["status"]["enum"]


def test_symbol_and_identifier_patterns_are_closed_and_case_strict() -> None:
    request = _load(SCHEMA_ROOT / "deep_research_request.v1.schema.json")
    symbol_pattern = re.compile(request["$defs"]["symbol"]["pattern"])
    assert symbol_pattern.fullmatch("000001.SZ")
    assert symbol_pattern.fullmatch("920685.BJ")
    for invalid in ("000001", "000001.SS", "000001.sz", "600000.SH:bad"):
        assert symbol_pattern.fullmatch(invalid) is None

    identifier_pattern = re.compile(request["$defs"]["identifier"]["pattern"])
    assert identifier_pattern.fullmatch("lower-id.v1")
    assert identifier_pattern.fullmatch("Upper-ID") is None


def test_source_locator_schema_is_caller_bound_and_run_independent() -> None:
    schema = _load(SCHEMA_ROOT / "source_locator.v1.schema.json")
    assert set(schema["required"]) == {
        "protocol_version",
        "version",
        "locator_id",
        "market",
        "cutoff",
        "created_at",
        "binding_set_ref",
        "authority",
        "semantic_sha256",
    }
    assert {"run_id", "source_binding_set_ref"}.isdisjoint(schema["properties"])


def test_source_role_matrix_schema_matches_complete_phase1_resource() -> None:
    resource = _load(RESOURCE_ROOT / "source_role_matrix.v1.json")
    schema = _load(SCHEMA_ROOT / "source_role_matrix.v1.schema.json")
    assert set(resource) == set(schema["required"])
    assert resource["completeness"] == "COMPLETE"
    assert resource["runtime_usable"] is True
    assert resource["pending_registry"] == []
    assert {row["role"] for row in resource["roles"] if row["phase"] == "PORTFOLIO"} == {
        "macro_overlay",
        "markov_overlay",
        "portfolio_required_inputs",
        "risk_policy_snapshot",
    }
    assert all(row["schema_status"] == "FROZEN" for row in resource["roles"])
    assert validate_source_role_matrix(resource) == resource


def test_generation_and_binding_total_orders_include_summary_and_dataset_edges() -> None:
    catalog = _load(SCHEMA_ROOT / "generation_catalog.v1.schema.json")
    assert catalog["properties"]["table_ordering"]["const"] == (
        "stage-role-table_id-summary_path-summary_sha-dataset_path-dataset_sha"
    )
    binding_set = _load(SCHEMA_ROOT / "source_binding_set.v1.schema.json")
    ordering = binding_set["properties"]["binding_ordering"]["const"]
    assert "summary_path-summary_sha" in ordering
    assert "dataset_path-dataset_sha" in ordering
    assert "disposition_id" in ordering


def test_source_object_schema_patterns_reject_bin_suffix_consistently() -> None:
    digest = "a" * 64
    base = f"data/private/v17_sources/protocol-v2/objects/aa/{digest}"
    checked = []
    for schema_path in sorted(SCHEMA_ROOT.glob("*.json")):
        schema = _load(schema_path)
        artifact_ref = schema.get("$defs", {}).get("artifactRef", {})
        relative_path = artifact_ref.get("properties", {}).get("relative_path", {})
        raw_pattern = relative_path.get("pattern")
        if (
            type(raw_pattern) is not str
            or "data/private/v17_sources/protocol-v2/" not in raw_pattern
            or "objects/" not in raw_pattern
        ):
            continue
        checked.append(schema_path.name)
        pattern = re.compile(raw_pattern)
        assert pattern.fullmatch(f"{base}.json")
        if "blob" in raw_pattern:
            assert pattern.fullmatch(f"{base}.blob")
        if "parquet" in raw_pattern:
            assert pattern.fullmatch(f"{base}.parquet")
        assert pattern.fullmatch(f"{base}.bin") is None
    assert checked == [
        "dataset_summary.v1.schema.json",
        "deep_research_report.v1.schema.json",
        "deep_research_request.v1.schema.json",
        "generation_catalog.v1.schema.json",
        "market_pointer.v1.schema.json",
        "portfolio_output.v1.schema.json",
        "shadow_ledger.v1.schema.json",
        "source_binding_set.v1.schema.json",
        "source_manifest.v1.schema.json",
    ]
