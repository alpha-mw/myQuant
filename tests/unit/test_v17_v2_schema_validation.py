from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from quant_investor.v17_v2_contract.canonical import canonical_resource_bytes
from quant_investor.v17_v2_contract.resources import (
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v2_contract.schema_validation import (
    SchemaValidationError,
    packaged_schema_versions,
    preflight_packaged_schema,
    schema_path_for_version,
    validate_canonical_contract_bytes,
    validate_canonical_schema_bytes,
    validate_instance_against_schema,
    validate_mapping_against_packaged_schema,
)
from quant_investor.v17_v2_contract.validators import validate_source_role_matrix


def _schema(**keywords: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "myquant.test.closed-subset.schema.v1",
        **keywords,
    }


def test_exact_packaged_schema_inventory_preflights() -> None:
    versions = packaged_schema_versions()
    assert len(versions) == 24
    assert versions == tuple(sorted(versions))
    assert len({schema_path_for_version(version) for version in versions}) == 24
    for version in versions:
        preflight_packaged_schema(load_packaged_json(schema_path_for_version(version)))


def test_canonical_schema_then_cross_document_dispatch_order() -> None:
    raw = read_packaged_asset("resources/source_role_matrix.v1.json")
    expected = load_packaged_json("resources/source_role_matrix.v1.json")
    callback_calls = 0

    def validate_relationships(document: object) -> dict[str, object]:
        nonlocal callback_calls
        callback_calls += 1
        return validate_source_role_matrix(document)  # type: ignore[arg-type]

    assert (
        validate_canonical_contract_bytes(
            raw,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
            cross_document_validator=validate_relationships,
        )
        == expected
    )
    assert callback_calls == 1

    noncanonical = json.dumps(expected, indent=2).encode("utf-8")
    with pytest.raises(SchemaValidationError, match="canonical compact JSON"):
        validate_canonical_contract_bytes(
            noncanonical,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
            cross_document_validator=validate_relationships,
        )
    assert callback_calls == 1

    schema_invalid = canonical_resource_bytes({**expected, "unexpected": True})
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_canonical_contract_bytes(
            schema_invalid,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
            cross_document_validator=validate_relationships,
        )
    assert callback_calls == 1


def test_acceptance_dispatch_requires_cross_document_validator() -> None:
    raw = read_packaged_asset("resources/source_role_matrix.v1.json")
    dispatcher: Any = validate_canonical_contract_bytes
    with pytest.raises(TypeError, match="cross_document_validator"):
        dispatcher(
            raw,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
        )
    with pytest.raises(SchemaValidationError, match="cross-document validator is required"):
        validate_canonical_contract_bytes(
            raw,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
            cross_document_validator=None,  # type: ignore[arg-type]
        )


def test_schema_only_bytes_helper_is_explicitly_nonaccepting() -> None:
    raw = read_packaged_asset("resources/source_role_matrix.v1.json")
    assert validate_canonical_schema_bytes(
        raw,
        expected_version="myquant.v17.v2.source-role-matrix.v1",
    ) == load_packaged_json("resources/source_role_matrix.v1.json")


def test_packaged_schema_rejects_extra_fields_before_cross_document_logic() -> None:
    matrix = load_packaged_json("resources/source_role_matrix.v1.json")
    changed = {**matrix, "unexpected": True}
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_mapping_against_packaged_schema(
            changed,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
        )


def test_closed_subset_rejects_unsupported_keywords() -> None:
    with pytest.raises(SchemaValidationError, match="unsupported schema keywords"):
        preflight_packaged_schema(_schema(contains={"const": 1}))


def test_closed_subset_distinguishes_boolean_integer_and_number() -> None:
    integer_schema = _schema(type="integer")
    validate_instance_against_schema(1, integer_schema)
    with pytest.raises(SchemaValidationError, match="invalid JSON type"):
        validate_instance_against_schema(True, integer_schema)
    validate_instance_against_schema(1, _schema(type="number"))
    with pytest.raises(SchemaValidationError, match="invalid JSON type"):
        validate_instance_against_schema(False, _schema(type="number"))


def test_closed_subset_rejects_ambiguous_one_of() -> None:
    ambiguous = _schema(oneOf=[{"type": "integer"}, {"type": "number"}])
    with pytest.raises(SchemaValidationError, match="exactly one oneOf"):
        validate_instance_against_schema(1, ambiguous)


def test_closed_subset_enforces_date_time_and_conditional_branches() -> None:
    validate_instance_against_schema(
        "2026-07-22T00:00:00Z",
        _schema(type="string", format="date-time"),
    )
    with pytest.raises(SchemaValidationError, match="valid date-time"):
        validate_instance_against_schema(
            "2026-07-22 00:00:00",
            _schema(type="string", format="date-time"),
        )

    matrix = load_packaged_json("resources/source_role_matrix.v1.json")
    changed = copy.deepcopy(matrix)
    changed["runtime_usable"] = False
    with pytest.raises(SchemaValidationError, match="does not match const"):
        validate_mapping_against_packaged_schema(
            changed,
            expected_version="myquant.v17.v2.source-role-matrix.v1",
        )
