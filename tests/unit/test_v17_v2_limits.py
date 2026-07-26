from __future__ import annotations

from collections.abc import Callable
import hashlib
import math
from pathlib import Path

import pytest

import quant_investor.v17_v2_contract.limits as limits_module
from quant_investor.v17_v2_contract.canonical import (
    CanonicalContractError,
    canonical_json_bytes,
    canonical_resource_bytes,
    canonicalize_set_like,
    load_canonical_resource,
    require_canonical_set_like_wire,
    require_typed_json_scalar,
    seal_semantic,
    semantic_sha256,
    stored_byte_sha256,
    strict_json_loads,
    typed_scalar_total_order_key,
    validate_semantic_seal,
    validate_json_limits,
)
from quant_investor.v17_v2_contract.limits import (
    LIMITS,
    ContractLimitError,
    checked_add,
    get_limit,
    require_nonnegative_int,
)

EXPECTED_LIMITS = {
    "fundamental_generation_manifest_bytes": 128 * 1024 * 1024,
    "general_json_bytes": 16 * 1024 * 1024,
    "max_batch_items": 65_536,
    "max_candidates": 1_024,
    "max_cell_utf8_bytes": 100_000,
    "max_container_members": 65_536,
    "max_dataset_bytes": 512 * 1024 * 1024 * 1024,
    "max_dataset_rows": 100_000_000,
    "max_dataset_shards": 4_096,
    "max_deep_reviews": 1_024,
    "max_depth": 64,
    "max_evidence_refs": 1_024,
    "max_integer_digits": 64,
    "max_key_utf8_bytes": 1_024,
    "max_ledger_artifacts": 128,
    "max_ledger_history": 64,
    "max_shard_bytes": 8 * 1024 * 1024 * 1024,
    "max_sources": 256,
    "max_string_utf8_bytes": 1_048_576,
    "max_symbol_open_days": 2_898,
    "max_total_nodes": 1_000_000,
    "max_universe_symbols": 10_000,
}


def test_limits_table_is_exact_and_resource_is_canonical() -> None:
    assert dict(LIMITS) == EXPECTED_LIMITS
    resource = Path(__file__).parents[2] / "quant_investor/v17_v2_contract/resources/limits.v1.json"
    raw = resource.read_bytes()
    payload = load_canonical_resource(raw, label="limits resource")
    assert raw == canonical_resource_bytes(payload)
    assert payload["protocol_version"] == "myquant.v17.v2"
    assert payload["version"] == "myquant.v17.v2.limits.v1"
    assert raw.endswith(b"\n")
    assert b" " not in raw


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: b'{ "limits":{}, "protocol_version":"myquant.v17.v2",'
        b' "version":"myquant.v17.v2.limits.v1" }\n',
        lambda raw: raw.replace(b'"version":', b'"version":"duplicate","version":', 1),
        lambda raw: b"\xef\xbb\xbf" + raw,
    ],
)
def test_limits_import_loader_rejects_noncanonical_or_ambiguous_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[bytes], bytes],
) -> None:
    source = (
        Path(__file__).parents[2] / "quant_investor/v17_v2_contract/resources/limits.v1.json"
    ).read_bytes()
    target = tmp_path / "limits.json"
    target.write_bytes(mutator(source))
    monkeypatch.setattr(limits_module, "_RESOURCE_PATH", target)
    with pytest.raises(ContractLimitError):
        limits_module._load_limits()


@pytest.mark.parametrize(("name", "maximum"), sorted(EXPECTED_LIMITS.items()))
def test_every_limit_is_inclusive_at_n_and_rejects_n_plus_one(
    name: str,
    maximum: int,
) -> None:
    assert get_limit(name) == maximum
    assert require_nonnegative_int(maximum, label=name, maximum=maximum) == maximum
    with pytest.raises(ContractLimitError):
        require_nonnegative_int(maximum + 1, label=name, maximum=maximum)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, -1])
def test_count_validators_require_exact_nonnegative_int(value: object) -> None:
    with pytest.raises(ContractLimitError):
        require_nonnegative_int(value, label="count")


def test_checked_add_enforces_aggregate_before_addition() -> None:
    maximum = EXPECTED_LIMITS["max_dataset_bytes"]
    assert checked_add(maximum - 1, 1, label="dataset bytes", maximum=maximum) == maximum
    with pytest.raises(ContractLimitError):
        checked_add(maximum - 1, 2, label="dataset bytes", maximum=maximum)
    with pytest.raises(ContractLimitError):
        checked_add(0, True, label="dataset bytes", maximum=maximum)


def test_strict_json_byte_limit_is_checked_before_decoding() -> None:
    maximum = EXPECTED_LIMITS["general_json_bytes"]
    exact = b"null" + (b" " * (maximum - 4))
    assert strict_json_loads(exact) is None
    with pytest.raises(CanonicalContractError, match="byte limit"):
        strict_json_loads(exact + b" ")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":1,"\\u0061":2}',
        b"\xef\xbb\xbf{}",
        b"[NaN]",
        b"[Infinity]",
        b"[-Infinity]",
        b"[1e999]",
    ],
)
def test_strict_json_rejects_duplicate_bom_and_nonfinite(raw: bytes) -> None:
    with pytest.raises(CanonicalContractError):
        strict_json_loads(raw)


def _nested_json_at_depth(depth: int) -> object:
    value: object = None
    for _ in range(depth - 1):
        value = [value]
    return value


def test_json_depth_is_root_one_and_inclusive() -> None:
    maximum = EXPECTED_LIMITS["max_depth"]
    validate_json_limits(_nested_json_at_depth(maximum))
    with pytest.raises(CanonicalContractError, match="depth"):
        validate_json_limits(_nested_json_at_depth(maximum + 1))


def test_json_container_member_limit_is_inclusive() -> None:
    maximum = EXPECTED_LIMITS["max_container_members"]
    validate_json_limits([None] * maximum)
    with pytest.raises(CanonicalContractError, match="member limit"):
        validate_json_limits([None] * (maximum + 1))


def _json_tree_with_nodes(total_nodes: int) -> list[list[None]]:
    # One root plus one node per child list plus one node per null leaf.
    maximum_members = EXPECTED_LIMITS["max_container_members"]
    child_count = math.ceil((total_nodes - 1) / (maximum_members + 1))
    leaf_count = total_nodes - 1 - child_count
    children: list[list[None]] = []
    while leaf_count:
        width = min(maximum_members, leaf_count)
        children.append([None] * width)
        leaf_count -= width
    assert len(children) == child_count
    return children


def test_json_total_node_limit_is_inclusive() -> None:
    maximum = EXPECTED_LIMITS["max_total_nodes"]
    exact = _json_tree_with_nodes(maximum)
    validate_json_limits(exact)
    exact[-1].append(None)
    with pytest.raises(CanonicalContractError, match="total nodes"):
        validate_json_limits(exact)


def test_json_string_key_and_integer_limits_are_inclusive() -> None:
    string_max = EXPECTED_LIMITS["max_string_utf8_bytes"]
    validate_json_limits("x" * string_max)
    with pytest.raises(CanonicalContractError, match="string byte"):
        validate_json_limits("x" * (string_max + 1))

    key_max = EXPECTED_LIMITS["max_key_utf8_bytes"]
    validate_json_limits({"k" * key_max: None})
    with pytest.raises(CanonicalContractError, match="key byte"):
        validate_json_limits({"k" * (key_max + 1): None})

    digits_max = EXPECTED_LIMITS["max_integer_digits"]
    validate_json_limits(int("9" * digits_max))
    with pytest.raises(CanonicalContractError, match="digit"):
        validate_json_limits(int("9" * (digits_max + 1)))
    assert strict_json_loads(("9" * digits_max).encode("ascii")) == int("9" * digits_max)
    with pytest.raises(CanonicalContractError, match="digit"):
        strict_json_loads(("9" * (digits_max + 1)).encode("ascii"))


def test_materialized_json_rejects_bool_as_integer_subclass_only_where_integer_required() -> None:
    validate_json_limits(True)
    with pytest.raises(CanonicalContractError):
        validate_json_limits({1: "non-string-key"})


def test_canonical_json_and_resource_wire_forms_are_exact() -> None:
    value = {"z": [2, 1], "a": "沪"}
    assert canonical_json_bytes(value) == '{"a":"沪","z":[2,1]}'.encode()
    assert canonical_resource_bytes(value) == '{"a":"沪","z":[2,1]}\n'.encode()
    assert load_canonical_resource(canonical_resource_bytes(value)) == value
    with pytest.raises(CanonicalContractError, match="not canonical"):
        load_canonical_resource(b'{ "a":"\\u6caa","z":[2,1] }\n')


def test_set_like_builder_rejects_duplicates_before_total_ordering() -> None:
    values = [
        {"id": "b", "stage": 1, "payload": 1},
        {"id": "a", "stage": 1, "payload": 2},
        {"id": "c", "stage": 0, "payload": 3},
    ]
    ordered = canonicalize_set_like(
        values,
        identity_key=lambda item: item["id"],
        order_key=lambda item: [item["stage"]],
        label="records",
    )
    assert [item["id"] for item in ordered] == ["c", "a", "b"]
    assert (
        require_canonical_set_like_wire(
            ordered,
            identity_key=lambda item: item["id"],
            order_key=lambda item: [item["stage"]],
            label="records",
        )
        == ordered
    )
    with pytest.raises(CanonicalContractError, match="wire order"):
        require_canonical_set_like_wire(
            values,
            identity_key=lambda item: item["id"],
            order_key=lambda item: [item["stage"]],
            label="records",
        )
    with pytest.raises(CanonicalContractError, match="duplicate"):
        canonicalize_set_like(
            [values[0], {**values[0], "payload": 99}],
            identity_key=lambda item: item["id"],
            order_key=lambda item: [item["stage"]],
            label="records",
        )


def test_semantic_hash_removes_only_root_seal_and_sorts_object_keys() -> None:
    left = {
        "version": "myquant.v17.v2.example.v1",
        "nested": {"semantic_sha256": "a" * 64, "value": 1},
        "items": ["a", "b"],
    }
    right = {
        "items": ["a", "b"],
        "nested": {"value": 1, "semantic_sha256": "a" * 64},
        "version": "myquant.v17.v2.example.v1",
    }
    assert semantic_sha256(left) == semantic_sha256(right)

    root_seal = {**left, "semantic_sha256": "f" * 64}
    assert semantic_sha256(root_seal) == semantic_sha256(left)

    nested_tamper = {
        **left,
        "nested": {"semantic_sha256": "b" * 64, "value": 1},
    }
    assert semantic_sha256(nested_tamper) != semantic_sha256(left)


def test_semantic_hash_preserves_order_sensitive_arrays() -> None:
    forward = {"items": [{"id": "a"}, {"id": "b"}]}
    reverse = {"items": [{"id": "b"}, {"id": "a"}]}
    assert semantic_sha256(forward) != semantic_sha256(reverse)


def test_semantic_seal_rejects_presealed_or_wrong_seal() -> None:
    payload = {"version": "myquant.v17.v2.example.v1", "items": [1, 2]}
    sealed = seal_semantic(payload)
    assert validate_semantic_seal(sealed) == sealed
    with pytest.raises(CanonicalContractError, match="must not be supplied"):
        seal_semantic(sealed)
    with pytest.raises(CanonicalContractError, match="mismatch"):
        validate_semantic_seal({**sealed, "semantic_sha256": "0" * 64})


def test_semantic_hash_excludes_newline_but_stored_byte_hash_includes_it() -> None:
    payload = {"version": "myquant.v17.v2.example.v1"}
    expected_semantic = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    expected_stored = hashlib.sha256(canonical_resource_bytes(payload)).hexdigest()
    assert semantic_sha256(payload) == expected_semantic
    assert stored_byte_sha256(payload) == expected_stored
    assert expected_semantic != expected_stored
    with pytest.raises(CanonicalContractError, match="byte limit"):
        semantic_sha256(payload, max_bytes=len(canonical_json_bytes(payload)) - 1)


def test_typed_scalar_total_order_is_stable_across_json_scalar_types() -> None:
    values: list[object] = ["é", 2, True, -1.5, None, "e\u0301", False, 1]
    ordered = sorted(
        values,
        key=lambda item: typed_scalar_total_order_key(item, allow_null=True),
    )
    assert ordered == [None, False, True, -1.5, 1, 2, "e\u0301", "é"]
    keys = [typed_scalar_total_order_key(item, allow_null=True) for item in ordered]
    assert keys == sorted(keys)
    assert keys[-2] != keys[-1]


def test_typed_scalar_numeric_domain_is_exact_for_large_ints_and_floats() -> None:
    below = 10**40
    above = below + 1
    assert typed_scalar_total_order_key(below, allow_null=False) < (
        typed_scalar_total_order_key(above, allow_null=False)
    )
    assert typed_scalar_total_order_key(-1.5, allow_null=False) < (
        typed_scalar_total_order_key(0, allow_null=False)
    )
    assert typed_scalar_total_order_key(0, allow_null=False) < (
        typed_scalar_total_order_key(0.5, allow_null=False)
    )


@pytest.mark.parametrize("value", [1.0, 0.0, -0.0, -2.0])
def test_typed_scalar_rejects_integral_float_alternate_encodings(value: float) -> None:
    with pytest.raises(CanonicalContractError, match="alternate integer encoding"):
        typed_scalar_total_order_key(value, allow_null=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_typed_scalar_rejects_nonfinite_float(value: float) -> None:
    with pytest.raises(CanonicalContractError, match="finite"):
        typed_scalar_total_order_key(value, allow_null=False)


def test_typed_scalar_keeps_bool_out_of_integer_domain_and_controls_null() -> None:
    bool_key = typed_scalar_total_order_key(True, allow_null=False)
    integer_key = typed_scalar_total_order_key(1, allow_null=False)
    assert bool_key[0] == 1
    assert integer_key[0] == 2
    assert bool_key != integer_key
    with pytest.raises(CanonicalContractError, match="cannot be null"):
        require_typed_json_scalar(None, allow_null=False)


def test_typed_scalar_enforces_integer_and_unicode_byte_limits() -> None:
    assert require_typed_json_scalar(int("9" * 64), allow_null=False) == int("9" * 64)
    with pytest.raises(CanonicalContractError, match="digit"):
        require_typed_json_scalar(int("9" * 65), allow_null=False)
    assert require_typed_json_scalar("e\u0301", allow_null=False) == "e\u0301"
