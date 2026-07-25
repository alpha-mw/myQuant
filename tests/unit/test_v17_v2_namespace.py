from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath
import stat

import pytest

from quant_investor.v17_v2_contract.canonical import (
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.namespace import (
    CollisionKind,
    ContentObjectExpectation,
    ContentObjectObservation,
    NAMESPACE_MAP,
    NamespaceContractError,
    NodeKind,
    RESULTS_ROOT,
    SOURCES_ROOT,
    classify_content_object_reuse,
    classify_namespace_collision,
    derive_content_object_path,
    namespace_path,
    node_kind_from_mode,
)

EXPECTED_PATHS = {
    "LATEST": "results/v17_shadow/protocol-v2/_latest/shadow.json",
    "LATEST_LOCK": "results/v17_shadow/protocol-v2/_latest/.latest.lock",
    "MODELS": "results/v17_shadow/protocol-v2/models/objects",
    "OUTCOMES": "results/v17_shadow/protocol-v2/outcomes",
    "RUN_EVENTS": "results/v17_shadow/protocol-v2/runs/run-1/events",
    "RUN_LEDGER": "results/v17_shadow/protocol-v2/runs/run-1/ledger.json",
    "RUN_LOCK": "results/v17_shadow/protocol-v2/runs/run-1/.ledger.lock",
    "RUN_RECEIPTS": "results/v17_shadow/protocol-v2/runs/run-1/receipts",
    "RUN_ROOT": "results/v17_shadow/protocol-v2/runs/run-1",
    "SOURCE_LOCATORS": "data/private/v17_sources/protocol-v2/locators",
    "SOURCE_MANIFESTS": "data/private/v17_sources/protocol-v2/manifests",
    "SOURCE_OBJECTS": "data/private/v17_sources/protocol-v2/objects",
}
EXPECTED_LATEST_CHILDREN = {
    "results/v17_shadow/protocol-v2/_latest/.latest.lock",
    "results/v17_shadow/protocol-v2/_latest/shadow.json",
}


def test_namespace_resource_is_canonical_and_paths_are_protocol_v2_only() -> None:
    resource = (
        Path(__file__).parents[2] / "quant_investor/v17_v2_contract/resources/namespace_map.v1.json"
    )
    raw = resource.read_bytes()
    payload = load_canonical_resource(raw, label="namespace map")
    assert raw == canonical_resource_bytes(payload)
    assert payload["protocol_version"] == "myquant.v17.v2"
    assert payload["version"] == "myquant.v17.v2.namespace-map.v1"
    assert payload["results_root"] == str(RESULTS_ROOT)
    assert payload["sources_root"] == str(SOURCES_ROOT)
    assert set(NAMESPACE_MAP) == set(EXPECTED_PATHS)
    for namespace_id, expected in EXPECTED_PATHS.items():
        run_id = "run-1" if "{run_id}" in NAMESPACE_MAP[namespace_id].path_template else None
        assert str(namespace_path(namespace_id, run_id=run_id)) == expected
        assert "/protocol-v2/" in f"/{expected}/"


def test_latest_namespace_owns_only_pointer_and_coordination_lock() -> None:
    latest_children = {
        spec.path_template
        for spec in NAMESPACE_MAP.values()
        if PurePosixPath(spec.path_template).parent
        == PurePosixPath("results/v17_shadow/protocol-v2/_latest")
    }
    assert latest_children == EXPECTED_LATEST_CHILDREN


def test_namespace_resolution_rejects_unknown_casefold_alias_and_unsafe_run_id() -> None:
    with pytest.raises(NamespaceContractError):
        namespace_path("latest")
    with pytest.raises(NamespaceContractError):
        namespace_path("RUN_LEDGER", run_id="run:1")
    with pytest.raises(NamespaceContractError):
        namespace_path("RUN_LEDGER")
    with pytest.raises(NamespaceContractError):
        namespace_path("LATEST", run_id="run-1")


@pytest.mark.parametrize(
    "legacy_path",
    [
        "results/v17_shadow/models",
        "results/v17_shadow/runs/run-1/ledger.json",
        "data/private/v17_sources/objects",
        "data/private/v17_sources/manifests",
    ],
)
def test_collision_classifier_rejects_legacy_flat_paths(legacy_path: str) -> None:
    with pytest.raises(NamespaceContractError, match="outside protocol-v2 roots"):
        classify_namespace_collision(
            legacy_path,
            expected_kind="directory",
            inventory={},
        )


def _inventory(
    target: str,
    *,
    leaf: NodeKind,
    overrides: dict[str, NodeKind] | None = None,
) -> dict[str, NodeKind]:
    path = PurePosixPath(target)
    result = {
        str(PurePosixPath(*path.parts[:index])): NodeKind.DIRECTORY
        for index in range(1, len(path.parts))
    }
    result[target] = leaf
    result.update(overrides or {})
    return result


def test_collision_classifier_accepts_missing_or_expected_existing_leaf() -> None:
    target = str(namespace_path("RUN_LEDGER", run_id="run-1"))
    clear = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=_inventory(target, leaf=NodeKind.MISSING),
    )
    assert clear.outcome is CollisionKind.CLEAR
    assert clear.safe_to_initialize
    assert not clear.is_collision

    existing = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=_inventory(target, leaf=NodeKind.FILE),
    )
    assert existing.outcome is CollisionKind.EXPECTED_EXISTING
    assert not existing.safe_to_initialize
    assert not existing.is_collision


@pytest.mark.parametrize(
    ("leaf", "outcome"),
    [
        (NodeKind.DIRECTORY, CollisionKind.LEAF_KIND_COLLISION),
        (NodeKind.OTHER, CollisionKind.LEAF_KIND_COLLISION),
        (NodeKind.SYMLINK, CollisionKind.LEAF_SYMLINK),
        (NodeKind.BROKEN_SYMLINK, CollisionKind.LEAF_BROKEN_SYMLINK),
    ],
)
def test_collision_classifier_rejects_every_leaf_collision_kind(
    leaf: NodeKind,
    outcome: CollisionKind,
) -> None:
    target = str(namespace_path("RUN_LEDGER", run_id="run-1"))
    report = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=_inventory(target, leaf=leaf),
    )
    assert report.outcome is outcome
    assert report.is_collision


@pytest.mark.parametrize(
    ("ancestor_kind", "outcome"),
    [
        (NodeKind.FILE, CollisionKind.ANCESTOR_NOT_DIRECTORY),
        (NodeKind.OTHER, CollisionKind.ANCESTOR_NOT_DIRECTORY),
        (NodeKind.SYMLINK, CollisionKind.ANCESTOR_SYMLINK),
        (NodeKind.BROKEN_SYMLINK, CollisionKind.ANCESTOR_BROKEN_SYMLINK),
    ],
)
def test_collision_classifier_rejects_ancestor_collisions(
    ancestor_kind: NodeKind,
    outcome: CollisionKind,
) -> None:
    target = str(namespace_path("RUN_LEDGER", run_id="run-1"))
    ancestor = "results/v17_shadow/protocol-v2"
    report = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=_inventory(
            target,
            leaf=NodeKind.MISSING,
            overrides={ancestor: ancestor_kind},
        ),
    )
    assert report.outcome is outcome
    assert report.conflict_path == ancestor


def test_collision_classifier_requires_complete_consistent_lstat_inventory() -> None:
    target = str(namespace_path("RUN_LEDGER", run_id="run-1"))
    inventory = _inventory(target, leaf=NodeKind.MISSING)
    del inventory["results/v17_shadow"]
    report = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=inventory,
    )
    assert report.outcome is CollisionKind.INVENTORY_INCOMPLETE

    inconsistent = _inventory(
        target,
        leaf=NodeKind.FILE,
        overrides={"results/v17_shadow/protocol-v2": NodeKind.MISSING},
    )
    report = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory=inconsistent,
    )
    assert report.outcome is CollisionKind.INVENTORY_INCONSISTENT


def test_cross_protocol_identity_collision_precedes_inventory_use() -> None:
    target = str(namespace_path("RUN_LEDGER", run_id="run-1"))
    report = classify_namespace_collision(
        target,
        expected_kind="file",
        inventory={},
        cross_protocol_identity_present=True,
    )
    assert report.outcome is CollisionKind.CROSS_PROTOCOL_ID_COLLISION
    assert report.is_collision


def test_node_kind_classification_uses_only_caller_supplied_lstat_facts() -> None:
    assert node_kind_from_mode(stat.S_IFREG | 0o600) is NodeKind.FILE
    assert node_kind_from_mode(stat.S_IFDIR | 0o700) is NodeKind.DIRECTORY
    assert node_kind_from_mode(stat.S_IFLNK | 0o777, link_target_exists=True) is NodeKind.SYMLINK
    assert (
        node_kind_from_mode(stat.S_IFLNK | 0o777, link_target_exists=False)
        is NodeKind.BROKEN_SYMLINK
    )
    assert node_kind_from_mode(stat.S_IFIFO | 0o600) is NodeKind.OTHER
    with pytest.raises(NamespaceContractError):
        node_kind_from_mode(stat.S_IFLNK | 0o777)


def test_content_addressed_reuse_requires_every_identity_and_metadata_field() -> None:
    digest = "a" * 64
    metadata = "b" * 64
    expected = ContentObjectExpectation(
        byte_sha256=digest,
        size_bytes=123,
        metadata_sha256=metadata,
        suffix="parquet",
    )
    derived = str(derive_content_object_path(digest, suffix="parquet"))
    assert derived == ("data/private/v17_sources/protocol-v2/objects/aa/" f"{digest}.parquet")
    observed = ContentObjectObservation(
        path=derived,
        kind=NodeKind.FILE,
        mode=0o600,
        link_count=1,
        size_bytes=123,
        byte_sha256=digest,
        metadata_sha256=metadata,
    )
    assert classify_content_object_reuse(expected, observed).allowed

    mutations = [
        replace(
            observed,
            path="data/private/v17_sources/protocol-v2/objects/wrong",
        ),
        replace(observed, kind=NodeKind.DIRECTORY),
        replace(observed, mode=0o644),
        replace(observed, link_count=2),
        replace(observed, size_bytes=124),
        replace(observed, byte_sha256="c" * 64),
        replace(observed, metadata_sha256="d" * 64),
    ]
    for mutation in mutations:
        decision = classify_content_object_reuse(expected, mutation)
        assert not decision.allowed


@pytest.mark.parametrize("suffix", ["blob", "json", "parquet"])
def test_content_addressed_path_accepts_only_canonical_object_suffixes(suffix: str) -> None:
    digest = "a" * 64
    assert str(derive_content_object_path(digest, suffix=suffix)) == (
        f"data/private/v17_sources/protocol-v2/objects/aa/{digest}.{suffix}"
    )


def test_content_addressed_path_rejects_bin_suffix() -> None:
    with pytest.raises(NamespaceContractError, match="content object suffix"):
        derive_content_object_path("a" * 64, suffix="bin")
