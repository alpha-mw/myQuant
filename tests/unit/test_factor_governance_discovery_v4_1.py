from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import pytest

from quant_investor.factors import governance_discovery_v4_1 as discovery
from quant_investor.factors.governance_screening_v4 import (
    build_candidate_catalog_v4,
)
from quant_investor.factors.governance_source_readback_v4_1 import (
    binding_semantic_sha256_v4_1,
)


WORKSPACE = Path("/Users/maxwell/mySpace")
REPOSITORY = WORKSPACE / "myQuant"
AQUANT_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
CYCLE_ID = "cn_full_a_v4_1_20260717"
RUN_ID = "factor_v4_1_discovery_core_test"

PREDECESSOR_ROOT = (
    REPOSITORY
    / "reports/factor_governance/private/v4_1_cycle"
    / "factor_v4_1_cutoff_20260718T151837Z"
)
BASE_ROOT = (
    REPOSITORY
    / "reports/factor_governance/private/v4_pre_admission"
    / "factor_v4_pre_admission_20260718_083224"
)

EXPECTED_SOURCE_OBJECTS = {
    "A_quant/app/data/schemas.py": (
        "2bc56bfea1e0dd6a31a230b72422e0238312f20d",
        "848f324ada44b1d6e4c944d7e156fa9901779da797c51d8076e7b56db0a55817",
    ),
    "A_quant/app/factor_sandbox/expression.py": (
        "d8acdd565b8bba27ffaf02ec44a7029ec63e832d",
        "df93622a33309aa28d065d6e8fd366de1ebf7d2be600b26170084f727a7dc936",
    ),
    "A_quant/app/factor_sandbox/matrix_dataset.py": (
        "ef6f6d0a408176a0e3151d619d097c5190d60ef8",
        "eab9ba96576d040622ae170fc36689a4ee62b64f13a91ae0efe9ff9cd8942547",
    ),
    "A_quant/app/factor_sandbox/operators.py": (
        "bd3365fb994a941caa62913156c2a6fb172bd697",
        "367f0c68a1e6f8c2e7f0fe168c91e23d77689f101fd203889d5c5b1c2bdb80a1",
    ),
    "A_quant/docs/factor_time_alignment_policy.md": (
        "ef4de17343b3d24bbb1560537bf0c4354b60ebdb",
        "e913ac9909927652b37571ee47c15d06e77b28227e1ee1f588179b435471f083",
    ),
    "A_quant/scripts/run_factor_batch_screen.py": (
        "6de605a9ebc6c4b1f9cd730c5ffe350d11e8aef9",
        "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312",
    ),
}

EXPECTED_ORDERED_NAMES_SHA = (
    "64078f603d4484cb7f2dd167275ab25e790e10613ac7046f6da66f541d32bbab"
)
EXPECTED_COMPATIBLE_NAMES_SHA = (
    "38e1d7268028436dfb23deb0543816030d97adab65997babe8361d0646e97f6e"
)
EXPECTED_ALIAS_NAMES_SHA = (
    "abb938af17b0875f72d994697de3c3a20209ad862b5fac6c535c91b0915c597d"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert isinstance(value, dict)
    return value


def _git_environment() -> dict[str, str]:
    # Deliberately do not inherit any GIT_* variable or user/system Git config.
    return {
        "HOME": "/nonexistent-factor-governance-test-home",
        "PATH": "/usr/bin:/bin",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
        "LANG": "C",
    }


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(WORKSPACE), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    ).stdout


def _read_pinned_git_objects() -> tuple[bytes, list[dict[str, str]]]:
    assert _git("rev-parse", "--show-toplevel").decode().strip() == str(WORKSPACE)
    assert _git("cat-file", "-t", AQUANT_COMMIT).decode().strip() == "commit"
    source_files: list[dict[str, str]] = []
    generator = b""
    for path in sorted(EXPECTED_SOURCE_OBJECTS):
        expected_oid, expected_raw_sha = EXPECTED_SOURCE_OBJECTS[path]
        tree_row = _git("ls-tree", AQUANT_COMMIT, "--", path).decode().rstrip("\n")
        prefix, observed_path = tree_row.split("\t", 1)
        mode, object_type, oid = prefix.split()
        assert observed_path == path
        assert (mode, object_type, oid) == ("100644", "blob", expected_oid)
        raw = _git("cat-file", "blob", oid)
        assert _sha256(raw) == expected_raw_sha
        source_files.append(
            {
                "path": path,
                "git_mode": "100644",
                "blob_oid": oid,
                "raw_sha256": expected_raw_sha,
            }
        )
        if path == discovery.AQUANT_GENERATOR_PATH:
            generator = raw
    assert generator
    return generator, source_files


def _code_bindings() -> list[dict[str, Any]]:
    paths = [
        REPOSITORY / "quant_investor/factors/aquant_expression.py",
        REPOSITORY / "quant_investor/factors/governance_cycle_state_v4_1.py",
        REPOSITORY
        / "quant_investor/factors/governance_discovery_readback_v4_1.py",
        REPOSITORY / "quant_investor/factors/governance_discovery_v4_1.py",
        REPOSITORY / "quant_investor/factors/governance_screening_v4.py",
        REPOSITORY / "quant_investor/factors/governance_source_readback_v4_1.py",
        REPOSITORY / "quant_investor/factors/governance_source_v4_1.py",
        REPOSITORY / "scripts/build_factor_v4_1_discovery.py",
    ]
    rows = []
    for path in sorted(paths):
        raw = path.read_bytes()
        rows.append(
            {
                "absolute_path": str(path),
                "raw_sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
    return rows


def _semantic_identity(filename: str, value: dict[str, Any], raw: bytes) -> str:
    if filename == "cutoff_input_binding.v4_1.json":
        return binding_semantic_sha256_v4_1(value)
    if filename == "cycle_state.precommitted.v4_1.json":
        return str(value["state_semantic_sha256"])
    if filename in {"design_source.v4_1.json", "source_chain_node.v4_1.json"}:
        return str(value["semantic_sha256"])
    if filename == "source_readback_report.v4_1.json":
        # The frozen PRECOMMITTED readback contract defines its semantic bytes
        # as the complete canonical newline-bearing object.
        return _sha256(discovery.canonical_file_bytes(value))
    raise AssertionError(filename)


def _predecessor_values() -> tuple[
    dict[str, dict[str, Any]], list[dict[str, str]]
]:
    values: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, str]] = []
    for filename in discovery.PREDECESSOR_BUNDLE_FILENAMES:
        path = PREDECESSOR_ROOT / filename
        raw = path.read_bytes()
        value = _strict_json(path)
        values[filename] = value
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": _sha256(raw),
                "semantic_sha256": _semantic_identity(filename, value, raw),
            }
        )
    return values, bindings


def _artifact_binding(filename: str, value: dict[str, Any]) -> dict[str, Any]:
    raw = discovery.canonical_file_bytes(value)
    semantic_field = discovery.SELF_HASH_FIELD_BY_FILENAME[filename]
    return {
        "filename": filename,
        "byte_sha256": _sha256(raw),
        "semantic_sha256": value[semantic_field],
        "size_bytes": len(raw),
        "mode": 0o600,
        "uid": os.getuid(),
        "nlink": 1,
    }


@pytest.fixture(scope="module")
def real_discovery() -> dict[str, Any]:
    generator_source, source_files = _read_pinned_git_objects()
    candidates = discovery.extract_aquant_candidates_from_source(generator_source)

    ontology_path = BASE_ROOT / "primitive_ontology.v4.json"
    catalog_path = BASE_ROOT / "candidate_catalog.v4.json"
    ontology_raw = ontology_path.read_bytes()
    base_catalog_raw = catalog_path.read_bytes()
    ontology = _strict_json(ontology_path)
    base_catalog = _strict_json(catalog_path)
    evaluator_raw = (
        REPOSITORY / "quant_investor/factors/aquant_expression.py"
    ).read_bytes()

    receipt = discovery.build_aquant_source_receipt_v4_1(
        repository_top_level=str(WORKSPACE),
        pinned_commit=AQUANT_COMMIT,
        source_files=source_files,
        candidates=candidates,
    )
    contract = discovery.build_local_compatibility_contract_v4_1(
        evaluator_source_byte_sha256=_sha256(evaluator_raw)
    )
    audit = discovery.build_source_idea_audit_v4_1(
        cycle_id=CYCLE_ID,
        candidates=candidates,
        source_receipt=receipt,
        compatibility_contract=contract,
        base_catalog=base_catalog,
    )
    catalog = discovery.build_discovery_catalog_v4_1(
        cycle_id=CYCLE_ID,
        base_ontology=ontology,
        base_catalog=base_catalog,
        source_receipt=receipt,
        compatibility_contract=contract,
        source_idea_audit=audit,
    )
    collision = discovery.build_structural_collision_audit_v4_1(
        cycle_id=CYCLE_ID,
        discovery_catalog=catalog,
    )

    predecessor, predecessor_bindings = _predecessor_values()
    predecessor_source = predecessor["source_chain_node.v4_1.json"]
    predecessor_state = predecessor["cycle_state.precommitted.v4_1.json"]
    source_node = discovery.build_discovery_source_node_v4_1(
        cycle_id=CYCLE_ID,
        run_id=RUN_ID,
        predecessor_bundle_bindings=predecessor_bindings,
        predecessor_source_node=predecessor_source,
        predecessor_source_node_byte_sha256=_sha256(
            (PREDECESSOR_ROOT / "source_chain_node.v4_1.json").read_bytes()
        ),
        predecessor_state=predecessor_state,
        predecessor_state_byte_sha256=_sha256(
            (PREDECESSOR_ROOT / "cycle_state.precommitted.v4_1.json").read_bytes()
        ),
        base_ontology=ontology,
        base_ontology_byte_sha256=_sha256(ontology_raw),
        base_catalog=base_catalog,
        base_catalog_byte_sha256=_sha256(base_catalog_raw),
        aquant_source_receipt=receipt,
        local_compatibility_contract=contract,
        source_idea_audit=audit,
        discovery_catalog=catalog,
        structural_collision_audit=collision,
        code_bindings=_code_bindings(),
    )
    predecessor_state_raw_sha = _sha256(
        (PREDECESSOR_ROOT / "cycle_state.precommitted.v4_1.json").read_bytes()
    )
    state = discovery.build_discovery_cycle_state_v4_1(
        predecessor_state=predecessor_state,
        predecessor_state_byte_sha256=predecessor_state_raw_sha,
        expected_predecessor_byte_sha256=predecessor_state_raw_sha,
        expected_predecessor_semantic_sha256=predecessor_state[
            "state_semantic_sha256"
        ],
        cycle_id=CYCLE_ID,
        cycle_root_sha256=predecessor_state["cycle_root_sha256"],
        discovery_source_node=source_node,
    )
    bundle: dict[str, dict[str, Any]] = {
        discovery.AQUANT_SOURCE_RECEIPT_FILENAME: receipt,
        discovery.SOURCE_IDEA_AUDIT_FILENAME: audit,
        discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME: contract,
        discovery.DISCOVERY_CATALOG_FILENAME: catalog,
        discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME: collision,
        discovery.DISCOVERY_SOURCE_NODE_FILENAME: source_node,
        discovery.DISCOVERY_CYCLE_STATE_FILENAME: state,
    }
    report = discovery.build_discovery_readback_report_v4_1(
        cycle_id=CYCLE_ID,
        run_id=RUN_ID,
        artifact_bindings=[
            _artifact_binding(filename, bundle[filename])
            for filename in discovery.PRE_READBACK_ARTIFACT_FILENAMES
        ],
    )
    bundle[discovery.DISCOVERY_READBACK_REPORT_FILENAME] = report
    return {
        "generator_source": generator_source,
        "source_files": source_files,
        "candidates": candidates,
        "ontology": ontology,
        "base_catalog": base_catalog,
        "ontology_raw": ontology_raw,
        "base_catalog_raw": base_catalog_raw,
        "predecessor": predecessor,
        "predecessor_bindings": predecessor_bindings,
        "bundle": bundle,
    }


def _replace_once(source: bytes, old: bytes, new: bytes) -> bytes:
    assert source.count(old) >= 1, old
    return source.replace(old, new, 1)


def _inject_before_gap(source: bytes, statement: bytes) -> bytes:
    marker = b'    gap = "(vwap - close) / close"\n'
    return _replace_once(source, marker, b"    " + statement + b"\n" + marker)


def _reseal(value: dict[str, Any], self_hash_field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[self_hash_field] = discovery.semantic_sha256(
        {key: item for key, item in result.items() if key != self_hash_field}
    )
    return result


def _rebuild_linked_bundle_from_audit(
    real_discovery: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebuild every descendant so a test tamper is not just a stale-link test."""

    original = real_discovery["bundle"]
    receipt = original[discovery.AQUANT_SOURCE_RECEIPT_FILENAME]
    contract = original[discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME]
    ontology = real_discovery["ontology"]
    base_catalog = real_discovery["base_catalog"]
    catalog = discovery.build_discovery_catalog_v4_1(
        cycle_id=CYCLE_ID,
        base_ontology=ontology,
        base_catalog=base_catalog,
        source_receipt=receipt,
        compatibility_contract=contract,
        source_idea_audit=audit,
    )
    collision = discovery.build_structural_collision_audit_v4_1(
        cycle_id=CYCLE_ID,
        discovery_catalog=catalog,
    )
    predecessor = real_discovery["predecessor"]
    predecessor_source = predecessor["source_chain_node.v4_1.json"]
    predecessor_state = predecessor["cycle_state.precommitted.v4_1.json"]
    predecessor_source_raw_sha = _sha256(
        (PREDECESSOR_ROOT / "source_chain_node.v4_1.json").read_bytes()
    )
    predecessor_state_raw_sha = _sha256(
        (PREDECESSOR_ROOT / "cycle_state.precommitted.v4_1.json").read_bytes()
    )
    node = discovery.build_discovery_source_node_v4_1(
        cycle_id=CYCLE_ID,
        run_id=RUN_ID,
        predecessor_bundle_bindings=real_discovery["predecessor_bindings"],
        predecessor_source_node=predecessor_source,
        predecessor_source_node_byte_sha256=predecessor_source_raw_sha,
        predecessor_state=predecessor_state,
        predecessor_state_byte_sha256=predecessor_state_raw_sha,
        base_ontology=ontology,
        base_ontology_byte_sha256=_sha256(real_discovery["ontology_raw"]),
        base_catalog=base_catalog,
        base_catalog_byte_sha256=_sha256(real_discovery["base_catalog_raw"]),
        aquant_source_receipt=receipt,
        local_compatibility_contract=contract,
        source_idea_audit=audit,
        discovery_catalog=catalog,
        structural_collision_audit=collision,
        code_bindings=_code_bindings(),
    )
    state = discovery.build_discovery_cycle_state_v4_1(
        predecessor_state=predecessor_state,
        predecessor_state_byte_sha256=predecessor_state_raw_sha,
        expected_predecessor_byte_sha256=predecessor_state_raw_sha,
        expected_predecessor_semantic_sha256=predecessor_state[
            "state_semantic_sha256"
        ],
        cycle_id=CYCLE_ID,
        cycle_root_sha256=predecessor_state["cycle_root_sha256"],
        discovery_source_node=node,
    )
    bundle: dict[str, dict[str, Any]] = {
        discovery.AQUANT_SOURCE_RECEIPT_FILENAME: receipt,
        discovery.SOURCE_IDEA_AUDIT_FILENAME: audit,
        discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME: contract,
        discovery.DISCOVERY_CATALOG_FILENAME: catalog,
        discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME: collision,
        discovery.DISCOVERY_SOURCE_NODE_FILENAME: node,
        discovery.DISCOVERY_CYCLE_STATE_FILENAME: state,
    }
    bundle[discovery.DISCOVERY_READBACK_REPORT_FILENAME] = (
        discovery.build_discovery_readback_report_v4_1(
            cycle_id=CYCLE_ID,
            run_id=RUN_ID,
            artifact_bindings=[
                _artifact_binding(filename, bundle[filename])
                for filename in discovery.PRE_READBACK_ARTIFACT_FILENAMES
            ],
        )
    )
    return bundle


def test_real_pinned_git_blob_ast_extraction_oracle(
    real_discovery: dict[str, Any],
) -> None:
    candidates = real_discovery["candidates"]
    assert len(candidates) == 100
    assert candidates[0]["name"] == "alpha_vwap_gap_reversion_s20"
    assert candidates[-1]["name"] == "alpha_vwap_low_debt_160"
    assert discovery.semantic_sha256([row["name"] for row in candidates]) == (
        EXPECTED_ORDERED_NAMES_SHA
    )
    assert [row["path"] for row in real_discovery["source_files"]] == sorted(
        EXPECTED_SOURCE_OBJECTS
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda source: _replace_once(
            source, b"list[BatchFactorCandidate]", b"list[object]"
        ),
        lambda source: _replace_once(
            source, b"seen.add(name)", b"seen.update([name])"
        ),
        lambda source: _replace_once(
            source, b"pos.format(w=window)", b"pos.upper()"
        ),
        lambda source: _replace_once(
            source, b"min(window, 60)", b"max(window, 60)"
        ),
        lambda source: _inject_before_gap(source, b"import os"),
        lambda source: _inject_before_gap(source, b'exec("pass")'),
        lambda source: _inject_before_gap(source, b'eval("1")'),
        lambda source: _inject_before_gap(source, b'compile("1", "x", "eval")'),
        lambda source: _replace_once(
            source,
            b'gap = "(vwap - close) / close"',
            b'gap = ["(vwap - close) / close"][0]',
        ),
        lambda source: _replace_once(
            source,
            b'gap = "(vwap - close) / close"',
            b'gap = [item for item in ["(vwap - close) / close"]]',
        ),
        lambda source: _replace_once(
            source,
            b'gap = "(vwap - close) / close"',
            b'gap = (lambda: "(vwap - close) / close")()',
        ),
        lambda source: _replace_once(
            source,
            b'gap = "(vwap - close) / close"',
            b"gap = later_bound_name",
        ),
        lambda source: _replace_once(
            source,
            b"    candidates: list[BatchFactorCandidate] = []\n"
            b"    seen: set[str] = set()\n",
            b"    seen: set[str] = set()\n"
            b"    candidates: list[BatchFactorCandidate] = []\n",
        ),
    ],
    ids=[
        "unknown-annotation",
        "unknown-mutation",
        "arbitrary-attribute",
        "unknown-call",
        "import",
        "exec",
        "eval",
        "compile",
        "subscript",
        "comprehension",
        "lambda",
        "use-before-bind",
        "reordered-initializers",
    ],
)
def test_generator_ast_interpreter_rejects_adversarial_constructs(
    real_discovery: dict[str, Any],
    mutator: Callable[[bytes], bytes],
) -> None:
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.extract_aquant_candidates_from_source(
            mutator(real_discovery["generator_source"])
        )


@pytest.mark.parametrize(
    "expression",
    [
        "close.real",
        "close[0]",
        "[x for x in close]",
        "(lambda x: x)(close)",
        "cs_rank(close, axis=1)",
        "obj.cs_rank(close)",
    ],
)
def test_expression_normalizer_rejects_non_allowlisted_ast(expression: str) -> None:
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.normalize_expression_ast_v4_1(expression)


def test_real_compatibility_alias_and_catalog_accounting(
    real_discovery: dict[str, Any],
) -> None:
    bundle = real_discovery["bundle"]
    audit = bundle[discovery.SOURCE_IDEA_AUDIT_FILENAME]
    catalog = bundle[discovery.DISCOVERY_CATALOG_FILENAME]

    assert audit["total_idea_count"] == 100
    assert audit["compatible_count"] == 43
    assert audit["incompatible_count"] == 57
    assert audit["new_candidate_count"] == 37
    assert audit["structural_alias_count"] == 6
    assert audit["compatible_ordered_names_semantic_sha256"] == (
        EXPECTED_COMPATIBLE_NAMES_SHA
    )
    assert audit["structural_alias_ordered_names_semantic_sha256"] == (
        EXPECTED_ALIAS_NAMES_SHA
    )
    assert catalog["member_count"] == 273
    assert catalog["selected_count"] == 267
    assert catalog["member_count"] - catalog["selected_count"] == 6

    base_catalog = real_discovery["base_catalog"]
    base_by_name = {row["name"]: row for row in base_catalog["candidates"]}
    base_members = [
        row for row in catalog["members"] if row["origin"] == "myquant"
    ]
    assert len(base_members) == 230
    assert {row["name"] for row in base_members} == set(base_by_name)
    for member in base_members:
        base = base_by_name[member["name"]]
        assert member["source_definition_sha256"] == base["definition_sha256"]
        assert member["candidate_id"] == (
            f"myquant:{base_catalog['semantic_sha256']}:{member['name']}"
        )
        assert member["catalog_role"] == "base_reference"
        assert member["selected"] is True
    assert all(
        type(row["initial_weight"]) is float and row["initial_weight"] == 0.0
        for row in catalog["members"]
    )


def test_native_predecessor_and_base_encodings_are_exact(
    real_discovery: dict[str, Any],
) -> None:
    ontology_raw = real_discovery["ontology_raw"]
    catalog_raw = real_discovery["base_catalog_raw"]
    assert not ontology_raw.endswith(b"\n")
    assert not catalog_raw.endswith(b"\n")
    assert _sha256(ontology_raw) == (
        "8ec6e7b4e5de7d2857b14226b2f208f025ce2f6a99a0caa024e51b7ee939ce71"
    )
    assert _sha256(catalog_raw) == (
        "24860fbaa6482ecbffccb4bc41fc842475f76e308b4b232ef5bffe427a61efa4"
    )
    assert real_discovery["ontology"]["semantic_sha256"] == (
        "b1f803026fff12b52392265fb81d314b77c865dc105706b7a243cbe5dd6c2e0b"
    )
    assert real_discovery["base_catalog"]["semantic_sha256"] == (
        "e427a71fd95be62aca85bc893a809d3c54cea965976cdcff9a0a4f1500b07c99"
    )

    predecessor_source_path = PREDECESSOR_ROOT / "source_chain_node.v4_1.json"
    source_raw = predecessor_source_path.read_bytes()
    source = real_discovery["predecessor"]["source_chain_node.v4_1.json"]
    assert source_raw.endswith(b"\n")
    assert _sha256(source_raw) == (
        "66e8b9501bd57ef836ccb10ef3729f483a5465d20a97e90567568b0202f13612"
    )
    assert source["semantic_sha256"] == (
        "aff8a890ea3b6871ec45803a7a183c0935ff7b69c45c50ad8c2beaf88a60d7e1"
    )
    assert source["semantic_sha256"] == discovery.byte_sha256(
        {key: item for key, item in source.items() if key != "semantic_sha256"}
    )

    state_path = PREDECESSOR_ROOT / "cycle_state.precommitted.v4_1.json"
    state = real_discovery["predecessor"]["cycle_state.precommitted.v4_1.json"]
    assert _sha256(state_path.read_bytes()) == (
        "8f7515692fea0eb90f05e0c4461387b38c73ebae362292ad1109e234bcd105d8"
    )
    assert state["state_semantic_sha256"] == (
        "3029ae8d84daa8318c7b5bdc1ec5fdeef9bca889c824acc58237c9850f40892f"
    )
    assert len(real_discovery["predecessor_bindings"]) == 5
    assert [row["filename"] for row in real_discovery["predecessor_bindings"]] == (
        list(discovery.PREDECESSOR_BUNDLE_FILENAMES)
    )


def test_source_node_binds_all_predecessor_artifacts_and_code(
    real_discovery: dict[str, Any],
) -> None:
    node = real_discovery["bundle"][discovery.DISCOVERY_SOURCE_NODE_FILENAME]
    assert node["predecessor_bundle_bindings"] == real_discovery[
        "predecessor_bindings"
    ]
    assert len(node["predecessor_bundle_bindings"]) == 5
    assert len(node["code_bindings"]) == 8
    assert node["code_bindings_semantic_sha256"] == discovery.semantic_sha256(
        node["code_bindings"]
    )
    assert {
        row["absolute_path"] for row in node["code_bindings"]
    } == {row["absolute_path"] for row in _code_bindings()}
    assert node["base_ontology"] == {
        "artifact_kind": "base_ontology",
        "byte_sha256": _sha256(real_discovery["ontology_raw"]),
        "semantic_sha256": real_discovery["ontology"]["semantic_sha256"],
    }
    assert node["base_catalog"] == {
        "artifact_kind": "base_catalog",
        "byte_sha256": _sha256(real_discovery["base_catalog_raw"]),
        "semantic_sha256": real_discovery["base_catalog"]["semantic_sha256"],
    }


def test_all_individual_artifacts_have_exact_schemas_and_self_hashes(
    real_discovery: dict[str, Any],
) -> None:
    bundle = real_discovery["bundle"]
    assert tuple(bundle) == discovery.CANONICAL_ARTIFACT_FILENAMES
    for filename in discovery.CANONICAL_ARTIFACT_FILENAMES:
        value = bundle[filename]
        assert discovery.validate_discovery_artifact_v4_1(filename, value) == value

        missing = copy.deepcopy(value)
        missing.pop(next(iter(missing)))
        with pytest.raises((TypeError, ValueError)):
            discovery.validate_discovery_artifact_v4_1(filename, missing)

        unknown = copy.deepcopy(value)
        unknown["unexpected_field"] = False
        with pytest.raises((TypeError, ValueError)):
            discovery.validate_discovery_artifact_v4_1(filename, unknown)

        tampered = copy.deepcopy(value)
        self_hash_field = discovery.SELF_HASH_FIELD_BY_FILENAME[filename]
        tampered[self_hash_field] = "f" * 64
        with pytest.raises((TypeError, ValueError)):
            discovery.validate_discovery_artifact_v4_1(filename, tampered)

    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_artifact_v4_1("unknown.json", {})


def test_cross_bundle_recomputes_links_compatibility_aliases_and_state_cas(
    real_discovery: dict[str, Any],
) -> None:
    bundle = real_discovery["bundle"]
    normalized = discovery.validate_discovery_bundle_v4_1(
        bundle,
        base_ontology=real_discovery["ontology"],
        base_catalog=real_discovery["base_catalog"],
    )
    assert normalized == bundle
    assert discovery.validate_discovery_bundle_values_v4_1(bundle) == bundle

    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(
            {
                key: value
                for key, value in bundle.items()
                if key != discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME
            }
        )
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(
            {**bundle, "extra.json": {}},
        )
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(
            bundle, base_ontology=real_discovery["ontology"]
        )


def test_cross_bundle_rejects_individually_valid_resealed_substitutions(
    real_discovery: dict[str, Any],
) -> None:
    original = real_discovery["bundle"]

    contract = copy.deepcopy(
        original[discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME]
    )
    contract["evaluator_source_byte_sha256"] = "1" * 64
    contract = _reseal(contract, "contract_semantic_sha256")
    discovery.validate_local_compatibility_contract_v4_1(contract)
    substituted = copy.deepcopy(original)
    substituted[discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME] = contract
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)

    state = copy.deepcopy(original[discovery.DISCOVERY_CYCLE_STATE_FILENAME])
    state["source_chain_node_sha256"] = "3" * 64
    state = _reseal(state, "state_semantic_sha256")
    discovery.validate_discovery_artifact_v4_1(
        discovery.DISCOVERY_CYCLE_STATE_FILENAME, state
    )
    substituted = copy.deepcopy(original)
    substituted[discovery.DISCOVERY_CYCLE_STATE_FILENAME] = state
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)

    collision = copy.deepcopy(
        original[discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME]
    )
    collision["discovery_catalog_sha256"] = "2" * 64
    collision = _reseal(collision, "audit_semantic_sha256")
    discovery.validate_structural_collision_audit_v4_1(collision)
    substituted = copy.deepcopy(original)
    substituted[discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME] = collision
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)

    audit = copy.deepcopy(original[discovery.SOURCE_IDEA_AUDIT_FILENAME])
    audit["cycle_id"] = "different_cycle"
    audit = _reseal(audit, "audit_semantic_sha256")
    discovery.validate_source_idea_audit_v4_1(audit)
    substituted = copy.deepcopy(original)
    substituted[discovery.SOURCE_IDEA_AUDIT_FILENAME] = audit
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)

    source_node = copy.deepcopy(
        original[discovery.DISCOVERY_SOURCE_NODE_FILENAME]
    )
    source_node["run_id"] = "different_run"
    source_node = _reseal(source_node, "semantic_sha256")
    discovery.validate_discovery_source_node_v4_1(source_node)
    substituted = copy.deepcopy(original)
    substituted[discovery.DISCOVERY_SOURCE_NODE_FILENAME] = source_node
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)

    state = copy.deepcopy(original[discovery.DISCOVERY_CYCLE_STATE_FILENAME])
    state["predecessor"]["byte_sha256"] = "4" * 64
    state = _reseal(state, "state_semantic_sha256")
    discovery.validate_discovery_artifact_v4_1(
        discovery.DISCOVERY_CYCLE_STATE_FILENAME, state
    )
    substituted = copy.deepcopy(original)
    substituted[discovery.DISCOVERY_CYCLE_STATE_FILENAME] = state
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(substituted)


def test_cross_bundle_recomputes_compatibility_after_a_fully_relinked_tamper(
    real_discovery: dict[str, Any],
) -> None:
    audit = copy.deepcopy(
        real_discovery["bundle"][discovery.SOURCE_IDEA_AUDIT_FILENAME]
    )
    idea = next(
        row
        for row in audit["ideas"]
        if row["compatibility_status"] == "compatible"
        and row["catalog_role"] == "new_candidate"
        and "ts_mean" in row["expression"]
    )
    old_expression = idea["expression"]
    assert "20" in old_expression
    idea["expression"] = old_expression.replace("20", "21", 1)
    assessment = discovery.assess_local_compatibility_v4_1(
        idea["expression"],
        real_discovery["bundle"][
            discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME
        ],
    )
    assert assessment["status"] == "compatible"
    idea["normalized_expression_ast"] = assessment["normalized_expression_ast"]
    idea["input_fields"] = assessment["input_fields"]
    idea["lookback"] = assessment["lookback"]
    # Retain the old fingerprint. The row and every descendant artifact can be
    # individually resealed, but the bundle validator must recompute it.
    audit = _reseal(audit, "audit_semantic_sha256")
    discovery.validate_source_idea_audit_v4_1(audit)
    linked_bundle = _rebuild_linked_bundle_from_audit(real_discovery, audit)
    for filename, value in linked_bundle.items():
        discovery.validate_discovery_artifact_v4_1(filename, value)
    with pytest.raises(
        discovery.FactorGovernanceDiscoveryV4_1Error,
        match="fingerprint mismatch",
    ):
        discovery.validate_discovery_bundle_v4_1(
            linked_bundle,
            base_ontology=real_discovery["ontology"],
            base_catalog=real_discovery["base_catalog"],
        )


def test_optional_base_rebuild_rejects_a_different_valid_base_catalog(
    real_discovery: dict[str, Any],
) -> None:
    original = real_discovery["base_catalog"]
    input_fields = {
        "name",
        "implementation",
        "expression",
        "direction",
        "params",
        "lookback",
        "slot",
        "input_fields",
        "primitive_ids",
    }
    candidates = [
        {key: copy.deepcopy(value) for key, value in row.items() if key in input_fields}
        for row in original["candidates"]
    ]
    candidates[0]["params"] = {**candidates[0]["params"], "test_variant": 1}
    alternate = build_candidate_catalog_v4(
        ontology=real_discovery["ontology"], candidates=candidates
    )
    assert alternate["semantic_sha256"] != original["semantic_sha256"]
    with pytest.raises(discovery.FactorGovernanceDiscoveryV4_1Error):
        discovery.validate_discovery_bundle_v4_1(
            real_discovery["bundle"],
            base_ontology=real_discovery["ontology"],
            base_catalog=alternate,
        )


def test_discovery_is_explicitly_non_formal_and_has_no_side_effects(
    real_discovery: dict[str, Any],
) -> None:
    bundle = real_discovery["bundle"]
    audit = bundle[discovery.SOURCE_IDEA_AUDIT_FILENAME]
    catalog = bundle[discovery.DISCOVERY_CATALOG_FILENAME]
    collision = bundle[discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME]
    source_node = bundle[discovery.DISCOVERY_SOURCE_NODE_FILENAME]
    state = bundle[discovery.DISCOVERY_CYCLE_STATE_FILENAME]
    report = bundle[discovery.DISCOVERY_READBACK_REPORT_FILENAME]

    assert audit["formal_admission_authority"] is False
    assert audit["statistics_status"] == "not_run"
    assert catalog["readiness"] == "EXPLORATORY_DISCOVERY"
    assert catalog["qualification"] is False
    assert catalog["formal_admission_authority"] is False
    assert catalog["statistics_status"] == "not_run"
    assert catalog["admission_duplicate_primitive_status"] == "not_run"
    assert catalog["high_correlation_dedup_status"] == "not_run"
    assert collision["formal_admission_authority"] is False
    assert collision["admission_duplicate_primitive_status"] == "not_run"
    assert collision["high_correlation_dedup_status"] == "not_run"
    assert source_node["holdout_status"] == "sealed_not_appended"
    assert source_node["qualification"] is False
    assert source_node["formal_admission_authority"] is False
    assert source_node["production_apply_enabled"] is False
    assert state["state"] == "DISCOVERY"
    assert state["holdout_unsealed"] is False
    assert state["predecessor"]["byte_sha256"] == (
        "8f7515692fea0eb90f05e0c4461387b38c73ebae362292ad1109e234bcd105d8"
    )
    assert state["predecessor"]["semantic_sha256"] == (
        "3029ae8d84daa8318c7b5bdc1ec5fdeef9bca889c824acc58237c9850f40892f"
    )
    assert report["readiness"] == "EXPLORATORY_DISCOVERY"
    assert report["qualification"] is False
    assert report["formal_admission_authority"] is False
    assert report["production_apply_enabled"] is False
    assert report["holdout_status"] == "sealed_not_appended"
    assert set(report["measurement_status"]) == set(
        discovery.MEASUREMENT_STATUS_FIELDS
    )
    assert set(report["measurement_status"].values()) == {"not_run"}
    assert report["blockers"] == list(discovery.DISCOVERY_BLOCKERS)
    assert set(report["side_effects"]) == set(discovery.SIDE_EFFECT_FIELDS)
    assert not any(report["side_effects"].values())
