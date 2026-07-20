from __future__ import annotations

import base64
import copy
import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import (
    governance_future_strict_exact_five_eval_v4_4 as evaluator,
)
from quant_investor.factors import (
    governance_future_strict_signal_computability_v4_4 as subject,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reseal(value: dict) -> dict:
    payload = copy.deepcopy(value)
    payload.pop("artifact_semantic_sha256", None)
    payload["artifact_semantic_sha256"] = subject.semantic_sha256_v4_4(payload)
    return payload


def _reseal_engine(value: dict) -> dict:
    payload = copy.deepcopy(value)
    payload.pop("result_semantic_sha256", None)
    payload["result_semantic_sha256"] = subject.semantic_sha256_v4_4(payload)
    return payload


def _reseal_collection(value: dict) -> dict:
    payload = copy.deepcopy(value)
    payload.pop("collection_sha256", None)
    payload["collection_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: item
            for key, item in payload.items()
            if key != "pass_id"
        }
    )
    return payload


def _reseal_operator_program_set(value: dict) -> dict:
    payload = copy.deepcopy(value)
    for program in payload.get("candidates", []):
        program.pop("program_semantic_sha256", None)
        program["program_semantic_sha256"] = subject.semantic_sha256_v4_4(
            program
        )
    payload.pop("artifact_semantic_sha256", None)
    payload["artifact_semantic_sha256"] = subject.semantic_sha256_v4_4(
        payload
    )
    return payload


def _promote_two_pass_engine_authority(value: dict) -> None:
    engine = value["passes"][0]["engines"][0]
    engine["authority"] = True
    value["passes"][0]["engines"][0] = _reseal_engine(engine)


def _insert_two_pass_collection_authority_and_reseal(value: dict) -> None:
    for pass_row in value["passes"]:
        collection = pass_row["collection"]
        collection["strict_source_binding"]["authority"] = True
        collection["collection_sha256"] = subject.semantic_sha256_v4_4(
            {
                key: item
                for key, item in collection.items()
                if key not in {"pass_id", "collection_sha256"}
            }
        )
        for engine_index, engine in enumerate(pass_row["engines"]):
            engine["collection_sha256"] = collection["collection_sha256"]
            pass_row["engines"][engine_index] = _reseal_engine(engine)


def _forge_all_negative_adjusted_descriptors_and_reseal(value: dict) -> None:
    for pass_row in value["passes"]:
        for engine_index, engine in enumerate(pass_row["engines"]):
            candidate = engine["candidates"][1]
            assert candidate["direction"] == -1.0
            candidate["direction_adjusted_matrix"] = copy.deepcopy(
                candidate["raw_matrix"]
            )
            pass_row["engines"][engine_index] = _reseal_engine(engine)


def _bitmap(matrix: np.ndarray, dates: list[str], symbols: list[str]) -> dict:
    values = np.asarray(matrix, dtype=np.uint8, order="C")
    return subject.build_binary_mask_descriptor_v4_4(
        uint8_values=values.tobytes(order="C"),
        dates=dates,
        symbols=symbols,
    )


def _runtime() -> dict:
    return subject.build_runtime_binding_v4_4(
        python_implementation="CPython",
        python_version="3.13.7",
        python_executable_byte_sha256=_h("python"),
        platform_system="Darwin",
        platform_release="25.0.0",
        machine="arm64",
        byteorder="little",
        import_root_tree_sha256=_h("runtime-import-root-tree"),
        distributions=[
            {
                "name": name,
                "version": version,
                "distribution_file_count": 3,
                "distribution_inventory_sha256": _h(name + "-files"),
                "native_binary_count": 1,
                "native_binary_inventory_sha256": _h(name + "-native"),
            }
            for name, version in (
                ("numpy", "2.4.3"),
                ("pandas", "3.0.1"),
                ("pyarrow", "24.0.0"),
                ("python-dateutil", "2.9.0.post0"),
                ("pytz", "2026.1"),
                ("six", "1.17.0"),
            )
        ],
    )


def _graph() -> dict[str, dict]:
    dates = [
        value.date().isoformat()
        for value in pd.bdate_range(end="2026-12-31", periods=90)
    ]
    symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    rows = np.arange(len(dates), dtype=np.float64)[:, None]
    columns = np.arange(len(symbols), dtype=np.float64)[None, :]
    raw_close = 20.0 + rows * 0.07 + columns * 1.3 + np.sin(rows / 9.0)
    raw_open = raw_close * (1.0 + 0.002 * np.cos(rows / 5.0 + columns))
    vol = 1000.0 + rows * 7.0 + columns * 31.0 + (rows % 4.0) * 3.0
    adj_close = raw_close * (1.0 + rows * 0.0002)
    pit = np.ones(raw_close.shape, dtype=bool)
    pit[5, 0] = False
    pit[70, 2] = False
    matrices = {
        "raw_close": raw_close.copy(),
        "raw_open": raw_open.copy(),
        "vol": vol.copy(),
        "adj_close": adj_close.copy(),
    }
    for matrix in matrices.values():
        matrix[~pit] = np.nan

    block = evaluator.build_input_block_v4_4(
        dates=dates,
        symbols=symbols,
        pit_mask=pit,
        **matrices,
    )
    operator_program_set = subject.operator_program_set_v4_4()
    source_outputs = evaluator.evaluate_pandas_engine_v4_4(
        block, operator_program_set=operator_program_set
    )
    local_outputs = evaluator.evaluate_numpy_engine_v4_4(
        block, operator_program_set=operator_program_set
    )
    evaluator.compare_exact_engine_outputs_v4_4(
        source_outputs, local_outputs, block
    )
    halo = evaluator.HALO
    proof_dates = dates[halo:]
    proof_pit = pit[halo:].copy()
    proof_source = {
        name: np.array(value[halo:], dtype=np.float64, order="C", copy=True)
        for name, value in source_outputs.items()
    }
    proof_local = {
        name: np.array(value[halo:], dtype=np.float64, order="C", copy=True)
        for name, value in local_outputs.items()
    }

    runtime = _runtime()
    cutoff = dates[-1]
    snapshot_id = "20261231T160000Z"
    prereg_cycle = subject.deterministic_preregistration_cycle_id_v4_4(
        cutoff=cutoff, snapshot_id=snapshot_id
    )
    strict_source = {
        "strict_source_binding_semantic_sha256": _h("strict-source"),
        "snapshot_manifest_byte_sha256": _h("snapshot"),
        "pit_generation_manifest_byte_sha256": _h("pit-manifest"),
        "pit_membership_byte_sha256": _h("pit-membership"),
        "table_inventory_semantic_sha256": _h("table-inventory"),
        "full_a_scope_count": 2,
        "full_a_scope_sha256": subject.full_a_scope_sha256_v4_4(symbols[:2]),
        "source_calendar_semantic_sha256": (
            subject.source_calendar_semantic_sha256_v4_4(
                dates, cutoff=cutoff
            )
        ),
        "recorded_latest_pointer_byte_sha256": _h("recorded-pointer"),
        "recorded_components_byte_sha256": _h("recorded-components"),
    }
    prereg = {
        "bundle_path": "/private/tmp/v4_4_prereg/" + prereg_cycle,
        "readback_byte_sha256": _h("prereg-readback-byte"),
        "readback_semantic_sha256": _h("prereg-readback-semantic"),
        "artifact_count": 27,
        "cycle_id": prereg_cycle,
        "candidate_rows_semantic_sha256": (
            subject.EXPECTED_CANDIDATE_ROWS_SEMANTIC_SHA256
        ),
    }
    code = [
        {"relative_path": path, "byte_sha256": _h(path)}
        for path in subject.CODE_BINDING_PATHS
    ]
    protected = {name: _h(name) for name in subject.PROTECTED_CONTROL_NAMES}
    manifest = subject.build_input_manifest_v4_4(
        cutoff=cutoff,
        snapshot_id=snapshot_id,
        proof_output_start=dates[halo],
        preregistration=prereg,
        strict_source_expected=strict_source,
        code_binding_set=code,
        runtime_binding_expected_semantic_sha256=runtime[
            "artifact_semantic_sha256"
        ],
        protected_control_expected_sha256=protected,
    )
    input_receipt = subject.build_input_receipt_v4_4(
        manifest=manifest,
        observed_preregistration=prereg,
        observed_code_binding_set=code,
        runtime_binding=runtime,
        observed_protected_control_sha256=protected,
    )
    block_manifest = evaluator.build_block_manifest_v4_4(dates, symbols)
    pit_descriptor = _bitmap(pit, dates, symbols)
    historical_axis = {
        "scope": "all_historical_pit_symbols",
        "cutoff_only": False,
        "contains_all_cutoff_full_a": True,
        "historical_only_symbol_count": 1,
        "hash_algorithm": subject.AXIS_HASH_ALGORITHM,
        "descriptor": copy.deepcopy(pit_descriptor["symbol_axis"]),
    }
    pit_membership = {
        "row_count": 4,
        "distinct_symbol_count": 4,
        "historical_union_symbol_count": 3,
        "duplicate_symbol_count": 0,
        "one_row_per_symbol": True,
        "effective_from_semantics": "inclusive",
        "effective_to_semantics": "exclusive",
        "blank_effective_to_semantics": "positive_infinity",
        "membership_byte_sha256": strict_source["pit_membership_byte_sha256"],
    }
    zero_counts = {field: 0 for field in subject.INPUT_FIELDS}
    data_receipt = subject.build_data_field_receipt_v4_4(
        manifest=manifest,
        input_receipt=input_receipt,
        source_calendar_open_sessions=dates,
        historical_symbol_axis=historical_axis,
        pit_membership_contract=pit_membership,
        pit_mask_descriptor=pit_descriptor,
        block_manifest=block_manifest,
        field_missing_counts=zero_counts,
        bars_outside_pit_interval_count=1,
        ignored_pre_analysis_row_count=0,
        outside_pit_non_null_counts=zero_counts,
        projected_row_count_per_pass=len(dates) * len(symbols),
    )
    proof_pit_descriptor = _bitmap(proof_pit, proof_dates, symbols)
    candidate_masks = [
        _bitmap(~np.isnan(proof_source[name]), proof_dates, symbols)
        for name in evaluator.FACTOR_NAMES
    ]
    candidate_raw_descriptors = {
        name: evaluator.matrix_hash_descriptor_v4_4(
            proof_source[name], dates=proof_dates, symbols=symbols
        )
        for name in evaluator.FACTOR_NAMES
    }
    candidate_non_null_masks = subject.build_candidate_non_null_mask_set_v4_4(
        proof_pit_mask=proof_pit_descriptor,
        candidate_masks=candidate_masks,
        raw_matrix_descriptors=list(candidate_raw_descriptors.values()),
        data_field_receipt=data_receipt,
    )
    input_descriptors = [
        {
            "field": field,
            "descriptor": evaluator.matrix_hash_descriptor_v4_4(
                matrices[field], dates=dates, symbols=symbols
            ),
        }
        for field in subject.INPUT_FIELDS
    ]
    collections = [
        subject.build_collection_descriptor_v4_4(
            pass_id=pass_id,
            data_field_receipt=data_receipt,
            input_matrix_descriptors=input_descriptors,
        )
        for pass_id in subject.PASS_IDS
    ]
    engine_results: list[list[dict]] = []
    for pass_id, collection in zip(subject.PASS_IDS, collections, strict=True):
        pandas_raw = {
            name: evaluator.matrix_hash_descriptor_v4_4(
                proof_source[name], dates=proof_dates, symbols=symbols
            )
            for name in evaluator.FACTOR_NAMES
        }
        pandas_adjusted = {
            definition["name"]: evaluator.matrix_hash_descriptor_v4_4(
                proof_source[definition["name"]]
                * float(definition["direction"]),
                dates=proof_dates,
                symbols=symbols,
            )
            for definition in subject.SOURCE_DEFINITION_BINDINGS
        }
        numpy_raw = {
            name: evaluator.matrix_hash_descriptor_v4_4(
                proof_local[name], dates=proof_dates, symbols=symbols
            )
            for name in evaluator.FACTOR_NAMES
        }
        numpy_adjusted = {
            definition["name"]: evaluator.matrix_hash_descriptor_v4_4(
                proof_local[definition["name"]]
                * float(definition["direction"]),
                dates=proof_dates,
                symbols=symbols,
            )
            for definition in subject.SOURCE_DEFINITION_BINDINGS
        }
        pandas_result = subject.build_engine_pass_result_v4_4(
            engine_id=evaluator.PANDAS_ENGINE_ID,
            pass_id=pass_id,
            collection_sha256=collection["collection_sha256"],
            data_field_receipt=data_receipt,
            operator_program_set=subject.operator_program_set_v4_4(),
            proof_pit_mask=proof_pit_descriptor,
            candidate_non_null_masks=candidate_non_null_masks,
            raw_matrix_descriptors=pandas_raw,
            adjusted_matrix_descriptors=pandas_adjusted,
        )
        numpy_result = subject.build_engine_pass_result_v4_4(
            engine_id=evaluator.NUMPY_ENGINE_ID,
            pass_id=pass_id,
            collection_sha256=collection["collection_sha256"],
            data_field_receipt=data_receipt,
            operator_program_set=subject.operator_program_set_v4_4(),
            proof_pit_mask=proof_pit_descriptor,
            candidate_non_null_masks=candidate_non_null_masks,
            raw_matrix_descriptors=numpy_raw,
            adjusted_matrix_descriptors=numpy_adjusted,
        )
        engine_results.append([pandas_result, numpy_result])
    equivalence = subject.build_two_pass_equivalence_receipt_v4_4(
        manifest=manifest,
        input_receipt=input_receipt,
        data_field_receipt=data_receipt,
        proof_pit_mask=proof_pit_descriptor,
        candidate_non_null_masks=candidate_non_null_masks,
        collections=collections,
        engine_results=engine_results,
    )
    proof = subject.build_proof_v4_4(
        manifest=manifest,
        input_receipt=input_receipt,
        data_field_receipt=data_receipt,
        two_pass_equivalence_receipt=equivalence,
    )
    artifacts = {
        subject.INPUT_MANIFEST_FILENAME: manifest,
        subject.INPUT_RECEIPT_FILENAME: input_receipt,
        subject.DATA_FIELD_RECEIPT_FILENAME: data_receipt,
        subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME: equivalence,
        subject.PROOF_FILENAME: proof,
    }
    bindings = []
    for filename, artifact in artifacts.items():
        raw = subject.canonical_file_bytes_v4_4(artifact)
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "semantic_sha256": (
                    subject.semantic_sha256_v4_4(artifact)
                    if filename == subject.INPUT_MANIFEST_FILENAME
                    else artifact["artifact_semantic_sha256"]
                ),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": 501,
                "nlink": 1,
            }
        )
    readback = subject.build_readback_v4_4(
        run_id=manifest["cycle_id"],
        artifacts=artifacts,
        artifact_bindings=bindings,
    )
    return {
        "runtime": runtime,
        "manifest": manifest,
        "input_receipt": input_receipt,
        "data_receipt": data_receipt,
        "proof_pit_mask": proof_pit_descriptor,
        "candidate_non_null_masks": candidate_non_null_masks,
        "proof_dates": proof_dates,
        "symbols": symbols,
        "equivalence": equivalence,
        "proof": proof,
        "artifacts": artifacts,
        "readback": readback,
    }


@pytest.fixture(scope="module")
def graph() -> dict[str, dict]:
    return _graph()


def test_fixed_inventory_resources_claims_and_runtime(graph: dict[str, dict]) -> None:
    assert subject.PROTOCOL_VERSION == "v4"
    assert subject.EVIDENCE_CONTRACT_VERSION == "v4.4"
    assert subject.FROZEN_PREVIOUS_CUTOFF == "2026-07-19"
    assert subject.PRIVATE_ROOT_SUFFIX[-1] == "v4_4_signal_computability_strict"
    assert len(subject.INPUT_FILENAMES) == 5
    assert subject.BUNDLE_FILENAMES[-1] == subject.READBACK_FILENAME
    assert subject.RESOURCE_CONTRACT["manifest_max_bytes"] == 65_536
    assert subject.RESOURCE_CONTRACT["dense_cell_count_per_block_max"] == 1_540_096
    assert subject.RESOURCE_CONTRACT["halo_session_count"] == 60
    assert subject.RESOURCE_CONTRACT["output_block_session_count"] == 128
    assert subject.RUNTIME_DISTRIBUTION_NAMES == (
        "numpy",
        "pandas",
        "pyarrow",
        "python-dateutil",
        "pytz",
        "six",
    )
    assert subject.POSITIVE_CLAIMS["readiness"] == (
        "NON_AUTHORIZING_STRICT_COMPUTABILITY_ONLY"
    )
    assert graph["runtime"]["execution_mode"] == "sealed_shadow_tree"
    assert graph["runtime"]["isolated_flags"] == ["-B", "-I", "-S"]
    assert graph["runtime"]["import_roots"] == list(subject.RUNTIME_IMPORT_ROOTS)
    assert [row["relative_path"] for row in subject.RUNTIME_IMPORT_ROOTS] == [
        "numpy",
        "numpy-2.4.3.dist-info",
        "pandas",
        "pandas-3.0.1.dist-info",
        "pyarrow",
        "pyarrow-24.0.0.dist-info",
        "dateutil",
        "python_dateutil-2.9.0.post0.dist-info",
        "pytz",
        "pytz-2026.1.post1.dist-info",
        "six-1.17.0.dist-info",
        "six.py",
    ]
    assert subject.validate_runtime_binding_v4_4(graph["runtime"]) == graph["runtime"]


def test_golden_operator_program_set_is_the_exact_schema_and_ir() -> None:
    program_set = subject.operator_program_set_v4_4()
    assert tuple(program_set) == (
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "execution_semantics",
        "candidate_count",
        "candidates",
        "artifact_semantic_sha256",
    )
    assert program_set["candidate_count"] == 5
    assert program_set["artifact_semantic_sha256"] == (
        subject.OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    )
    assert subject.OPERATOR_PROGRAM_SET_SEMANTIC_SHA256 == (
        "49a79bcba2bfe960e3cb2ca9846063c140520d36a8d5567ea60e0bc3d1c04f17"
    )
    semantics = program_set["execution_semantics"]
    assert semantics["execution_partitioning"] == (
        "deterministic_block_local_by_manifest_input_block"
    )
    assert semantics["rolling_state_lifecycle"] == (
        "reset_before_every_manifest_input_block"
    )
    assert semantics["historical_halo_session_count"] == 60
    assert semantics["maximum_output_session_count_per_block"] == 128
    assert semantics["pandas_fixed_window_accumulator_semantics"] == (
        "pandas_3.0.1_within_each_manifest_input_block"
    )
    assert semantics["monolithic_pandas_bit_equivalence_claimed"] is False
    assert semantics["historical_halo_session_count"] == (
        subject.RESOURCE_CONTRACT["halo_session_count"]
    )
    assert semantics["maximum_output_session_count_per_block"] == (
        subject.RESOURCE_CONTRACT["output_block_session_count"]
    )
    assert not hasattr(subject, "OPERATOR_PROGRAM_SCHEMA_VERSION")
    assert tuple(program_set["candidates"][0]) == (
        "order",
        "name",
        "direction",
        "definition_identity_sha256",
        "source_repository",
        "source_commit",
        "source_tree_oid",
        "source_relative_path",
        "source_blob_oid",
        "source_raw_sha256",
        "source_ast_sha256",
        "field_semantics_sha256",
        "field_adapter",
        "nodes",
        "output_node_id",
        "program_semantic_sha256",
    )
    assert [
        [node["opcode"] for node in program["nodes"]]
        for program in program_set["candidates"]
    ] == [
        [
            "source",
            "rolling_min",
            "subtract",
            "rolling_max",
            "subtract",
            "divide",
            "cross_section_rank",
        ],
        [
            "source",
            "source",
            "shift",
            "subtract",
            "constant",
            "add",
            "divide",
            "absolute",
            "rolling_mean",
        ],
        [
            "source",
            "shift",
            "divide",
            "constant",
            "subtract",
            "rolling_std",
            "rolling_std",
            "constant",
            "add",
            "divide",
        ],
        [
            "source",
            "shift",
            "subtract",
            "sign",
            "source",
            "shift",
            "subtract",
            "sign",
            "multiply",
            "rolling_mean",
        ],
        [
            "source",
            "shift",
            "divide",
            "constant",
            "subtract",
            "rolling_std",
            "rolling_std",
        ],
    ]
    for definition, program in zip(
        subject.SOURCE_DEFINITION_BINDINGS,
        program_set["candidates"],
        strict=True,
    ):
        assert type(program["direction"]) is float
        assert definition["operator_program_sha256"] == program[
            "program_semantic_sha256"
        ]
        assert definition["operator_program_set_sha256"] == program_set[
            "artifact_semantic_sha256"
        ]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["candidates"][0].__setitem__("schema_version", "x"),
        lambda value: value["candidates"][0].__setitem__("direction", 1),
        lambda value: value["candidates"][0]["nodes"][1]["parameters"].__setitem__(
            "min_periods", 2
        ),
        lambda value: value["candidates"][1]["nodes"][5]["inputs"].reverse(),
        lambda value: value["candidates"][4]["nodes"][0]["parameters"].__setitem__(
            "canonical_input", "raw_close"
        ),
        lambda value: value["execution_semantics"].__setitem__(
            "pit_remask", "only_at_output"
        ),
        lambda value: value["execution_semantics"].__setitem__(
            "monolithic_pandas_bit_equivalence_claimed", True
        ),
        lambda value: value["execution_semantics"].__setitem__(
            "rolling_state_lifecycle", "carry_across_manifest_blocks"
        ),
    ],
)
def test_any_resealed_operator_program_substitution_is_rejected(mutator) -> None:
    payload = subject.operator_program_set_v4_4()
    mutator(payload)
    payload = _reseal_operator_program_set(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_operator_program_set_v4_4(payload)


def test_engine_pass_builder_is_contract_owned_and_has_one_exact_api() -> None:
    assert not hasattr(evaluator, "build_engine_pass_result_v4_4")
    assert tuple(
        inspect.signature(subject.build_engine_pass_result_v4_4).parameters
    ) == (
        "pass_id",
        "engine_id",
        "collection_sha256",
        "data_field_receipt",
        "operator_program_set",
        "proof_pit_mask",
        "candidate_non_null_masks",
        "raw_matrix_descriptors",
        "adjusted_matrix_descriptors",
    )


def test_exact_graph_validates_and_is_sealed_bundle_only(graph: dict[str, dict]) -> None:
    complete = {**graph["artifacts"], subject.READBACK_FILENAME: graph["readback"]}
    assert tuple(subject.validate_complete_bundle_v4_4(complete)) == (
        subject.BUNDLE_FILENAMES
    )
    assert graph["proof"]["claims"] == subject.POSITIVE_CLAIMS
    assert graph["readback"]["readback_scope"] == "SEALED_BUNDLE_GRAPH_ONLY"
    for field in (
        "external_predecessor_revalidated",
        "immutable_source_revalidated",
        "protected_controls_revalidated",
        "external_state_claimed",
    ):
        assert graph["readback"][field] is False


def test_operator_program_hashes_cross_bind_the_full_artifact_chain(
    graph: dict[str, dict],
) -> None:
    set_sha = subject.OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    data = graph["data_receipt"]
    assert data["operator_program_set"]["artifact_semantic_sha256"] == set_sha
    assert data["operator_program_set_semantic_sha256"] == set_sha
    assert graph["equivalence"]["operator_program_set_semantic_sha256"] == set_sha
    assert graph["proof"]["operator_program_set_semantic_sha256"] == set_sha
    assert graph["readback"]["operator_program_set_semantic_sha256"] == set_sha
    expected_program_shas = [
        program["program_semantic_sha256"]
        for program in data["operator_program_set"]["candidates"]
    ]
    for pass_row in graph["equivalence"]["passes"]:
        assert pass_row["collection"][
            "operator_program_set_semantic_sha256"
        ] == set_sha
        for engine in pass_row["engines"]:
            assert engine["operator_program_set_semantic_sha256"] == set_sha
            assert [
                row["operator_program_semantic_sha256"]
                for row in engine["candidates"]
            ] == expected_program_shas


def test_every_operator_program_chain_substitution_fails_closed(
    graph: dict[str, dict],
) -> None:
    zero = "0" * 64

    data = copy.deepcopy(graph["data_receipt"])
    data["operator_program_set_semantic_sha256"] = zero
    data = _reseal(data)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_data_field_receipt_v4_4(
            data,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
        )

    pass_row = graph["equivalence"]["passes"][0]
    collection = copy.deepcopy(pass_row["collection"])
    collection["operator_program_set_semantic_sha256"] = zero
    collection["collection_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in collection.items()
            if key not in {"pass_id", "collection_sha256"}
        }
    )
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_collection_descriptor_v4_4(
            collection,
            pass_id=pass_row["pass_id"],
            data_field_receipt=graph["data_receipt"],
        )

    engine = copy.deepcopy(pass_row["engines"][0])
    engine["operator_program_set_semantic_sha256"] = zero
    engine = _reseal_engine(engine)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_engine_pass_result_v4_4(
            engine,
            pass_id=pass_row["pass_id"],
            engine_id=subject.ENGINE_IDS[0],
            collection_sha256=pass_row["collection"]["collection_sha256"],
            data_field_receipt=graph["data_receipt"],
            proof_pit_mask=graph["proof_pit_mask"],
            candidate_non_null_masks=graph["candidate_non_null_masks"],
        )

    candidate = copy.deepcopy(pass_row["engines"][0])
    candidate["candidates"][0]["operator_program_semantic_sha256"] = zero
    candidate = _reseal_engine(candidate)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_engine_pass_result_v4_4(
            candidate,
            pass_id=pass_row["pass_id"],
            engine_id=subject.ENGINE_IDS[0],
            collection_sha256=pass_row["collection"]["collection_sha256"],
            data_field_receipt=graph["data_receipt"],
            proof_pit_mask=graph["proof_pit_mask"],
            candidate_non_null_masks=graph["candidate_non_null_masks"],
        )

    equivalence = copy.deepcopy(graph["equivalence"])
    equivalence["operator_program_set_semantic_sha256"] = zero
    equivalence = _reseal(equivalence)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            equivalence,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )

    proof = copy.deepcopy(graph["proof"])
    proof["operator_program_set_semantic_sha256"] = zero
    proof = _reseal(proof)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_proof_v4_4(
            proof,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
            two_pass_equivalence_receipt=graph["equivalence"],
        )

    readback = copy.deepcopy(graph["readback"])
    readback["operator_program_set_semantic_sha256"] = zero
    readback = _reseal(readback)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_readback_v4_4(
            readback, artifacts=graph["artifacts"]
        )


def test_private_bundle_contract_and_base_binding_normalization(
    graph: dict[str, dict],
) -> None:
    bundle_contract = subject.private_bundle_contract_v4_4()
    assert bundle_contract.root_suffix == subject.PRIVATE_ROOT_SUFFIX
    assert bundle_contract.input_filenames == subject.INPUT_FILENAMES
    assert bundle_contract.readback_report_filename == subject.READBACK_REPORT_FILENAME
    assert bundle_contract.max_artifact_bytes == 16 * 1024 * 1024
    assert bundle_contract.max_bundle_bytes == 64 * 1024 * 1024
    for filename, artifact in graph["artifacts"].items():
        assert bundle_contract.validate_artifact(filename, artifact) == artifact

    base_bindings = [
        {
            key: value
            for key, value in row.items()
            if key != "semantic_sha256"
        }
        for row in graph["readback"]["artifact_bindings"]
    ]
    rebuilt = bundle_contract.build_readback_report(
        run_id=graph["manifest"]["cycle_id"],
        artifacts=graph["artifacts"],
        artifact_bindings=base_bindings,
    )
    assert rebuilt == graph["readback"]


@pytest.mark.parametrize(
    ("filename", "mutator"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            lambda value: value["stage1_claims"].__setitem__(
                "external_state_authority_claimed", True
            ),
        ),
        (
            subject.INPUT_RECEIPT_FILENAME,
            lambda value: value["resource_contract"].__setitem__(
                "halo_session_count", 59
            ),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            lambda value: value.__setitem__(
                "strict_source_evidence_status", "HEALTHY_SOURCE_AUTHORIZED"
            ),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            lambda value: value["source_access"].__setitem__(
                "serving_read", True
            ),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            lambda value: value["negative_claims"]["authority"].__setitem__(
                "new_risk_authorized", True
            ),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            lambda value: value["claims"].__setitem__(
                "independent_engine_equivalence", "production_authority"
            ),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            _promote_two_pass_engine_authority,
        ),
        (
            subject.PROOF_FILENAME,
            lambda value: value["claims"].__setitem__(
                "readiness", "PRODUCTION_AUTHORIZED"
            ),
        ),
        (
            subject.PROOF_FILENAME,
            lambda value: value["selection_disclosures"].__setitem__(
                "outcome_informed_selection", False
            ),
        ),
        (
            subject.READBACK_FILENAME,
            lambda value: value.__setitem__("external_state_claimed", True),
        ),
        (
            subject.READBACK_FILENAME,
            lambda value: value.__setitem__(
                "strict_source_evidence_status", "PRODUCTION_HEALTHY"
            ),
        ),
    ],
)
def test_standalone_artifact_validation_rejects_resealed_claim_promotion(
    graph: dict[str, dict], filename: str, mutator
) -> None:
    source = (
        graph["readback"]
        if filename == subject.READBACK_FILENAME
        else graph["artifacts"][filename]
    )
    payload = copy.deepcopy(source)
    mutator(payload)
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_artifact_v4_4(filename, payload)


def test_standalone_input_receipt_rejects_resealed_manifest_binding_authority(
    graph: dict[str, dict],
) -> None:
    payload = copy.deepcopy(graph["artifacts"][subject.INPUT_RECEIPT_FILENAME])
    payload["input_manifest_binding"]["authority"] = True
    payload = _reseal(payload)
    assert payload["artifact_semantic_sha256"] == subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_semantic_sha256"
        }
    )
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="input receipt manifest binding fields invalid",
    ):
        subject.validate_artifact_v4_4(subject.INPUT_RECEIPT_FILENAME, payload)


@pytest.mark.parametrize(
    ("filename", "path", "match"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            ("preregistration",),
            "preregistration fields invalid",
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            ("source_calendar",),
            "source calendar fields invalid",
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            ("historical_symbol_axis",),
            "historical symbol axis fields invalid",
        ),
    ],
)
def test_standalone_artifact_validation_rejects_resealed_copied_object_authority(
    graph: dict[str, dict], filename: str, path: tuple[str, ...], match: str
) -> None:
    payload = copy.deepcopy(graph["artifacts"][filename])
    target = payload
    for key in path:
        target = target[key]
    target["authority"] = True
    payload = _reseal(payload)
    assert payload["artifact_semantic_sha256"] == subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_semantic_sha256"
        }
    )
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match=match,
    ):
        subject.validate_artifact_v4_4(filename, payload)


@pytest.mark.parametrize(
    ("filename", "mutator"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            lambda value: value["strict_source_expected"].__setitem__(
                "authority", True
            ),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            lambda value: value["strict_source_binding"].__setitem__(
                "authority", True
            ),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            _insert_two_pass_collection_authority_and_reseal,
        ),
        (
            subject.PROOF_FILENAME,
            lambda value: value["predecessor_bindings"].__setitem__(
                "authority", True
            ),
        ),
        (
            subject.READBACK_FILENAME,
            lambda value: value["artifact_bindings"][0].__setitem__(
                "authority", True
            ),
        ),
    ],
)
def test_standalone_validator_rejects_unknown_nested_authority_insertions_after_reseal(
    graph: dict[str, dict], filename: str, mutator
) -> None:
    source = (
        graph["readback"]
        if filename == subject.READBACK_FILENAME
        else graph["artifacts"][filename]
    )
    payload = copy.deepcopy(source)
    mutator(payload)
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="fields invalid",
    ):
        subject.validate_artifact_v4_4(filename, payload)


def test_standalone_two_pass_rejects_forged_raw_adjusted_matrix_descriptor_relation(
    graph: dict[str, dict],
) -> None:
    payload = copy.deepcopy(graph["equivalence"])
    _forge_all_negative_adjusted_descriptors_and_reseal(payload)
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="negative-direction",
    ):
        subject.validate_artifact_v4_4(
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME, payload
        )


@pytest.mark.parametrize("standalone", [False, True])
def test_resealed_two_pass_rejects_bool_candidate_order(
    graph: dict[str, dict], standalone: bool
) -> None:
    payload = copy.deepcopy(graph["equivalence"])
    for pass_row in payload["passes"]:
        for engine_index, engine in enumerate(pass_row["engines"]):
            engine["candidates"][0]["order"] = True
            pass_row["engines"][engine_index] = _reseal_engine(engine)
    payload = _reseal(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="candidate 1 order must be a positive integer",
    ):
        if standalone:
            subject.validate_artifact_v4_4(
                subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME, payload
            )
        else:
            subject.validate_two_pass_equivalence_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
                data_field_receipt=graph["data_receipt"],
            )


@pytest.mark.parametrize(
    ("field", "forged_value", "match"),
    [
        ("order", True, "mask 1 order must be a positive integer"),
        (
            "outside_pit_non_null_count",
            False,
            "mask 1 outside-PIT count must be a non-negative integer",
        ),
    ],
)
def test_standalone_candidate_mask_rejects_bool_integer_after_reseal(
    graph: dict[str, dict], field: str, forged_value: bool, match: str
) -> None:
    payload = copy.deepcopy(graph["equivalence"])
    mask_set = payload["candidate_non_null_masks"]
    mask_set["rows"][0][field] = forged_value
    mask_set["set_semantic_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in mask_set.items()
            if key != "set_semantic_sha256"
        }
    )
    payload = _reseal(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match=match,
    ):
        subject.validate_artifact_v4_4(
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME, payload
        )


def test_resealed_operator_program_set_rejects_integral_float_candidate_count() -> None:
    payload = subject.operator_program_set_v4_4()
    payload["candidate_count"] = 5.0
    payload = _reseal_operator_program_set(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="operator program-set candidate count must be a positive integer",
    ):
        subject.validate_operator_program_set_v4_4(payload)


@pytest.mark.parametrize("standalone", [False, True])
def test_preregistration_rejects_integral_float_artifact_count(
    graph: dict[str, dict], standalone: bool
) -> None:
    if standalone:
        payload = copy.deepcopy(graph["input_receipt"])
        payload["preregistration"]["artifact_count"] = 27.0
        payload = _reseal(payload)
    else:
        payload = copy.deepcopy(graph["manifest"])
        payload["preregistration"]["artifact_count"] = 27.0

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="preregistration artifact count must be a positive integer",
    ):
        if standalone:
            subject.validate_artifact_v4_4(
                subject.INPUT_RECEIPT_FILENAME, payload
            )
        else:
            subject.validate_input_manifest_v4_4(payload)


@pytest.mark.parametrize("standalone", [False, True])
def test_resealed_cutoff_scope_rejects_integral_float_count(
    graph: dict[str, dict], standalone: bool
) -> None:
    payload = copy.deepcopy(graph["data_receipt"])
    payload["cutoff_full_a_scope"]["count"] = 2.0
    payload = _reseal(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="cutoff full-A scope count must be a positive integer",
    ):
        if standalone:
            subject.validate_artifact_v4_4(
                subject.DATA_FIELD_RECEIPT_FILENAME, payload
            )
        else:
            subject.validate_data_field_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
            )


@pytest.mark.parametrize("standalone", [False, True])
def test_resealed_candidate_mask_set_rejects_integral_float_count(
    graph: dict[str, dict], standalone: bool
) -> None:
    payload = copy.deepcopy(graph["equivalence"])
    mask_set = payload["candidate_non_null_masks"]
    mask_set["candidate_count"] = 5.0
    mask_set["set_semantic_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in mask_set.items()
            if key != "set_semantic_sha256"
        }
    )
    payload = _reseal(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="candidate non-null mask set candidate count must be a positive integer",
    ):
        if standalone:
            subject.validate_artifact_v4_4(
                subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME, payload
            )
        else:
            subject.validate_two_pass_equivalence_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
                data_field_receipt=graph["data_receipt"],
            )


def test_resealed_contextual_collection_rejects_integral_float_projected_rows(
    graph: dict[str, dict],
) -> None:
    pass_row = graph["equivalence"]["passes"][0]
    payload = copy.deepcopy(pass_row["collection"])
    payload["projected_row_count_per_pass"] = float(
        payload["projected_row_count_per_pass"]
    )
    payload = _reseal_collection(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="collection projected row count must be a positive integer",
    ):
        subject.validate_collection_descriptor_v4_4(
            payload,
            pass_id=pass_row["pass_id"],
            data_field_receipt=graph["data_receipt"],
        )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("row_count", "row count must be a positive integer"),
        ("column_count", "column count must be a positive integer"),
    ],
)
@pytest.mark.parametrize("standalone", [False, True])
def test_resealed_engine_rejects_integral_float_shape(
    graph: dict[str, dict], field: str, match: str, standalone: bool
) -> None:
    if standalone:
        payload = copy.deepcopy(graph["equivalence"])
        for pass_row in payload["passes"]:
            for index, engine in enumerate(pass_row["engines"]):
                engine[field] = float(engine[field])
                pass_row["engines"][index] = _reseal_engine(engine)
        payload = _reseal(payload)
    else:
        pass_row = graph["equivalence"]["passes"][0]
        payload = copy.deepcopy(pass_row["engines"][0])
        payload[field] = float(payload[field])
        payload = _reseal_engine(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match=match,
    ):
        if standalone:
            subject.validate_artifact_v4_4(
                subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME, payload
            )
        else:
            subject.validate_engine_pass_result_v4_4(
                payload,
                pass_id=pass_row["pass_id"],
                engine_id=subject.ENGINE_IDS[0],
                collection_sha256=pass_row["collection"]["collection_sha256"],
                data_field_receipt=graph["data_receipt"],
                proof_pit_mask=graph["proof_pit_mask"],
                candidate_non_null_masks=graph["candidate_non_null_masks"],
            )


@pytest.mark.parametrize(
    ("artifact_name", "fixed_field"),
    [
        ("data_receipt", "selection_disclosures"),
        ("data_receipt", "negative_claims"),
        ("equivalence", "claims"),
        ("equivalence", "selection_disclosures"),
        ("equivalence", "negative_claims"),
        ("proof", "claims"),
        ("proof", "selection_disclosures"),
        ("proof", "negative_claims"),
        ("readback", "claims"),
        ("readback", "selection_disclosures"),
        ("readback", "negative_claims"),
    ],
)
def test_resealed_contextual_fixed_claims_reject_integer_boolean_alias(
    graph: dict[str, dict], artifact_name: str, fixed_field: str
) -> None:
    payload = copy.deepcopy(graph[artifact_name])
    if fixed_field == "claims":
        payload[fixed_field]["exact_five_atomic"] = 1
    elif fixed_field == "selection_disclosures":
        payload[fixed_field]["outcome_informed_selection"] = 1
    else:
        payload[fixed_field]["authority"]["measurement_authorized"] = 0
    payload = _reseal(payload)

    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="differs from the immutable contract",
    ):
        if artifact_name == "data_receipt":
            subject.validate_data_field_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
            )
        elif artifact_name == "equivalence":
            subject.validate_two_pass_equivalence_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
                data_field_receipt=graph["data_receipt"],
            )
        elif artifact_name == "proof":
            subject.validate_proof_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
                data_field_receipt=graph["data_receipt"],
                two_pass_equivalence_receipt=graph["equivalence"],
            )
        else:
            subject.validate_readback_v4_4(
                payload, artifacts=graph["artifacts"]
            )


@pytest.mark.parametrize(
    "filename",
    [
        subject.INPUT_RECEIPT_FILENAME,
        subject.DATA_FIELD_RECEIPT_FILENAME,
        subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
        subject.PROOF_FILENAME,
    ],
)
@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("cutoff", "2026-12-31T00:00:00"),
        ("proof_output_start", "2027-01-01"),
        ("snapshot_id", "20261230T160000Z"),
        ("cycle_id", "cn_full_a_v4_4_strict_computability_forged"),
    ],
)
def test_standalone_identity_tuple_rejects_resealed_forgery(
    graph: dict[str, dict], filename: str, field: str, forged_value: str
) -> None:
    payload = copy.deepcopy(graph["artifacts"][filename])
    payload[field] = forged_value
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.validate_artifact_v4_4(filename, payload)


@pytest.mark.parametrize(
    ("filename", "path"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            ("input_manifest_binding", "filename"),
        ),
        (
            subject.PROOF_FILENAME,
            ("predecessor_bindings", "input_manifest", "filename"),
        ),
    ],
)
def test_standalone_input_manifest_binding_filename_is_exact(
    graph: dict[str, dict], filename: str, path: tuple[str, ...]
) -> None:
    payload = copy.deepcopy(graph["artifacts"][filename])
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "forged_input_manifest.json"
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="manifest.*filename",
    ):
        subject.validate_artifact_v4_4(filename, payload)


@pytest.mark.parametrize(
    ("filename", "path"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            ("input_manifest_binding", "semantic_sha256"),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            ("input_manifest_semantic_sha256",),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            ("input_receipt_semantic_sha256",),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            ("input_manifest_semantic_sha256",),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            ("input_receipt_semantic_sha256",),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            ("data_field_receipt_semantic_sha256",),
        ),
        (
            subject.PROOF_FILENAME,
            ("predecessor_bindings", "input_manifest", "byte_sha256"),
        ),
        (
            subject.PROOF_FILENAME,
            ("predecessor_bindings", "input_manifest", "semantic_sha256"),
        ),
        (
            subject.PROOF_FILENAME,
            (
                "predecessor_bindings",
                "preregistration_readback",
                "byte_sha256",
            ),
        ),
        (
            subject.PROOF_FILENAME,
            (
                "predecessor_bindings",
                "preregistration_readback",
                "semantic_sha256",
            ),
        ),
        (
            subject.PROOF_FILENAME,
            ("predecessor_bindings", "input_receipt_semantic_sha256"),
        ),
        (
            subject.PROOF_FILENAME,
            ("predecessor_bindings", "data_field_receipt_semantic_sha256"),
        ),
        (
            subject.PROOF_FILENAME,
            (
                "predecessor_bindings",
                "two_pass_equivalence_receipt_semantic_sha256",
            ),
        ),
    ],
)
def test_standalone_predecessor_sha_shapes_reject_resealed_forgery(
    graph: dict[str, dict], filename: str, path: tuple[str, ...]
) -> None:
    payload = copy.deepcopy(graph["artifacts"][filename])
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "not-a-sha"
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="SHA",
    ):
        subject.validate_artifact_v4_4(filename, payload)


@pytest.mark.parametrize(
    ("filename", "path"),
    [
        (
            subject.INPUT_RECEIPT_FILENAME,
            ("strict_source_expected", "strict_source_binding_semantic_sha256"),
        ),
        (
            subject.DATA_FIELD_RECEIPT_FILENAME,
            ("strict_source_binding", "strict_source_binding_semantic_sha256"),
        ),
        (
            subject.TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
            (
                "passes",
                0,
                "collection",
                "strict_source_binding",
                "strict_source_binding_semantic_sha256",
            ),
        ),
        (
            subject.PROOF_FILENAME,
            ("strict_source_binding_semantic_sha256",),
        ),
    ],
)
def test_standalone_strict_source_sha_shapes_reject_resealed_forgery(
    graph: dict[str, dict], filename: str, path: tuple[str | int, ...]
) -> None:
    payload = copy.deepcopy(graph["artifacts"][filename])
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "not-a-sha"
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="SHA",
    ):
        subject.validate_artifact_v4_4(filename, payload)


def test_standalone_proof_preregistration_cycle_is_deterministic(
    graph: dict[str, dict],
) -> None:
    payload = copy.deepcopy(graph["proof"])
    payload["predecessor_bindings"]["preregistration_readback"][
        "cycle_id"
    ] = "safe_but_forged_preregistration_cycle"
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="preregistration cycle_id",
    ):
        subject.validate_artifact_v4_4(subject.PROOF_FILENAME, payload)


@pytest.mark.parametrize(
    ("run_id", "cycle_id"),
    [
        ("../unsafe", None),
        ("safe_but_wrong_run", None),
        ("safe_but_forged_cycle", "safe_but_forged_cycle"),
    ],
)
def test_standalone_readback_run_identity_is_safe_and_deterministic(
    graph: dict[str, dict], run_id: str, cycle_id: str | None
) -> None:
    payload = copy.deepcopy(graph["readback"])
    payload["run_id"] = run_id
    if cycle_id is not None:
        payload["cycle_id"] = cycle_id
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="cycle|run",
    ):
        subject.validate_artifact_v4_4(subject.READBACK_FILENAME, payload)


def test_standalone_readback_proof_semantic_sha_shape_is_exact(
    graph: dict[str, dict],
) -> None:
    payload = copy.deepcopy(graph["readback"])
    payload["proof_semantic_sha256"] = "not-a-sha"
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="proof semantic SHA",
    ):
        subject.validate_artifact_v4_4(subject.READBACK_FILENAME, payload)


def test_manifest_rejects_missing_unknown_and_nested_unknown_fields(
    graph: dict[str, dict],
) -> None:
    missing = copy.deepcopy(graph["manifest"])
    missing.pop("snapshot_id")
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_input_manifest_v4_4(missing)

    unknown = copy.deepcopy(graph["manifest"])
    unknown["unknown"] = False
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_input_manifest_v4_4(unknown)

    nested = copy.deepcopy(graph["manifest"])
    nested["preregistration"]["labels"] = []
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_input_manifest_v4_4(nested)


def test_future_cutoff_snapshot_and_deterministic_cycle_fail_closed(
    graph: dict[str, dict],
) -> None:
    for field, value in (
        ("cutoff", "2026-07-19"),
        ("snapshot_id", "20261230T160000Z"),
        ("cycle_id", "cn_full_a_v4_4_wrong"),
    ):
        payload = copy.deepcopy(graph["manifest"])
        payload[field] = value
        with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
            subject.validate_input_manifest_v4_4(payload)
    assert graph["manifest"]["cycle_id"].startswith(
        "cn_full_a_v4_4_strict_computability_"
    )
    assert graph["manifest"]["preregistration"]["cycle_id"].startswith(
        "cn_full_a_v4_4_2026"
    )


@pytest.mark.parametrize(
    ("section", "mutator"),
    [
        ("resource_contract", lambda value: value.__setitem__("halo_session_count", 59)),
        ("code_binding_set", lambda value: value.reverse()),
        (
            "source_definition_bindings",
            lambda value: value[0].__setitem__("direction", -1),
        ),
    ],
)
def test_manifest_rejects_resource_code_source_and_control_mutation(
    graph: dict[str, dict], section: str, mutator
) -> None:
    payload = copy.deepcopy(graph["manifest"])
    mutator(payload[section])
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_input_manifest_v4_4(payload)


def test_observed_protected_control_drift_is_rejected(graph: dict[str, dict]) -> None:
    observed = copy.deepcopy(
        graph["manifest"]["protected_control_expected_sha256"]
    )
    observed["registry"] = "0" * 64
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.build_input_receipt_v4_4(
            manifest=graph["manifest"],
            observed_preregistration=graph["manifest"]["preregistration"],
            observed_code_binding_set=graph["manifest"]["code_binding_set"],
            runtime_binding=graph["runtime"],
            observed_protected_control_sha256=observed,
        )


def test_input_receipt_builder_requires_the_validated_runtime_object(
    graph: dict[str, dict],
) -> None:
    parameters = inspect.signature(subject.build_input_receipt_v4_4).parameters
    assert "runtime_binding" in parameters
    assert "observed_runtime_binding_semantic_sha256" not in parameters
    assert graph["input_receipt"]["runtime_binding_semantic_sha256"] == graph[
        "runtime"
    ]["artifact_semantic_sha256"]

    base_kwargs = {
        "manifest": graph["manifest"],
        "observed_preregistration": graph["manifest"]["preregistration"],
        "observed_code_binding_set": graph["manifest"]["code_binding_set"],
        "observed_protected_control_sha256": graph["manifest"][
            "protected_control_expected_sha256"
        ],
    }
    for invalid_runtime in (
        graph["manifest"]["runtime_binding_expected_semantic_sha256"],
        {
            "artifact_semantic_sha256": graph["manifest"][
                "runtime_binding_expected_semantic_sha256"
            ]
        },
    ):
        with pytest.raises(
            subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
        ):
            subject.build_input_receipt_v4_4(
                **base_kwargs, runtime_binding=invalid_runtime
            )

    valid_but_different = copy.deepcopy(graph["runtime"])
    valid_but_different["python_version"] = "3.13.8"
    valid_but_different = _reseal(valid_but_different)
    assert (
        subject.validate_runtime_binding_v4_4(valid_but_different)
        == valid_but_different
    )
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="differs from stage-1 manifest",
    ):
        subject.build_input_receipt_v4_4(
            **base_kwargs, runtime_binding=valid_but_different
        )


def test_full_a_sha_has_no_trailing_newline_and_is_not_axis_hash() -> None:
    symbols = ["000002.SZ", "000001.SZ"]
    expected = hashlib.sha256(b"000001.SZ\n000002.SZ").hexdigest()
    trailing = hashlib.sha256(b"000001.SZ\n000002.SZ\n").hexdigest()
    assert subject.full_a_scope_sha256_v4_4(symbols) == expected
    assert expected != trailing
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.full_a_scope_sha256_v4_4(["000001.SZ", "000001.SZ"])


def test_cutoff_only_symbol_axis_and_duplicate_pit_symbol_are_rejected(
    graph: dict[str, dict],
) -> None:
    cutoff_only = copy.deepcopy(graph["data_receipt"])
    cutoff_only["historical_symbol_axis"]["historical_only_symbol_count"] = 0
    cutoff_only = _reseal(cutoff_only)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_data_field_receipt_v4_4(
            cutoff_only,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
        )

    duplicate = copy.deepcopy(graph["data_receipt"])
    duplicate["pit_membership_contract"]["duplicate_symbol_count"] = 1
    duplicate = _reseal(duplicate)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_data_field_receipt_v4_4(
            duplicate,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
        )


def test_calendar_halo_table_projection_and_source_access_are_exact(
    graph: dict[str, dict],
) -> None:
    for mutate in (
        lambda data: data.__setitem__("proof_output_start", data["cutoff"]),
        lambda data: data["table_projection"].reverse(),
        lambda data: data["source_access"].__setitem__("current_pointer_read", True),
        lambda data: data["outside_pit_non_null_counts"].__setitem__("vol", 1),
    ):
        payload = copy.deepcopy(graph["data_receipt"])
        mutate(payload)
        payload = _reseal(payload)
        with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
            subject.validate_data_field_receipt_v4_4(
                payload,
                manifest=graph["manifest"],
                input_receipt=graph["input_receipt"],
            )


def test_resealed_block_manifest_row_drift_is_rejected(
    graph: dict[str, dict],
) -> None:
    payload = copy.deepcopy(graph["data_receipt"])
    block_manifest = payload["block_manifest"]
    block_manifest["blocks"][0]["future_halo_row_count"] = 1
    block_manifest["manifest_semantic_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in block_manifest.items()
            if key != "manifest_semantic_sha256"
        }
    )
    payload = _reseal(payload)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="exact-once past-only",
    ):
        subject.validate_data_field_receipt_v4_4(
            payload,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
        )


def test_nonbinary_pit_and_nonzero_padding_are_rejected(
    graph: dict[str, dict],
) -> None:
    legacy = copy.deepcopy(graph["data_receipt"])
    shape = (
        legacy["historical_date_axis_descriptor"]["count"],
        legacy["historical_symbol_axis"]["descriptor"]["count"],
    )
    legacy["pit_mask_descriptor"] = evaluator.matrix_hash_descriptor_v4_4(
        np.full(shape, 0.5, dtype=np.float64),
        dates=legacy["source_calendar"]["open_sessions"],
        symbols=legacy["block_manifest"]["full_historical_symbols"],
    )
    legacy = _reseal(legacy)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_data_field_receipt_v4_4(
            legacy,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
        )

    descriptor = copy.deepcopy(graph["proof_pit_mask"])
    packed = bytearray(base64.b64decode(descriptor["packed_bits_base64"]))
    assert descriptor["padding_bit_count"] > 0
    packed[-1] |= 0x80
    descriptor["packed_bits_base64"] = base64.b64encode(packed).decode("ascii")
    descriptor["packed_bits_sha256"] = hashlib.sha256(packed).hexdigest()
    descriptor["one_count"] += 1
    descriptor["zero_count"] -= 1
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="padding bits",
    ):
        subject.validate_binary_mask_descriptor_v4_4(descriptor)


def test_resealed_candidate_non_null_bitmap_leak_is_rejected(
    graph: dict[str, dict],
) -> None:
    equivalence = copy.deepcopy(graph["equivalence"])
    proof_pit = equivalence["proof_pit_mask"]
    pit_bytes = base64.b64decode(proof_pit["packed_bits_base64"])
    outside_bit = next(
        index
        for index in range(proof_pit["bit_count"])
        if not (pit_bytes[index // 8] >> (index % 8)) & 1
    )
    row = equivalence["candidate_non_null_masks"]["rows"][0]
    candidate_bytes = bytearray(
        base64.b64decode(row["mask"]["packed_bits_base64"])
    )
    candidate_bytes[outside_bit // 8] |= 1 << (outside_bit % 8)
    row["mask"] = subject.build_packed_binary_mask_descriptor_v4_4(
        packed_bits=bytes(candidate_bytes),
        bit_count=proof_pit["bit_count"],
        dates=graph["proof_dates"],
        symbols=graph["symbols"],
    )
    mask_set = equivalence["candidate_non_null_masks"]
    mask_set["set_semantic_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in mask_set.items()
            if key != "set_semantic_sha256"
        }
    )
    equivalence = _reseal(equivalence)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="outside PIT",
    ):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            equivalence,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )


def test_authority_claim_and_disclosure_promotion_are_rejected(
    graph: dict[str, dict],
) -> None:
    promoted = copy.deepcopy(graph["proof"])
    promoted["negative_claims"]["authority"]["healthy_source_receipt"] = True
    promoted = _reseal(promoted)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_proof_v4_4(
            promoted,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
            two_pass_equivalence_receipt=graph["equivalence"],
        )

    disclosure = copy.deepcopy(graph["proof"])
    disclosure["selection_disclosures"]["outcome_informed_selection"] = False
    disclosure = _reseal(disclosure)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_proof_v4_4(
            disclosure,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
            two_pass_equivalence_receipt=graph["equivalence"],
        )


def test_engine_and_pass_drift_are_rejected(graph: dict[str, dict]) -> None:
    engine_drift = copy.deepcopy(graph["equivalence"])
    engine = engine_drift["passes"][0]["engines"][1]
    engine["candidates"][0]["name"] = "wrong"
    engine_drift["passes"][0]["engines"][1] = _reseal_engine(engine)
    engine_drift = _reseal(engine_drift)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            engine_drift,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )

    pass_drift = copy.deepcopy(graph["equivalence"])
    collection = pass_drift["passes"][1]["collection"]
    descriptor = collection["input_matrix_descriptors"][0]["descriptor"]
    descriptor["matrix_sha256"] = _h("fresh-pass-drift")
    descriptor["bit_pattern_sha256"] = descriptor["matrix_sha256"]
    collection["collection_sha256"] = subject.semantic_sha256_v4_4(
        {
            key: value
            for key, value in collection.items()
            if key not in {"pass_id", "collection_sha256"}
        }
    )
    for index, engine in enumerate(pass_drift["passes"][1]["engines"]):
        engine["collection_sha256"] = collection["collection_sha256"]
        pass_drift["passes"][1]["engines"][index] = _reseal_engine(engine)
    pass_drift = _reseal(pass_drift)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            pass_drift,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )


def test_old_engine_result_shape_is_rejected(graph: dict[str, dict]) -> None:
    pass_row = graph["equivalence"]["passes"][0]
    legacy = copy.deepcopy(pass_row["engines"][0])
    legacy.pop("proof_pit_mask_semantic_sha256")
    legacy.pop("operator_program_set_semantic_sha256")
    legacy["pit_mask"] = copy.deepcopy(graph["proof_pit_mask"])
    for row in legacy["candidates"]:
        row.pop("operator_program_semantic_sha256")
    legacy = _reseal_engine(legacy)
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="fields invalid",
    ):
        subject.validate_engine_pass_result_v4_4(
            legacy,
            pass_id=pass_row["pass_id"],
            engine_id=subject.ENGINE_IDS[0],
            collection_sha256=pass_row["collection"]["collection_sha256"],
            data_field_receipt=graph["data_receipt"],
            proof_pit_mask=graph["proof_pit_mask"],
            candidate_non_null_masks=graph["candidate_non_null_masks"],
        )


@pytest.mark.parametrize("candidate_index", [0, 1])
def test_forged_direction_adjusted_descriptor_is_rejected(
    graph: dict[str, dict], candidate_index: int
) -> None:
    pass_row = graph["equivalence"]["passes"][0]
    engine = pass_row["engines"][0]
    raw = {
        row["name"]: copy.deepcopy(row["raw_matrix"])
        for row in engine["candidates"]
    }
    adjusted = {
        row["name"]: copy.deepcopy(row["direction_adjusted_matrix"])
        for row in engine["candidates"]
    }
    forged_name = engine["candidates"][candidate_index]["name"]
    if candidate_index == 0:
        adjusted[forged_name] = copy.deepcopy(
            engine["candidates"][1]["direction_adjusted_matrix"]
        )
    else:
        adjusted[forged_name] = copy.deepcopy(raw[forged_name])
    with pytest.raises(
        subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error
    ):
        subject.build_engine_pass_result_v4_4(
            pass_id=pass_row["pass_id"],
            engine_id=subject.ENGINE_IDS[0],
            collection_sha256=pass_row["collection"]["collection_sha256"],
            data_field_receipt=graph["data_receipt"],
            operator_program_set=subject.operator_program_set_v4_4(),
            proof_pit_mask=graph["proof_pit_mask"],
            candidate_non_null_masks=graph["candidate_non_null_masks"],
            raw_matrix_descriptors=raw,
            adjusted_matrix_descriptors=adjusted,
        )


def test_cross_hash_tamper_is_rejected(graph: dict[str, dict]) -> None:
    payload = copy.deepcopy(graph["proof"])
    payload["predecessor_bindings"][
        "data_field_receipt_semantic_sha256"
    ] = "0" * 64
    payload = _reseal(payload)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_proof_v4_4(
            payload,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
            two_pass_equivalence_receipt=graph["equivalence"],
        )


def test_readback_scope_negatives_and_file_identity_are_exact(
    graph: dict[str, dict],
) -> None:
    for mutate in (
        lambda value: value.__setitem__("readback_scope", "CURRENT_STATE"),
        lambda value: value.__setitem__("immutable_source_revalidated", True),
        lambda value: value["artifact_bindings"][0].__setitem__("mode", 0o644),
        lambda value: value["artifact_bindings"][1].__setitem__("nlink", 2),
    ):
        payload = copy.deepcopy(graph["readback"])
        mutate(payload)
        payload = _reseal(payload)
        with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
            subject.validate_readback_v4_4(payload, artifacts=graph["artifacts"])


def test_banned_label_outcome_and_unknown_engine_fields_are_rejected(
    graph: dict[str, dict],
) -> None:
    manifest = copy.deepcopy(graph["manifest"])
    manifest["source_definition_bindings"][0]["forward_return"] = "x"
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_input_manifest_v4_4(manifest)

    equivalence = copy.deepcopy(graph["equivalence"])
    equivalence["passes"][0]["engines"][0]["labels"] = []
    equivalence["passes"][0]["engines"][0] = _reseal_engine(
        equivalence["passes"][0]["engines"][0]
    )
    equivalence = _reseal(equivalence)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            equivalence,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )

    bool_direction = copy.deepcopy(graph["equivalence"])
    bool_direction["passes"][0]["engines"][0]["candidates"][0][
        "direction"
    ] = True
    bool_direction["passes"][0]["engines"][0] = _reseal_engine(
        bool_direction["passes"][0]["engines"][0]
    )
    bool_direction = _reseal(bool_direction)
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.validate_two_pass_equivalence_receipt_v4_4(
            bool_direction,
            manifest=graph["manifest"],
            input_receipt=graph["input_receipt"],
            data_field_receipt=graph["data_receipt"],
        )


def test_canonical_parser_rejects_duplicates_and_noncanonical_bytes() -> None:
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.parse_canonical_json_file_bytes_v4_4(b'{"a":1,"a":2}\n')
    with pytest.raises(subject.FactorGovernanceFutureStrictSignalComputabilityV4_4Error):
        subject.parse_canonical_json_file_bytes_v4_4(b'{ "a": 1 }\n')
