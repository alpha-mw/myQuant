from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
import random
import struct

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import (
    governance_future_strict_exact_five_eval_v4_4 as evaluator,
)
from quant_investor.factors import (
    governance_future_strict_signal_computability_v4_4 as contract,
)


def _dates(row_count: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-02", periods=row_count, name="trade_date")


def _exact_arrays(
    row_count: int, symbol_count: int = 4
) -> tuple[tuple[str, ...], dict[str, np.ndarray], np.ndarray]:
    symbols = tuple(f"S{position:02d}" for position in range(symbol_count))
    step = np.arange(row_count, dtype=np.int64)
    close = np.empty((row_count, symbol_count), dtype=np.float64)
    adjusted = np.empty_like(close)
    for column in range(symbol_count):
        if column == symbol_count - 1:
            close[:, column] = np.float64(8.0)
            adjusted[:, column] = np.float64(16.0)
        else:
            close[:, column] = np.ldexp(
                np.full(row_count, float(1 << column), dtype=np.float64), step
            )
            adjusted[:, column] = np.ldexp(
                np.full(row_count, float(2 << column), dtype=np.float64), step
            )
    open_price = np.array(close, dtype=np.float64, order="C", copy=True)
    if row_count > 1:
        open_price[1:] = close[:-1]
    volume = (
        1_000_000.0
        + np.arange(row_count, dtype=np.float64)[:, None]
        * np.arange(1, symbol_count + 1, dtype=np.float64)[None, :]
    )
    pit = np.ones((row_count, symbol_count), dtype=bool)
    return symbols, {
        "raw_close": close,
        "raw_open": open_price,
        "vol": np.array(volume, dtype=np.float64, order="C", copy=True),
        "adj_close": adjusted,
    }, pit


def _input_block(row_count: int) -> evaluator.InputBlockV4_4:
    symbols, arrays, pit = _exact_arrays(row_count)
    return evaluator.build_input_block_v4_4(
        dates=_dates(row_count),
        symbols=symbols,
        pit_mask=pit,
        **arrays,
    )


def _randomized_input_block(row_count: int = 317) -> evaluator.InputBlockV4_4:
    rng = np.random.default_rng(20260719)
    symbol_count = 5
    symbols = tuple(f"S{position:02d}" for position in range(symbol_count))
    returns = rng.normal(0.0004, 0.018, size=(row_count, symbol_count))
    close = 20.0 * np.exp(np.cumsum(returns, axis=0))
    open_price = close * np.exp(
        rng.normal(0.0, 0.007, size=(row_count, symbol_count))
    )
    volume = np.exp(
        rng.normal(13.0, 0.3, size=(row_count, symbol_count))
    )
    adjustment = np.exp(np.linspace(-0.03, 0.04, row_count))[:, None]
    adjusted_close = close * adjustment
    pit = np.ones((row_count, symbol_count), dtype=bool)
    close[88, 1] = np.nan
    pit[145, 2] = False
    return evaluator.build_input_block_v4_4(
        dates=_dates(row_count),
        symbols=symbols,
        raw_close=np.array(close, dtype=np.float64, order="C", copy=True),
        raw_open=np.array(open_price, dtype=np.float64, order="C", copy=True),
        vol=np.array(volume, dtype=np.float64, order="C", copy=True),
        adj_close=np.array(
            adjusted_close, dtype=np.float64, order="C", copy=True
        ),
        pit_mask=pit,
    )


def _program_set() -> dict[str, object]:
    return contract.operator_program_set_v4_4()


def _evaluate_pandas(
    block: evaluator.InputBlockV4_4,
) -> dict[str, np.ndarray]:
    return evaluator.evaluate_pandas_engine_v4_4(
        block, operator_program_set=_program_set()
    )


def _evaluate_numpy(
    block: evaluator.InputBlockV4_4,
) -> dict[str, np.ndarray]:
    return evaluator.evaluate_numpy_engine_v4_4(
        block, operator_program_set=_program_set()
    )


def _rehash_program_set(program_set: dict[str, object]) -> dict[str, object]:
    for program in program_set["candidates"]:
        content = {
            key: copy.deepcopy(value)
            for key, value in program.items()
            if key != "program_semantic_sha256"
        }
        program["program_semantic_sha256"] = contract.semantic_sha256_v4_4(
            content
        )
    content = {
        key: copy.deepcopy(value)
        for key, value in program_set.items()
        if key != "artifact_semantic_sha256"
    }
    program_set["artifact_semantic_sha256"] = contract.semantic_sha256_v4_4(
        content
    )
    return program_set


def _bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def test_constants_input_block_and_no_label_api_surface() -> None:
    assert evaluator.HALO == 60
    assert evaluator.OUTPUT_BLOCK == 128
    assert evaluator.INPUT_FIELDS == ("raw_close", "raw_open", "vol", "adj_close")
    assert evaluator.FACTOR_DIRECTIONS == {
        "alpha_range_position_momentum_20d": 1.0,
        "pv_low_overnight_gap_20d": -1.0,
        "pv_low_vol_ratio_10_60": -1.0,
        "pv_price_volume_consistency_20d": 1.0,
        "pv_low_vol_of_vol_20d": -1.0,
    }

    symbols, arrays, pit = _exact_arrays(61)
    original = arrays["raw_close"]
    block = evaluator.build_input_block_v4_4(
        dates=_dates(61), symbols=symbols, pit_mask=pit, **arrays
    )
    original[0, 0] = 999.0
    assert block.raw_close[0, 0] != 999.0
    for field in (*evaluator.INPUT_FIELDS, "pit_mask"):
        assert getattr(block, field).flags.writeable is False
    assert evaluator.validate_input_block_v4_4(block) is block

    banned = {"label", "labels", "outcome", "outcomes", "statistics", "path"}
    public_functions = [
        getattr(evaluator, name)
        for name in evaluator.__all__
        if inspect.isfunction(getattr(evaluator, name))
    ]
    for function in public_functions:
        assert banned.isdisjoint(inspect.signature(function).parameters)


@pytest.mark.parametrize(
    ("proof_rows", "expected_offsets"),
    [
        (59, [(0, 119, 60, 119)]),
        (60, [(0, 120, 60, 120)]),
        (61, [(0, 121, 60, 121)]),
        (127, [(0, 187, 60, 187)]),
        (128, [(0, 188, 60, 188)]),
        (129, [(0, 188, 60, 188), (128, 189, 188, 189)]),
        (191, [(0, 188, 60, 188), (128, 251, 188, 251)]),
        (192, [(0, 188, 60, 188), (128, 252, 188, 252)]),
    ],
)
def test_exact_block_boundaries_and_complete_no_future_coverage(
    proof_rows: int, expected_offsets: list[tuple[int, int, int, int]]
) -> None:
    row_count = evaluator.HALO + proof_rows
    symbols = ("AAA", "BBB", "CCC")
    dates = _dates(row_count)
    manifest = evaluator.build_block_manifest_v4_4(dates, symbols)
    assert manifest["proof_output_calendar"] == [
        value.strftime("%Y-%m-%d") for value in dates[evaluator.HALO :]
    ]
    assert manifest["proof_output_row_count"] == proof_rows
    observed_offsets = [
        (
            row["input_start_offset"],
            row["input_end_offset"],
            row["output_start_offset"],
            row["output_end_offset"],
        )
        for row in manifest["blocks"]
    ]
    assert observed_offsets == expected_offsets

    covered: list[int] = []
    previous_end = evaluator.HALO
    for row in manifest["blocks"]:
        assert row["output_start_offset"] == previous_end
        assert row["input_start_offset"] == row["output_start_offset"] - 60
        assert row["input_end_offset"] == row["output_end_offset"]
        assert row["future_halo_row_count"] == 0
        assert row["local_output_start_offset"] == 60
        assert row["symbol_axis"] == manifest["symbol_axis"]
        covered.extend(
            range(row["output_start_offset"], row["output_end_offset"])
        )
        previous_end = row["output_end_offset"]
    assert covered == list(range(60, row_count))
    assert manifest["blocks"][0]["output_start_offset"] == 60
    assert manifest["blocks"][0]["input_start_offset"] == 0
    assert evaluator.validate_block_manifest_v4_4(
        manifest, dates=dates, symbols=symbols
    ) == manifest


def test_manifest_and_block_slice_reject_axis_or_offset_drift() -> None:
    block = _input_block(189)
    manifest = evaluator.build_block_manifest_v4_4(block.dates, block.symbols)
    selected = evaluator.slice_input_block_v4_4(block, manifest["blocks"][1])
    assert selected.dates == block.dates[128:189]
    assert selected.symbols == block.symbols
    assert selected.raw_close.shape == (61, 4)

    drifted_manifest = copy.deepcopy(manifest)
    drifted_manifest["blocks"][1]["input_start_offset"] += 1
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="deterministic partition",
    ):
        evaluator.validate_block_manifest_v4_4(drifted_manifest)

    drifted_row = copy.deepcopy(manifest["blocks"][0])
    drifted_row["symbol_axis"]["sha256"] = "0" * 64
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="deterministic source partition",
    ):
        evaluator.slice_input_block_v4_4(block, drifted_row)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fractional", "exact integer"),
        ("boolean", "exact integer"),
        ("string", "exact integer"),
        ("missing", "fields mismatch"),
        ("extra", "fields mismatch"),
        ("date_drift", "dates violate"),
    ],
)
def test_block_slices_reject_nonexact_row_schema(
    case: str, message: str
) -> None:
    source = _input_block(189)
    row = evaluator.build_block_manifest_v4_4(
        source.dates, source.symbols
    )["blocks"][0]
    input_part = evaluator.slice_input_block_v4_4(source, row)
    outputs = {
        name: np.zeros(
            (row["input_row_count"], len(source.symbols)), dtype=np.float64
        )
        for name in evaluator.FACTOR_NAMES
    }
    mutated = copy.deepcopy(row)
    if case == "fractional":
        mutated["input_start_offset"] = 0.5
    elif case == "boolean":
        mutated["block_index"] = False
    elif case == "string":
        mutated["input_end_offset"] = str(mutated["input_end_offset"])
    elif case == "missing":
        mutated.pop("output_row_count")
    elif case == "extra":
        mutated["unexpected"] = 0
    elif case == "date_drift":
        mutated["output_last_date"] = "2099-12-31"
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(case)

    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match=message,
    ):
        evaluator.slice_input_block_v4_4(source, mutated)
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match=message,
    ):
        evaluator.slice_non_halo_outputs_v4_4(
            outputs, mutated, source_block=input_part
        )


@pytest.mark.parametrize("shape", [(187, 4), (188, 3), (189, 4)])
def test_non_halo_slice_rejects_nonexact_output_shape(
    shape: tuple[int, int],
) -> None:
    source = _input_block(189)
    row = evaluator.build_block_manifest_v4_4(
        source.dates, source.symbols
    )["blocks"][0]
    input_part = evaluator.slice_input_block_v4_4(source, row)
    outputs = {
        name: np.zeros(shape, dtype=np.float64)
        for name in evaluator.FACTOR_NAMES
    }
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="shape is invalid",
    ):
        evaluator.slice_non_halo_outputs_v4_4(
            outputs, row, source_block=input_part
        )


@pytest.mark.parametrize(
    "case", ["coherent_dates", "alternate_axis", "matching_smaller_shape"]
)
def test_non_halo_slice_rejects_coherent_provenance_substitution(
    case: str,
) -> None:
    source = _input_block(189)
    row = evaluator.build_block_manifest_v4_4(
        source.dates, source.symbols
    )["blocks"][0]
    input_part = evaluator.slice_input_block_v4_4(source, row)
    mutated = copy.deepcopy(row)
    output_symbol_count = len(source.symbols)
    if case == "coherent_dates":
        mutated.update(
            {
                "input_first_date": "2099-01-04",
                "output_first_date": "2099-04-01",
                "input_last_date": "2099-09-01",
                "output_last_date": "2099-09-01",
            }
        )
    elif case == "alternate_axis":
        mutated["symbol_axis"] = evaluator.build_block_manifest_v4_4(
            source.dates, ("ALT0", "ALT1", "ALT2", "ALT3")
        )["symbol_axis"]
    elif case == "matching_smaller_shape":
        mutated["symbol_axis"] = evaluator.build_block_manifest_v4_4(
            source.dates, ("ALT0", "ALT1", "ALT2")
        )["symbol_axis"]
        output_symbol_count = 3
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(case)
    outputs = {
        name: np.zeros(
            (row["input_row_count"], output_symbol_count), dtype=np.float64
        )
        for name in evaluator.FACTOR_NAMES
    }
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="exact input block provenance",
    ):
        evaluator.slice_non_halo_outputs_v4_4(
            outputs, mutated, source_block=input_part
        )


def test_independent_engines_repeat_and_assemble_under_block_local_state() -> None:
    source = _input_block(252)
    manifest = evaluator.build_block_manifest_v4_4(source.dates, source.symbols)
    pandas_parts = {name: [] for name in evaluator.FACTOR_NAMES}
    numpy_parts = {name: [] for name in evaluator.FACTOR_NAMES}
    for row in manifest["blocks"]:
        input_part = evaluator.slice_input_block_v4_4(source, row)
        pandas_part = _evaluate_pandas(input_part)
        numpy_part = _evaluate_numpy(input_part)
        pandas_repeat = _evaluate_pandas(input_part)
        numpy_repeat = _evaluate_numpy(input_part)
        assert evaluator.compare_exact_engine_outputs_v4_4(
            pandas_part, numpy_part, input_part
        )["exact"] is True
        for name in evaluator.FACTOR_NAMES:
            assert _bits_equal(pandas_part[name], pandas_repeat[name])
            assert _bits_equal(numpy_part[name], numpy_repeat[name])
        pandas_selected = evaluator.slice_non_halo_outputs_v4_4(
            pandas_part, row, source_block=input_part
        )
        numpy_selected = evaluator.slice_non_halo_outputs_v4_4(
            numpy_part, row, source_block=input_part
        )
        for name in evaluator.FACTOR_NAMES:
            pandas_parts[name].append(pandas_selected[name])
            numpy_parts[name].append(numpy_selected[name])

    for name in evaluator.FACTOR_NAMES:
        pandas_assembled = np.concatenate(pandas_parts[name], axis=0)
        numpy_assembled = np.concatenate(numpy_parts[name], axis=0)
        assert _bits_equal(pandas_assembled, numpy_assembled)


def test_randomized_boundaries_are_block_local_not_monolithic() -> None:
    source = _randomized_input_block()
    manifest = evaluator.build_block_manifest_v4_4(source.dates, source.symbols)
    program_set = _program_set()
    semantics = program_set["execution_semantics"]
    assert semantics["execution_partitioning"] == (
        "deterministic_block_local_by_manifest_input_block"
    )
    assert semantics["rolling_state_lifecycle"] == (
        "reset_before_every_manifest_input_block"
    )
    assert semantics["historical_halo_session_count"] == evaluator.HALO
    assert (
        semantics["maximum_output_session_count_per_block"]
        == evaluator.OUTPUT_BLOCK
    )
    assert semantics["pandas_fixed_window_accumulator_semantics"] == (
        "pandas_3.0.1_within_each_manifest_input_block"
    )
    assert semantics["monolithic_pandas_bit_equivalence_claimed"] is False

    pandas_parts = {name: [] for name in evaluator.FACTOR_NAMES}
    numpy_parts = {name: [] for name in evaluator.FACTOR_NAMES}
    for row in manifest["blocks"]:
        input_part = evaluator.slice_input_block_v4_4(source, row)
        pandas_first = evaluator.evaluate_pandas_engine_v4_4(
            input_part, operator_program_set=program_set
        )
        numpy_first = evaluator.evaluate_numpy_engine_v4_4(
            input_part, operator_program_set=program_set
        )
        pandas_repeat = evaluator.evaluate_pandas_engine_v4_4(
            input_part, operator_program_set=program_set
        )
        numpy_repeat = evaluator.evaluate_numpy_engine_v4_4(
            input_part, operator_program_set=program_set
        )
        for name in evaluator.FACTOR_NAMES:
            assert _bits_equal(pandas_first[name], numpy_first[name])
            assert _bits_equal(pandas_first[name], pandas_repeat[name])
            assert _bits_equal(numpy_first[name], numpy_repeat[name])
        pandas_selected = evaluator.slice_non_halo_outputs_v4_4(
            pandas_first, row, source_block=input_part
        )
        numpy_selected = evaluator.slice_non_halo_outputs_v4_4(
            numpy_first, row, source_block=input_part
        )
        for name in evaluator.FACTOR_NAMES:
            pandas_parts[name].append(pandas_selected[name])
            numpy_parts[name].append(numpy_selected[name])

    pandas_assembled = {
        name: np.concatenate(pandas_parts[name], axis=0)
        for name in evaluator.FACTOR_NAMES
    }
    numpy_assembled = {
        name: np.concatenate(numpy_parts[name], axis=0)
        for name in evaluator.FACTOR_NAMES
    }
    for name in evaluator.FACTOR_NAMES:
        assert _bits_equal(pandas_assembled[name], numpy_assembled[name])

    # This unsupported diagnostic deliberately keeps one rolling state across
    # the full calendar.  Its finite low-bit drift at later block boundaries
    # demonstrates why it is not the contract oracle.
    monolithic_diagnostic = (
        evaluator._evaluate_pandas_validated_operator_program_v4_4(
            source, program_set
        )
    )
    names_with_expected_monolithic_drift: set[str] = set()
    first_block_output_count = manifest["blocks"][0]["output_row_count"]
    for name in evaluator.FACTOR_NAMES:
        block_local = pandas_assembled[name]
        monolithic = monolithic_diagnostic[name][evaluator.HALO :]
        assert np.array_equal(np.isnan(block_local), np.isnan(monolithic))
        finite_difference = (
            np.isfinite(block_local)
            & np.isfinite(monolithic)
            & (block_local.view(np.uint64) != monolithic.view(np.uint64))
        )
        assert not finite_difference[:first_block_output_count].any()
        if finite_difference.any():
            names_with_expected_monolithic_drift.add(name)
    assert names_with_expected_monolithic_drift == {
        "pv_low_overnight_gap_20d",
        "pv_low_vol_ratio_10_60",
        "pv_low_vol_of_vol_20d",
    }


def test_test_only_structural_program_mutation_drives_both_interpreters() -> None:
    symbols, arrays, pit = _exact_arrays(61)
    arrays["raw_close"][3, 0] = -0.0
    arrays["raw_close"][4, 0] = 0.0
    block = evaluator.build_input_block_v4_4(
        dates=_dates(61), symbols=symbols, pit_mask=pit, **arrays
    )
    mutated = _program_set()
    first_program = mutated["candidates"][0]
    first_program["nodes"] = [copy.deepcopy(first_program["nodes"][0])]
    first_program["output_node_id"] = "n000"
    _rehash_program_set(mutated)

    structurally_valid = contract._validate_operator_program_set_structure_v4_4(
        mutated
    )
    with pytest.raises(
        contract.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="fixed golden exact five",
    ):
        contract.validate_operator_program_set_v4_4(structurally_valid)
    with pytest.raises(
        contract.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match="fixed golden exact five",
    ):
        evaluator.evaluate_pandas_engine_v4_4(
            block, operator_program_set=structurally_valid
        )

    pandas_outputs = evaluator._evaluate_pandas_validated_operator_program_v4_4(
        block, structurally_valid
    )
    numpy_outputs = evaluator._evaluate_numpy_validated_operator_program_v4_4(
        block, structurally_valid
    )
    name = first_program["name"]
    assert _bits_equal(pandas_outputs[name], block.raw_close)
    assert _bits_equal(numpy_outputs[name], block.raw_close)
    assert np.signbit(pandas_outputs[name][3, 0])
    assert not np.signbit(pandas_outputs[name][4, 0])
    assert evaluator.compare_exact_engine_outputs_v4_4(
        pandas_outputs, numpy_outputs, block
    )["exact"] is True


@pytest.mark.parametrize(
    ("node_index", "mutation", "message"),
    [
        (1, {"opcode": "unknown"}, "id/opcode"),
        (1, {"inputs": ["n999"]}, "prior nodes"),
        (1, {"parameters": {"window": 0, "min_periods": 1}}, "invalid"),
    ],
)
def test_contract_rejects_invalid_operator_nodes(
    node_index: int, mutation: dict[str, object], message: str
) -> None:
    invalid = _program_set()
    invalid["candidates"][0]["nodes"][node_index].update(mutation)
    with pytest.raises(
        contract.FactorGovernanceFutureStrictSignalComputabilityV4_4Error,
        match=message,
    ):
        contract.validate_operator_program_set_v4_4(invalid)


def test_interpreter_ast_has_no_candidate_or_formula_window_dispatch() -> None:
    module_tree = ast.parse(inspect.getsource(evaluator))
    interpreter_names = {
        "_evaluate_pandas_validated_operator_program_v4_4",
        "_evaluate_numpy_validated_operator_program_v4_4",
    }
    interpreters = [
        node
        for node in module_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in interpreter_names
    ]
    assert {node.name for node in interpreters} == interpreter_names
    for interpreter in interpreters:
        string_literals = {
            node.value
            for node in ast.walk(interpreter)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert set(evaluator.FACTOR_NAMES).isdisjoint(string_literals)
        numeric_literals = {
            node.value
            for node in ast.walk(interpreter)
            if isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        }
        assert {5, 10, 20, 60}.isdisjoint(numeric_literals)
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            for node in ast.walk(interpreter)
        )


def test_streaming_non_halo_descriptors_equal_block_local_assembly() -> None:
    block = _input_block(252)
    manifest = evaluator.build_block_manifest_v4_4(block.dates, block.symbols)
    chunks = []
    assembled = {name: [] for name in evaluator.FACTOR_NAMES}
    for row in manifest["blocks"]:
        input_part = evaluator.slice_input_block_v4_4(block, row)
        outputs = _evaluate_pandas(input_part)
        selected = evaluator.slice_non_halo_outputs_v4_4(
            outputs, row, source_block=input_part
        )
        dates = block.dates[row["output_start_offset"] : row["output_end_offset"]]
        chunks.append((dates, selected))
        for name in evaluator.FACTOR_NAMES:
            assembled[name].append(selected[name])
    descriptors = evaluator.build_streaming_global_descriptors_v4_4(
        chunks,
        symbols=block.symbols,
        expected_dates=block.dates[60:],
    )
    for name in evaluator.FACTOR_NAMES:
        matrix = np.concatenate(assembled[name], axis=0)
        expected = evaluator.matrix_hash_descriptor_v4_4(
            matrix, dates=block.dates[60:], symbols=block.symbols
        )
        assert descriptors[name] == expected
        assert evaluator.validate_global_descriptor_v4_4(
            descriptors[name],
            expected_dates=block.dates[60:],
            expected_symbols=block.symbols,
            matrix=matrix,
        ) == expected


def test_historical_full_symbol_axis_rank_with_delisted_and_future_mix() -> None:
    dates = _dates(3)
    symbols = ("AAA_DELISTED", "BBB_CURRENT", "CCC_CURRENT", "ZZZ_FUTURE")
    raw_close = np.array(
        [[1.0, 1.0, 2.0, 100.0], [2.0, 2.0, 1.0, 101.0], [4.0, 3.0, 0.5, 102.0]],
        dtype=np.float64,
    )
    raw_open = np.array(raw_close, copy=True)
    vol = np.array(
        [[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0], [12.0, 22.0, 32.0, 42.0]],
        dtype=np.float64,
    )
    pit = np.array(
        [[True, True, True, False], [True, True, True, False], [False, True, True, True]],
        dtype=bool,
    )
    block = evaluator.build_input_block_v4_4(
        dates=dates,
        symbols=symbols,
        raw_close=raw_close,
        raw_open=raw_open,
        vol=vol,
        adj_close=np.array(raw_close, copy=True),
        pit_mask=pit,
    )
    source = _evaluate_pandas(block)
    local = _evaluate_numpy(block)
    evaluator.compare_exact_engine_outputs_v4_4(
        source, local, block, require_positive_proof=False
    )
    ranked = source["alpha_range_position_momentum_20d"]
    assert ranked[1, 0] == np.float64(5.0 / 6.0)
    assert ranked[1, 1] == np.float64(5.0 / 6.0)
    assert ranked[1, 2] == np.float64(1.0 / 3.0)
    assert np.isnan(ranked[1, 3])
    assert np.isnan(ranked[2, 0])
    assert tuple(block.symbols) == symbols


def test_each_factor_first_computable_session_is_exact() -> None:
    block = _input_block(61)
    source = _evaluate_pandas(block)
    local = _evaluate_numpy(block)
    evaluator.compare_exact_engine_outputs_v4_4(source, local, block)
    expected_first = {
        "alpha_range_position_momentum_20d": 1,
        "pv_low_overnight_gap_20d": 20,
        "pv_low_vol_ratio_10_60": 60,
        "pv_price_volume_consistency_20d": 20,
        "pv_low_vol_of_vol_20d": 24,
    }
    for name, first in expected_first.items():
        finite_rows = np.flatnonzero(np.isfinite(source[name]).any(axis=1))
        assert finite_rows[0] == first
        assert not np.isfinite(source[name][:first]).any()
    assert np.isfinite(source["alpha_range_position_momentum_20d"][1]).any()


def test_overnight_uses_twenty_opens_and_lagged_closes_not_current_close() -> None:
    block = _input_block(62)
    baseline = _evaluate_pandas(block)[
        "pv_low_overnight_gap_20d"
    ]
    changed_close = np.array(block.raw_close, dtype=np.float64, copy=True)
    changed_close[60, 0] = np.nextafter(changed_close[60, 0], np.inf)
    changed = evaluator.build_input_block_v4_4(
        dates=block.dates,
        symbols=block.symbols,
        raw_close=changed_close,
        raw_open=np.array(block.raw_open, dtype=np.float64, copy=True),
        vol=np.array(block.vol, dtype=np.float64, copy=True),
        adj_close=np.array(block.adj_close, dtype=np.float64, copy=True),
        pit_mask=np.array(block.pit_mask, dtype=bool, copy=True),
    )
    changed_output = _evaluate_pandas(changed)[
        "pv_low_overnight_gap_20d"
    ]
    assert baseline[60, 0].view(np.uint64) == changed_output[60, 0].view(np.uint64)
    assert baseline[61, 0].view(np.uint64) != changed_output[61, 0].view(np.uint64)

    twenty = _input_block(20)
    twenty_one = _input_block(21)
    assert not np.isfinite(
        _evaluate_pandas(twenty)[
            "pv_low_overnight_gap_20d"
        ]
    ).any()
    assert np.isfinite(
        _evaluate_pandas(twenty_one)[
            "pv_low_overnight_gap_20d"
        ][20]
    ).any()


def test_node_level_pit_mask_and_nan_gap_semantics() -> None:
    symbols, arrays, pit = _exact_arrays(100)
    pit[30, 0] = False
    arrays["raw_close"][10, 2] = np.nan
    block = evaluator.build_input_block_v4_4(
        dates=_dates(100), symbols=symbols, pit_mask=pit, **arrays
    )
    source = _evaluate_pandas(block)
    local = _evaluate_numpy(block)
    evaluator.compare_exact_engine_outputs_v4_4(source, local, block)
    for name in evaluator.FACTOR_NAMES:
        assert np.isnan(source[name][30, 0])
        assert not np.isinf(source[name]).any()
    overnight = source["pv_low_overnight_gap_20d"][:, 0]
    assert np.isnan(overnight[30:51]).all()
    assert np.isfinite(overnight[51])


def test_hostile_nan_generated_infinity_signed_zero_ties_and_pit_holes() -> None:
    symbols, arrays, pit = _exact_arrays(65)
    arrays["raw_close"][:, 1] = arrays["raw_close"][:, 0]
    arrays["raw_open"][:, 1] = arrays["raw_open"][:, 0]
    arrays["raw_close"][4, 0] = -0.0
    arrays["raw_close"][5, 0] = 0.0
    arrays["raw_close"][6, 1] = -np.float64(1e-9)
    arrays["raw_open"][7, 1] = 1.0
    arrays["raw_close"][12, 2] = np.nan
    pit[30, 0] = False
    block = evaluator.build_input_block_v4_4(
        dates=_dates(65), symbols=symbols, pit_mask=pit, **arrays
    )

    pandas_outputs = _evaluate_pandas(block)
    numpy_outputs = _evaluate_numpy(block)
    assert evaluator.compare_exact_engine_outputs_v4_4(
        pandas_outputs, numpy_outputs, block
    )["exact"] is True
    for outputs in (pandas_outputs, numpy_outputs):
        assert all(not np.isinf(matrix).any() for matrix in outputs.values())
        assert all(np.isnan(matrix[30, 0]) for matrix in outputs.values())
    ranked = pandas_outputs["alpha_range_position_momentum_20d"]
    assert ranked[2, 0].view(np.uint64) == ranked[2, 1].view(np.uint64)
    assert np.isnan(ranked[0, -1])
    overnight = pandas_outputs["pv_low_overnight_gap_20d"]
    assert np.isnan(overnight[20:27, 1]).all()


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
@pytest.mark.parametrize("field", evaluator.INPUT_FIELDS)
def test_input_rejects_positive_and_negative_infinity(
    field: str, bad_value: float
) -> None:
    symbols, arrays, pit = _exact_arrays(61)
    arrays[field][5, 1] = bad_value
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match=field,
    ):
        evaluator.build_input_block_v4_4(
            dates=_dates(61), symbols=symbols, pit_mask=pit, **arrays
        )


def test_input_rejects_non_float64_shape_and_unsorted_axes_but_allows_nan() -> None:
    symbols, arrays, pit = _exact_arrays(61)
    wrong_dtype = {**arrays, "raw_close": arrays["raw_close"].astype(np.float32)}
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="float64",
    ):
        evaluator.build_input_block_v4_4(
            dates=_dates(61), symbols=symbols, pit_mask=pit, **wrong_dtype
        )
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="sorted",
    ):
        evaluator.build_input_block_v4_4(
            dates=_dates(61),
            symbols=tuple(reversed(symbols)),
            pit_mask=pit,
            **arrays,
        )
    arrays["raw_close"][4, 0] = np.nan
    block = evaluator.build_input_block_v4_4(
        dates=_dates(61), symbols=symbols, pit_mask=pit, **arrays
    )
    assert np.isnan(block.raw_close[4, 0])


def test_finite_overflow_is_converted_to_nan_and_never_output_infinity() -> None:
    symbols, arrays, pit = _exact_arrays(61)
    arrays["raw_close"][0, 0] = np.nextafter(0.0, 1.0)
    arrays["raw_close"][1, 0] = np.finfo(np.float64).max
    arrays["raw_open"][1, 0] = np.finfo(np.float64).max
    block = evaluator.build_input_block_v4_4(
        dates=_dates(61), symbols=symbols, pit_mask=pit, **arrays
    )
    source = _evaluate_pandas(block)
    local = _evaluate_numpy(block)
    evaluator.compare_exact_engine_outputs_v4_4(source, local, block)
    for outputs in (source, local):
        assert all(not np.isinf(matrix).any() for matrix in outputs.values())


def test_descriptor_canonical_nan_signed_zero_infinities_and_negated_hash() -> None:
    dates = ("2026-07-20", "2026-07-21")
    symbols = ("A", "B", "C")
    matrix = np.array(
        [[0.0, -0.0, np.nan], [np.inf, -np.inf, 1.0]], dtype=np.float64
    )
    descriptor = evaluator.matrix_hash_descriptor_v4_4(
        matrix, dates=dates, symbols=symbols
    )
    assert descriptor["row_count"] == 2
    assert descriptor["column_count"] == 3
    assert descriptor["finite_count"] == 3
    assert descriptor["nan_count"] == 1
    assert descriptor["positive_infinity_count"] == 1
    assert descriptor["negative_infinity_count"] == 1
    assert descriptor["positive_finite_count"] == 1
    assert descriptor["negative_finite_count"] == 0
    assert descriptor["positive_zero_count"] == 1
    assert descriptor["negative_zero_count"] == 1
    assert descriptor["byte_count"] == 48
    assert descriptor["matrix_sha256"] == descriptor["bit_pattern_sha256"]
    assert descriptor["date_axis"]["sha256"] == hashlib.sha256(
        b"2026-07-20\n2026-07-21\n"
    ).hexdigest()
    assert descriptor["symbol_axis"]["sha256"] == hashlib.sha256(
        b"A\nB\nC\n"
    ).hexdigest()
    negated = evaluator.matrix_hash_descriptor_v4_4(
        -matrix, dates=dates, symbols=symbols
    )
    assert (
        descriptor["elementwise_negated_sha256"]
        == negated["bit_pattern_sha256"]
    )

    first_nan = np.array([[np.nan]], dtype=np.float64)
    second_nan = np.array([[0.0]], dtype=np.float64)
    second_nan.view(np.uint64)[0, 0] = np.uint64(0x7FF0000000000001)
    first = evaluator.matrix_hash_descriptor_v4_4(
        first_nan, dates=(dates[0],), symbols=(symbols[0],)
    )
    second = evaluator.matrix_hash_descriptor_v4_4(
        second_nan, dates=(dates[0],), symbols=(symbols[0],)
    )
    assert first["bit_pattern_sha256"] == second["bit_pattern_sha256"]


def test_exact_matrix_comparison_covers_nan_inf_and_signed_zero() -> None:
    left = np.array([[np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0]], dtype=np.float64)
    right = np.array(left, copy=True)
    right.view(np.uint64)[0, 0] = np.uint64(0x7FF0000000000001)
    assert evaluator.compare_exact_matrices_v4_4(left, right)["exact"] is True

    zero_drift = np.array(right, copy=True)
    zero_drift[0, 4] = 0.0
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="divergence",
    ):
        evaluator.compare_exact_matrices_v4_4(left, zero_drift)

    infinity_drift = np.array(right, copy=True)
    infinity_drift[0, 1] = -np.inf
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="divergence",
    ):
        evaluator.compare_exact_matrices_v4_4(left, infinity_drift)


def test_deliberate_engine_divergence_is_rejected_without_tolerance() -> None:
    block = _input_block(61)
    source = _evaluate_pandas(block)
    local = {
        name: np.array(matrix, dtype=np.float64, order="C", copy=True)
        for name, matrix in _evaluate_numpy(block).items()
    }
    name = "pv_low_vol_ratio_10_60"
    local[name][60, 0] = np.nextafter(local[name][60, 0], np.inf)
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="divergence",
    ):
        evaluator.compare_exact_engine_outputs_v4_4(source, local, block)


def test_names_directions_come_from_program_and_no_pass_or_compat_api_remains() -> None:
    programs = _program_set()["candidates"]
    assert evaluator.FACTOR_NAMES == tuple(row["name"] for row in programs)
    assert evaluator.FACTOR_DIRECTIONS == {
        row["name"]: row["direction"] for row in programs
    }
    assert not hasattr(evaluator, "ENGINE_PASS_SCHEMA_VERSION")
    assert not hasattr(evaluator, "build_engine_pass_result_v4_4")
    assert not hasattr(evaluator, "evaluate_pandas_source_dag_v4_4")
    assert not hasattr(evaluator, "evaluate_numpy_local_formulas_v4_4")


def test_descriptor_and_streaming_validator_reject_tamper_and_order_drift() -> None:
    matrix = np.array([[0.0], [-0.0]], dtype=np.float64)
    descriptor = evaluator.matrix_hash_descriptor_v4_4(
        matrix, dates=("2026-07-20", "2026-07-21"), symbols=("A",)
    )
    tampered = copy.deepcopy(descriptor)
    tampered["negative_zero_count"] = 0
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="finite sign accounting",
    ):
        evaluator.validate_global_descriptor_v4_4(tampered)

    state = evaluator.StreamingMatrixDescriptorV4_4(("A",))
    state.update(("2026-07-21",), np.array([[1.0]], dtype=np.float64))
    with pytest.raises(
        evaluator.FactorGovernanceFutureStrictExactFiveEvalV4_4Error,
        match="chronological",
    ):
        state.update(("2026-07-20",), np.array([[2.0]], dtype=np.float64))


def test_price_volume_consistency_uses_physical_volume_without_rescaling() -> None:
    block = _input_block(21)
    baseline = _evaluate_pandas(block)[
        "pv_price_volume_consistency_20d"
    ]
    assert baseline[20, 0] == 1.0
    scaled_volume = np.array(block.vol * 1_000_000.0, dtype=np.float64, copy=True)
    scaled = evaluator.build_input_block_v4_4(
        dates=block.dates,
        symbols=block.symbols,
        raw_close=np.array(block.raw_close, dtype=np.float64, copy=True),
        raw_open=np.array(block.raw_open, dtype=np.float64, copy=True),
        vol=scaled_volume,
        adj_close=np.array(block.adj_close, dtype=np.float64, copy=True),
        pit_mask=np.array(block.pit_mask, dtype=bool, copy=True),
    )
    observed = _evaluate_pandas(scaled)[
        "pv_price_volume_consistency_20d"
    ]
    assert _bits_equal(baseline, observed)


def _fma_grid() -> list[tuple[float, float, float]]:
    """Edge values plus a deterministic pseudo-random spread of bit patterns."""

    special = (
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        3.0,
        7.0,
        0.1,
        -0.1,
        1e16,
        -1e16,
        1e308,
        -1e308,
        5e-324,
        -5e-324,
        2.2250738585072014e-308,
        math.inf,
        -math.inf,
        math.nan,
    )
    cases = [(x, y, z) for x in special for y in special for z in special]
    rng = random.Random(20260804)
    for _ in range(20000):
        cases.append(
            (
                rng.uniform(-1e6, 1e6),
                rng.uniform(-1e6, 1e6),
                rng.uniform(-1e6, 1e6),
            )
        )
    for _ in range(20000):
        cases.append(
            tuple(  # type: ignore[arg-type]
                struct.unpack(">d", rng.randbytes(8))[0] for _ in range(3)
            )
        )
    return cases


def _fma_outcome(function, x: float, y: float, z: float):
    try:
        return ("value", struct.pack(">d", function(x, y, z)))
    except Exception as exc:  # noqa: BLE001 - exception type is the contract
        return (type(exc).__name__, str(exc))


@pytest.mark.skipif(
    not hasattr(math, "fma"),
    reason="no interpreter-provided math.fma to compare the fallback against",
)
def test_fma_fallback_is_bit_identical_to_math_fma() -> None:
    """The 3.10-3.12 fallback must not perturb a single receipt bit.

    ``math.fma`` is 3.13+, so on every Python version this project claims to
    support the fallback is what actually runs.  Rounding twice would change
    the rolling variance recurrence and therefore every downstream hash.
    """

    for x, y, z in _fma_grid():
        reference = _fma_outcome(math.fma, x, y, z)
        observed = _fma_outcome(evaluator._fma_exact, x, y, z)
        if reference[0] == "value" and observed[0] == "value":
            reference_value = struct.unpack(">d", reference[1])[0]
            observed_value = struct.unpack(">d", observed[1])[0]
            if math.isnan(reference_value) and math.isnan(observed_value):
                continue
        assert observed == reference, (x, y, z)


def test_fma_fallback_rounds_once() -> None:
    """A case where the fused result differs from ``x * y + z``."""

    x = 1.0 + 2.0**-52
    y = 1.0 - 2.0**-52
    z = -1.0
    assert x * y + z == 0.0
    assert evaluator._fma_exact(x, y, z) == -(2.0**-104)


def test_fma_dispatch_prefers_the_interpreter_primitive() -> None:
    expected = getattr(math, "fma", evaluator._fma_exact)
    assert evaluator._fma is expected
