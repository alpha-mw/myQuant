from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_exact_five_no_label_eval_v4_4 as subject


def _fixture(rows: int = 100) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = pd.date_range("2025-01-02", periods=rows, freq="B", name="trade_date")
    columns = pd.Index(["000001.SZ", "000002.SZ", "600000.SH"], name="ts_code")
    step = np.arange(rows, dtype=np.float64)[:, None]
    offsets = np.asarray([0.0, 3.0, 7.0])[None, :]
    close = pd.DataFrame(
        10.0 + step * np.asarray([0.07, 0.04, 0.09])[None, :] + offsets,
        index=index,
        columns=columns,
    )
    close.iloc[35:42, 1] = close.iloc[34, 1]
    open_price = close.mul(1.0 + np.sin(step / 7.0) * 0.002)
    volume = pd.DataFrame(
        100_000.0 + step * np.asarray([110.0, 170.0, 90.0])[None, :] + offsets,
        index=index,
        columns=columns,
    )
    adjusted = close.copy()
    adjusted.iloc[50:, 0] *= 0.5
    mask = pd.DataFrame(True, index=index, columns=columns, dtype=bool)
    mask.iloc[12:18, 2] = False
    return {
        "raw_close": close,
        "raw_open": open_price,
        "vol": volume,
        "adj_close": adjusted,
    }, mask


def _descriptors(values: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        name: subject.matrix_hash_descriptor_v4_4(matrix)["matrix_sha256"]
        for name, matrix in values.items()
    }


def test_two_independent_engines_match_exactly_for_exact_five() -> None:
    inputs, mask = _fixture()
    source = subject.evaluate_source_dag_v4_4(inputs, mask)
    local = subject.evaluate_local_formulas_v4_4(inputs, mask)
    assert tuple(source) == tuple(subject.CANDIDATE_DIRECTIONS)
    assert _descriptors(source) == _descriptors(local)
    for matrix in source.values():
        assert matrix.where(~mask).isna().all().all()


def test_field_adapters_are_candidate_specific_and_have_no_fallback() -> None:
    inputs, mask = _fixture()
    baseline = _descriptors(subject.evaluate_source_dag_v4_4(inputs, mask))

    adjusted = {name: value.copy() for name, value in inputs.items()}
    adjusted["adj_close"].iloc[45:, 1] *= 1.7
    changed = _descriptors(subject.evaluate_source_dag_v4_4(adjusted, mask))
    assert changed["pv_low_vol_of_vol_20d"] != baseline["pv_low_vol_of_vol_20d"]
    for name in tuple(subject.CANDIDATE_DIRECTIONS)[:-1]:
        assert changed[name] == baseline[name]

    volume = {name: value.copy() for name, value in inputs.items()}
    volume["vol"].iloc[30:80, 0] = volume["vol"].iloc[30:80, 0].iloc[::-1].to_numpy()
    volume_changed = _descriptors(subject.evaluate_source_dag_v4_4(volume, mask))
    assert (
        volume_changed["pv_price_volume_consistency_20d"]
        != baseline["pv_price_volume_consistency_20d"]
    )
    for name in (
        "alpha_range_position_momentum_20d",
        "pv_low_overnight_gap_20d",
        "pv_low_vol_ratio_10_60",
        "pv_low_vol_of_vol_20d",
    ):
        assert volume_changed[name] == baseline[name]

    missing = dict(inputs)
    del missing["vol"]
    missing["volume"] = inputs["vol"]
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="exact ordered raw_close/raw_open/vol/adj_close",
    ):
        subject.evaluate_source_dag_v4_4(missing, mask)


def test_source_programs_are_closed_and_reject_tamper_or_malicious_ast() -> None:
    tampered = copy.deepcopy(list(subject.SOURCE_PROGRAMS_V4_4))
    tampered[0]["program"] = tampered[0]["program"].replace(",20,1", ",21,1")
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="literal definition mismatch",
    ):
        subject.validate_source_programs_v4_4(tampered)

    malicious = copy.deepcopy(list(subject.SOURCE_PROGRAMS_V4_4))
    malicious[1]["program"] = "__import__('os').system('true')"
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="literal definition mismatch",
    ):
        subject.validate_source_programs_v4_4(malicious)


def test_axes_mask_and_future_prefix_are_fail_closed_and_causal() -> None:
    inputs, mask = _fixture(110)
    full = subject.evaluate_source_dag_v4_4(inputs, mask)
    prefix_inputs = {name: value.iloc[:100].copy() for name, value in inputs.items()}
    prefix_mask = mask.iloc[:100].copy()
    prefix = subject.evaluate_source_dag_v4_4(prefix_inputs, prefix_mask)
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[:100], prefix[name])

    unsorted = {name: value.iloc[::-1].copy() for name, value in inputs.items()}
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="date axis must be strictly ordered",
    ):
        subject.evaluate_source_dag_v4_4(unsorted, mask.iloc[::-1])

    duplicate_mask = pd.concat([mask.iloc[:1], mask])
    duplicate_inputs = {
        name: pd.concat([value.iloc[:1], value]) for name, value in inputs.items()
    }
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="date axis must be strictly ordered",
    ):
        subject.evaluate_source_dag_v4_4(duplicate_inputs, duplicate_mask)


def test_matrix_descriptor_normalizes_nan_but_preserves_special_bits() -> None:
    index = pd.date_range("2026-01-02", periods=5, freq="B")
    frame = pd.DataFrame(
        [np.nan, np.inf, -np.inf, -0.0, 0.0],
        index=index,
        columns=["000001.SZ"],
    )
    descriptor = subject.matrix_hash_descriptor_v4_4(frame)
    assert descriptor["nan_count"] == 1
    assert descriptor["positive_infinity_count"] == 1
    assert descriptor["negative_infinity_count"] == 1
    assert descriptor["negative_zero_count"] == 1

    alternative_nan = frame.copy()
    alternative_nan.iloc[0, 0] = np.asarray(
        [np.uint64(0x7FF8000000000001)], dtype="uint64"
    ).view("float64")[0]
    assert (
        subject.matrix_hash_descriptor_v4_4(alternative_nan)["matrix_sha256"]
        == descriptor["matrix_sha256"]
    )
    assert (
        subject.matrix_hash_descriptor_v4_4(alternative_nan)["bit_pattern_sha256"]
        == descriptor["bit_pattern_sha256"]
    )


def test_engine_pass_is_exact_self_hashed_and_direction_bound() -> None:
    inputs, mask = _fixture()
    outputs = subject.evaluate_source_dag_v4_4(inputs, mask)
    collection = hashlib.sha256(b"fresh-pass-one").hexdigest()
    result = subject.build_engine_pass_result_v4_4(
        engine_id=subject.SOURCE_ENGINE_ID,
        pass_id="fresh_pass_1",
        collection_sha256=collection,
        outputs=outputs,
        pit_mask=mask,
    )
    assert subject.validate_engine_pass_result_v4_4(result) == result
    assert result["candidates"][1]["direction"] == -1.0
    assert (
        result["candidates"][1]["raw_matrix"]["matrix_sha256"]
        != result["candidates"][1]["direction_adjusted_matrix"]["matrix_sha256"]
    )

    tampered = copy.deepcopy(result)
    tampered["candidates"][1]["direction"] = 1.0
    body = {key: value for key, value in tampered.items() if key != "result_semantic_sha256"}
    tampered["result_semantic_sha256"] = subject.semantic_sha256_v4_4(body)
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="exact-five oracle",
    ):
        subject.validate_engine_pass_result_v4_4(tampered)

    tampered = copy.deepcopy(result)
    tampered["candidates"][1]["direction_adjusted_matrix"] = copy.deepcopy(
        tampered["candidates"][1]["raw_matrix"]
    )
    body = {
        key: value for key, value in tampered.items() if key != "result_semantic_sha256"
    }
    tampered["result_semantic_sha256"] = subject.semantic_sha256_v4_4(body)
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="negative-direction adjusted descriptor transform mismatch",
    ):
        subject.validate_engine_pass_result_v4_4(tampered)


def test_negative_direction_rejects_positionwise_sign_mismatch() -> None:
    inputs, mask = _fixture()
    outputs = {
        name: value.copy()
        for name, value in subject.evaluate_source_dag_v4_4(inputs, mask).items()
    }
    negative_name = tuple(outputs)[1]
    finite_positions = np.argwhere(np.isfinite(outputs[negative_name].to_numpy()))
    assert len(finite_positions) >= 2
    first, second = finite_positions[:2]
    outputs[negative_name].iat[int(first[0]), int(first[1])] = 1.0
    outputs[negative_name].iat[int(second[0]), int(second[1])] = -2.0
    result = subject.build_engine_pass_result_v4_4(
        engine_id=subject.SOURCE_ENGINE_ID,
        pass_id="fresh_pass_1",
        collection_sha256=hashlib.sha256(b"positionwise-direction-test").hexdigest(),
        outputs=outputs,
        pit_mask=mask,
    )
    candidate = result["candidates"][1]
    assert candidate["name"] == negative_name
    adjusted = outputs[candidate["name"]] * -1.0
    values = adjusted.to_numpy(dtype=np.float64, copy=False)
    positive = np.argwhere(np.isfinite(values) & (values > 0.0))
    negative = np.argwhere(np.isfinite(values) & (values < 0.0))
    assert len(positive) > 0 and len(negative) > 0
    misplaced = adjusted.copy()
    for row_index, column_index in (positive[0], negative[0]):
        misplaced.iat[int(row_index), int(column_index)] *= -1.0

    misplaced_descriptor = subject.matrix_hash_descriptor_v4_4(misplaced)
    original_descriptor = candidate["direction_adjusted_matrix"]
    assert (
        misplaced_descriptor["magnitude_bits_sha256"]
        == original_descriptor["magnitude_bits_sha256"]
    )
    for field in (
        "positive_finite_count",
        "negative_finite_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_zero_count",
        "negative_zero_count",
    ):
        assert misplaced_descriptor[field] == original_descriptor[field]

    tampered = copy.deepcopy(result)
    tampered["candidates"][1]["direction_adjusted_matrix"] = misplaced_descriptor
    body = {
        key: value for key, value in tampered.items() if key != "result_semantic_sha256"
    }
    tampered["result_semantic_sha256"] = subject.semantic_sha256_v4_4(body)
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="negative-direction adjusted descriptor transform mismatch",
    ):
        subject.validate_engine_pass_result_v4_4(tampered)


def test_one_of_five_failure_and_value_outside_pit_block_atomic_result() -> None:
    inputs, mask = _fixture()
    outputs = subject.evaluate_local_formulas_v4_4(inputs, mask)
    broken = {name: value.copy() for name, value in outputs.items()}
    broken["pv_low_vol_of_vol_20d"].loc[:, :] = np.nan
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="no finite in-PIT observation",
    ):
        subject.build_engine_pass_result_v4_4(
            engine_id=subject.LOCAL_ENGINE_ID,
            pass_id="fresh_pass_1",
            collection_sha256=hashlib.sha256(b"one").hexdigest(),
            outputs=broken,
            pit_mask=mask,
        )

    outside = {name: value.copy() for name, value in outputs.items()}
    outside["alpha_range_position_momentum_20d"].iloc[13, 2] = 1.0
    assert mask.iloc[13, 2] is np.False_ or mask.iloc[13, 2] == False
    with pytest.raises(
        subject.FactorGovernanceExactFiveEvalV4_4Error,
        match="outside the PIT mask",
    ):
        subject.build_engine_pass_result_v4_4(
            engine_id=subject.LOCAL_ENGINE_ID,
            pass_id="fresh_pass_1",
            collection_sha256=hashlib.sha256(b"two").hexdigest(),
            outputs=outside,
            pit_mask=mask,
        )


def test_zero_denominators_ties_and_mask_reentry_match_across_engines() -> None:
    inputs, mask = _fixture()
    inputs["raw_close"].iloc[:30, :] = 10.0
    mask.iloc[25:30, 0] = False
    source = subject.evaluate_source_dag_v4_4(inputs, mask)
    local = subject.evaluate_local_formulas_v4_4(inputs, mask)
    assert _descriptors(source) == _descriptors(local)
    assert source["alpha_range_position_momentum_20d"].iloc[:20].isna().all().all()
    for matrix in source.values():
        assert matrix.iloc[25:30, 0].isna().all()
