from __future__ import annotations

import pytest

from quant_investor.factors.matrix import (
    FIELD_AMOUNT,
    FIELD_BENCHMARK_CLOSE,
    FIELD_BENCHMARK_RET,
    FIELD_CLOSE,
    FIELD_INDUSTRY,
    FIELD_RET1,
    FIELD_VOLUME,
    FIELD_VWAP,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
    add_standard_derived_fields,
    build_standard_derived_fields,
    compute_coverage,
    make_expression_result_id,
    make_factor_matrix_id,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)


def _contract(required_fields: list[str] | None = None) -> MatrixDataContract:
    symbols = ["BBB", "AAA"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    return MatrixDataContract(
        contract_id=make_matrix_contract_id(
            universe="CN",
            benchmark="CSI300",
            symbols=symbols,
            dates=dates,
        ),
        universe="CN",
        benchmark="CSI300",
        symbols=symbols,
        dates=dates,
        required_fields=required_fields or [FIELD_CLOSE, FIELD_AMOUNT, FIELD_VOLUME],
        optional_fields=[FIELD_INDUSTRY, FIELD_BENCHMARK_CLOSE],
        field_sources={FIELD_CLOSE: "fixture", FIELD_AMOUNT: "fixture", FIELD_VOLUME: "fixture"},
        point_in_time_flags={FIELD_CLOSE: True, FIELD_AMOUNT: True, FIELD_VOLUME: True},
        metadata={"fixture": True},
    )


def _bundle() -> MatrixDataBundle:
    contract = _contract()
    fields = {
        FIELD_CLOSE: [[10.0, 11.0, 12.0], [20.0, 22.0, 0.0]],
        FIELD_AMOUNT: [[100.0, 220.0, None], [1000.0, 0.0, 3600.0]],
        FIELD_VOLUME: [[10.0, 20.0, 30.0], [100.0, 0.0, 300.0]],
        FIELD_BENCHMARK_CLOSE: [[100.0, 110.0, 121.0], [100.0, 110.0, 121.0]],
    }
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(
            contract_id=contract.contract_id,
            field_names=fields.keys(),
        ),
        contract=contract,
        fields=fields,
        universe_mask=[[True, False, True], [True, True, True]],
        tradability_mask=[[True, True, True], [False, True, True]],
        industry_by_symbol={"AAA": "Tech", "BBB": "Finance"},
        metadata={"fixture": True},
    )


def test_matrix_data_contract_round_trip_and_deterministic_lists() -> None:
    contract = _contract()
    round_trip = MatrixDataContract.from_dict(contract.to_dict())

    assert round_trip.to_dict() == contract.to_dict()
    assert contract.symbols == ["AAA", "BBB"]
    assert contract.required_fields == [FIELD_AMOUNT, FIELD_CLOSE, FIELD_VOLUME]
    assert contract.optional_fields == [FIELD_BENCHMARK_CLOSE, FIELD_INDUSTRY]


def test_matrix_data_contract_rejects_duplicates_and_non_ascending_dates() -> None:
    with pytest.raises(ValueError, match="symbols"):
        MatrixDataContract(
            contract_id="contract-dup",
            universe="CN",
            benchmark="CSI300",
            symbols=["AAA", "AAA"],
            dates=["2026-01-01"],
            required_fields=[FIELD_CLOSE],
        )

    with pytest.raises(ValueError, match="dates"):
        MatrixDataContract(
            contract_id="contract-date",
            universe="CN",
            benchmark="CSI300",
            symbols=["AAA"],
            dates=["2026-01-02", "2026-01-01"],
            required_fields=[FIELD_CLOSE],
        )


def test_matrix_bundle_validates_shapes_required_fields_and_masks() -> None:
    bundle = _bundle()
    assert MatrixDataBundle.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()

    bad_fields = dict(bundle.fields)
    bad_fields[FIELD_CLOSE] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match=FIELD_CLOSE):
        MatrixDataBundle(
            bundle_id="bad-shape",
            contract=bundle.contract,
            fields=bad_fields,
        )

    missing_fields = dict(bundle.fields)
    missing_fields.pop(FIELD_CLOSE)
    with pytest.raises(ValueError, match="required field"):
        MatrixDataBundle(
            bundle_id="missing-close",
            contract=bundle.contract,
            fields=missing_fields,
        )

    with pytest.raises(ValueError, match="universe_mask"):
        MatrixDataBundle(
            bundle_id="bad-mask",
            contract=bundle.contract,
            fields=bundle.fields,
            universe_mask=[[True]],
        )


def test_industry_required_field_can_use_symbol_level_mapping() -> None:
    contract = _contract(required_fields=[FIELD_CLOSE, FIELD_INDUSTRY])
    bundle = MatrixDataBundle(
        bundle_id="bundle-industry-map",
        contract=contract,
        fields={FIELD_CLOSE: [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]},
        industry_by_symbol={"AAA": "Tech", "BBB": "Finance"},
    )

    assert bundle.industry_by_symbol == {"AAA": "Tech", "BBB": "Finance"}


def test_factor_matrix_coverage_round_trip_and_value_validation() -> None:
    values = [[1.0, None, 3.0], [4.0, None, 6.0]]
    coverage_ratio, missing_ratio = compute_coverage(values)
    matrix = FactorMatrix(
        matrix_id=make_factor_matrix_id(
            expression="close",
            symbols=["AAA", "BBB"],
            dates=["2026-01-01", "2026-01-02", "2026-01-03"],
        ),
        expression="close",
        symbols=["AAA", "BBB"],
        dates=["2026-01-01", "2026-01-02", "2026-01-03"],
        values=values,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
    )

    assert coverage_ratio == pytest.approx(4 / 6)
    assert missing_ratio == pytest.approx(2 / 6)
    assert FactorMatrix.from_dict(matrix.to_dict()).to_dict() == matrix.to_dict()

    with pytest.raises(ValueError, match="finite numeric"):
        FactorMatrix(
            matrix_id="bad-factor-matrix",
            expression="close",
            symbols=["AAA"],
            dates=["2026-01-01"],
            values=[[float("inf")]],
            coverage_ratio=1.0,
            missing_ratio=0.0,
        )


def test_build_standard_derived_fields_computes_vwap_ret1_and_benchmark_ret() -> None:
    bundle = _bundle()
    derived = build_standard_derived_fields(bundle)

    assert derived[FIELD_VWAP][0] == [10.0, 11.0, None]
    assert derived[FIELD_VWAP][1] == [10.0, None, 12.0]
    assert derived[FIELD_RET1][0][0] is None
    assert derived[FIELD_RET1][0][1] == pytest.approx(0.1)
    assert derived[FIELD_RET1][1][2] == pytest.approx(-1.0)
    assert derived[FIELD_BENCHMARK_RET][0][0] is None
    assert derived[FIELD_BENCHMARK_RET][0][1:] == pytest.approx([0.1, 0.1])
    assert derived[FIELD_BENCHMARK_RET][1][0] is None
    assert derived[FIELD_BENCHMARK_RET][1][1:] == pytest.approx([0.1, 0.1])

    enriched = add_standard_derived_fields(bundle)
    assert enriched.has_field(FIELD_VWAP)
    assert enriched.has_field(FIELD_RET1)
    assert enriched.has_field(FIELD_BENCHMARK_RET)
    assert not bundle.has_field(FIELD_VWAP)


def test_existing_benchmark_ret_is_not_overwritten_by_default() -> None:
    bundle = _bundle().with_field(
        FIELD_BENCHMARK_RET,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    enriched = add_standard_derived_fields(bundle)

    assert enriched.get_field(FIELD_BENCHMARK_RET) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_deterministic_id_helpers_return_stable_ids() -> None:
    symbols = ["BBB", "AAA"]
    dates = ["2026-01-01", "2026-01-02"]
    contract_id = make_matrix_contract_id(
        universe="CN",
        benchmark="CSI300",
        symbols=symbols,
        dates=dates,
    )

    assert contract_id == make_matrix_contract_id(
        universe="CN",
        benchmark="CSI300",
        symbols=symbols,
        dates=dates,
    )
    close_volume_bundle_id = make_matrix_bundle_id(
        contract_id=contract_id,
        field_names=[FIELD_CLOSE, FIELD_VOLUME],
    )
    volume_close_bundle_id = make_matrix_bundle_id(
        contract_id=contract_id,
        field_names=[FIELD_VOLUME, FIELD_CLOSE],
    )
    assert close_volume_bundle_id == volume_close_bundle_id
    matrix_id = make_factor_matrix_id(expression="close", symbols=symbols, dates=dates)
    assert matrix_id == make_factor_matrix_id(expression="close", symbols=symbols, dates=dates)
    assert make_expression_result_id(expression="close", matrix_id=matrix_id) == (
        make_expression_result_id(expression="close", matrix_id=matrix_id)
    )
