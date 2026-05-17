from __future__ import annotations

import json

from quant_investor.factors.matrix import (
    MatrixDataBundle,
    MatrixDataContract,
    make_matrix_bundle_id,
    make_matrix_contract_id,
)
from quant_investor.factors.tradability import (
    FIELD_AMOUNT,
    FIELD_DELISTED,
    FIELD_IS_ST,
    FIELD_LIMIT_DOWN,
    FIELD_LIMIT_UP,
    FIELD_LISTING_DATE,
    FIELD_LISTING_DAYS,
    FIELD_LOW_LIQUIDITY,
    FIELD_SUSPENDED,
    FIELD_VALID_PRICE,
    FIELD_VALID_VOLUME,
    FIELD_VOLUME,
    FIELD_VWAP,
    TRADABILITY_AUDIT_FAIL,
    TRADABILITY_AUDIT_PASS,
    TRADABILITY_AUDIT_WARN,
    TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE,
    TRADABILITY_ISSUE_DELISTED,
    TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    TRADABILITY_ISSUE_LOW_AMOUNT,
    TRADABILITY_ISSUE_NEW_LISTING,
    TRADABILITY_ISSUE_NO_VALID_PRICE,
    TRADABILITY_ISSUE_NO_VALID_VOLUME,
    TRADABILITY_ISSUE_ST_FILTERED,
    TRADABILITY_ISSUE_SUSPENDED,
    AShareTradabilityConfig,
    FactorTradabilityAuditReport,
    build_ashare_tradability_mask,
    build_listing_days_matrix,
    build_tradability_audit_report,
    build_valid_price_matrix,
    build_valid_volume_matrix,
    make_tradability_config_id,
    normalize_bool_matrix,
    normalize_float_matrix,
    render_tradability_audit_markdown,
)


SYMBOLS = ["AAA", "BBB"]
DATES = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]


def _matrix(value):
    return [[value for _date in DATES] for _symbol in SYMBOLS]


def _config(**overrides) -> AShareTradabilityConfig:
    config = AShareTradabilityConfig(config_id="placeholder", **overrides)
    config.config_id = make_tradability_config_id(config)
    return config


def _bundle(fields: dict[str, list[list[object]]]) -> MatrixDataBundle:
    contract = MatrixDataContract(
        contract_id=make_matrix_contract_id(
            universe="CN",
            benchmark="CSI300",
            symbols=SYMBOLS,
            dates=DATES,
        ),
        universe="CN",
        benchmark="CSI300",
        symbols=SYMBOLS,
        dates=DATES,
        optional_fields=list(fields),
        field_sources={field_name: "fixture" for field_name in fields},
        point_in_time_flags={field_name: True for field_name in fields},
        metadata={"preserve_symbol_order": True},
    )
    return MatrixDataBundle(
        bundle_id=make_matrix_bundle_id(contract_id=contract.contract_id, field_names=fields),
        contract=contract,
        fields=fields,
        universe_mask=_matrix(True),
        tradability_mask=_matrix(True),
        metadata={"fixture": True},
    )


def _clean_fields() -> dict[str, list[list[object]]]:
    return {
        FIELD_VWAP: [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]],
        FIELD_VOLUME: _matrix(1000.0),
        FIELD_AMOUNT: [[10000.0, 11000.0, 12000.0, 13000.0], [20000.0, 21000.0, 22000.0, 23000.0]],
        FIELD_SUSPENDED: _matrix(False),
        FIELD_LIMIT_UP: _matrix(False),
        FIELD_LIMIT_DOWN: _matrix(False),
        FIELD_IS_ST: _matrix(False),
        FIELD_DELISTED: _matrix(False),
        FIELD_LISTING_DAYS: _matrix(120),
        FIELD_VALID_PRICE: _matrix(True),
        FIELD_VALID_VOLUME: _matrix(True),
        FIELD_LOW_LIQUIDITY: _matrix(False),
    }


def test_normalize_bool_matrix_accepts_common_encodings_and_defaults() -> None:
    matrix = normalize_bool_matrix(
        [[True, 0, "1", None], ["false", "yes", "no", "true"]],
        symbols=SYMBOLS,
        dates=DATES,
        default=True,
    )

    assert matrix == [
        [True, False, True, True],
        [False, True, False, True],
    ]


def test_normalize_float_matrix_filters_invalid_values() -> None:
    matrix = normalize_float_matrix(
        [["1.5", "bad", float("nan"), float("inf")], [0, None, "-2.25", ""]],
        symbols=SYMBOLS,
        dates=DATES,
    )

    assert matrix == [[1.5, None, None, None], [0.0, None, -2.25, None]]


def test_valid_price_volume_and_listing_helpers() -> None:
    fields = _clean_fields()
    fields.pop(FIELD_VALID_PRICE)
    fields.pop(FIELD_VALID_VOLUME)
    bundle = _bundle(fields)
    config = _config()

    assert build_valid_price_matrix(bundle, config) == _matrix(True)
    assert build_valid_volume_matrix(bundle, config) == _matrix(True)
    assert build_listing_days_matrix(bundle, config) == _matrix(120)

    fields_without_vwap = _clean_fields()
    fields_without_vwap.pop(FIELD_VWAP)
    fields_without_vwap.pop(FIELD_VALID_PRICE)
    derived_bundle = _bundle(fields_without_vwap)
    assert build_valid_price_matrix(derived_bundle, config) == _matrix(True)

    listing_date_fields = _clean_fields()
    listing_date_fields.pop(FIELD_LISTING_DAYS)
    listing_date_fields[FIELD_LISTING_DATE] = _matrix("2025-12-15")
    listing_date_bundle = _bundle(listing_date_fields)
    assert build_listing_days_matrix(listing_date_bundle, config)[0][:2] == [17, 18]


def test_clean_tradability_mask_is_all_executable_and_round_trips() -> None:
    bundle = _bundle(_clean_fields())
    before = bundle.to_dict()
    mask = build_ashare_tradability_mask(bundle, config=_config())

    assert mask.can_trade_mask == _matrix(True)
    assert mask.can_buy_mask == _matrix(True)
    assert mask.can_sell_mask == _matrix(True)
    assert mask.can_hold_mask == _matrix(True)
    assert mask.research_eligible_mask == _matrix(True)
    assert all(not cell for row in mask.issue_codes_by_cell for cell in row)
    assert bundle.to_dict() == before
    assert mask.from_dict(mask.to_dict()).to_dict() == mask.to_dict()
    json.dumps(mask.to_dict(), ensure_ascii=False, sort_keys=True)


def test_tradability_mask_applies_ashare_blockers_and_warnings() -> None:
    fields = _clean_fields()
    fields[FIELD_SUSPENDED][0][0] = True
    fields[FIELD_LIMIT_UP][0][1] = True
    fields[FIELD_LIMIT_DOWN][0][2] = True
    fields[FIELD_IS_ST][0][3] = True
    fields[FIELD_DELISTED][1][0] = True
    fields[FIELD_LISTING_DAYS][1][1] = 10
    fields[FIELD_VALID_PRICE][1][2] = False
    fields[FIELD_VALID_VOLUME][1][3] = False
    fields[FIELD_AMOUNT][0][3] = 50.0
    mask = build_ashare_tradability_mask(
        _bundle(fields),
        config=_config(min_amount=1000.0),
    )

    assert mask.can_trade_mask[0][0] is False
    assert mask.can_buy_mask[0][0] is False
    assert mask.can_sell_mask[0][0] is False
    assert mask.research_eligible_mask[0][0] is False
    assert mask.issue_codes_by_cell[0][0] == [TRADABILITY_ISSUE_SUSPENDED]

    assert mask.can_buy_mask[0][1] is False
    assert mask.can_sell_mask[0][1] is True
    assert mask.issue_codes_by_cell[0][1] == [TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED]

    assert mask.can_sell_mask[0][2] is False
    assert mask.can_buy_mask[0][2] is True
    assert mask.issue_codes_by_cell[0][2] == [TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED]

    assert mask.can_buy_mask[0][3] is False
    assert mask.research_eligible_mask[0][3] is False
    assert mask.can_sell_mask[0][3] is True
    assert mask.issue_codes_by_cell[0][3] == [
        TRADABILITY_ISSUE_LOW_AMOUNT,
        TRADABILITY_ISSUE_ST_FILTERED,
    ]

    assert mask.can_trade_mask[1][0] is False
    assert mask.can_hold_mask[1][0] is False
    assert mask.issue_codes_by_cell[1][0] == [TRADABILITY_ISSUE_DELISTED]
    assert mask.can_buy_mask[1][1] is False
    assert mask.research_eligible_mask[1][1] is False
    assert mask.issue_codes_by_cell[1][1] == [TRADABILITY_ISSUE_NEW_LISTING]
    assert mask.can_trade_mask[1][2] is False
    assert mask.issue_codes_by_cell[1][2] == [TRADABILITY_ISSUE_NO_VALID_PRICE]
    assert mask.can_trade_mask[1][3] is False
    assert mask.issue_codes_by_cell[1][3] == [TRADABILITY_ISSUE_NO_VALID_VOLUME]


def test_tradability_audit_report_counts_verdicts_and_markdown() -> None:
    clean_report = build_tradability_audit_report(
        build_ashare_tradability_mask(_bundle(_clean_fields()), config=_config()),
        generated_at="2026-04-27T00:00:00Z",
    )
    assert clean_report.verdict == TRADABILITY_AUDIT_PASS

    warning_fields = _clean_fields()
    warning_fields[FIELD_LIMIT_UP][0][0] = True
    warning_report = build_tradability_audit_report(
        build_ashare_tradability_mask(_bundle(warning_fields), config=_config()),
        generated_at="2026-04-27T00:00:00Z",
    )
    assert warning_report.verdict == TRADABILITY_AUDIT_WARN
    assert warning_report.warning_count == 1

    fail_fields = _clean_fields()
    fail_fields[FIELD_SUSPENDED][0][0] = True
    fail_mask = build_ashare_tradability_mask(_bundle(fail_fields), config=_config())
    fail_report = build_tradability_audit_report(
        fail_mask,
        generated_at="2026-04-27T00:00:00Z",
    )

    assert fail_report.verdict == TRADABILITY_AUDIT_FAIL
    assert fail_report.symbols_count == 2
    assert fail_report.dates_count == 4
    assert fail_report.tradable_cell_count == 7
    assert fail_report.blocked_cell_count == 1
    assert fail_report.buy_blocked_cell_count == 1
    assert fail_report.sell_blocked_cell_count == 1
    assert fail_report.research_eligible_cell_count == 7
    assert fail_report.issue_summary == {TRADABILITY_ISSUE_SUSPENDED: 1}
    assert fail_report.blocker_count == 1
    assert FactorTradabilityAuditReport.from_dict(fail_report.to_dict()).to_dict() == fail_report.to_dict()
    markdown = render_tradability_audit_markdown(fail_report)
    assert TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE in markdown
