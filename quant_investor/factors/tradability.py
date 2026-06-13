"""Offline A-share tradability and execution-feasibility audits.

This module diagnoses whether factor backtest weights would have been
executable under local A-share trading constraints. It is audit-only: helpers
here do not change stock selection, factor scores, RiskGuard, portfolio
construction, target weights, orders, providers, LLMs, or execution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.factors.backtest import FactorWeightMatrix, SingleFactorBacktestRun
from quant_investor.factors.matrix import MatrixDataBundle
from quant_investor.factors.tradability_types import (
    DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME,
    DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME,
    DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR,
    DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME,
    DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME,
    DEFAULT_TRADABILITY_MASKS_FILENAME,
    EXECUTION_AUDIT_STATUS_BLOCKED,
    EXECUTION_AUDIT_STATUS_FEASIBLE,
    EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE,
    EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE,
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_DELISTED,
    FIELD_IS_ST,
    FIELD_LIMIT_DOWN,
    FIELD_LIMIT_UP,
    FIELD_LISTING_DATE,
    FIELD_LISTING_DAYS,
    FIELD_LOW_LIQUIDITY,
    FIELD_OPEN,
    FIELD_SUSPENDED,
    FIELD_VALID_PRICE,
    FIELD_VALID_VOLUME,
    FIELD_VOLUME,
    FIELD_VWAP,
    TRADE_DIRECTION_BUY,
    TRADE_DIRECTION_HOLD,
    TRADE_DIRECTION_SELL,
    TRADABILITY_AUDIT_FAIL,
    TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE,
    TRADABILITY_AUDIT_PASS,
    TRADABILITY_AUDIT_WARN,
    TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION,
    TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION,
    TRADABILITY_ISSUE_BLOCKER,
    TRADABILITY_ISSUE_DELISTED,
    TRADABILITY_ISSUE_INFO,
    TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED,
    TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED,
    TRADABILITY_ISSUE_LOW_AMOUNT,
    TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH,
    TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD,
    TRADABILITY_ISSUE_NEW_LISTING,
    TRADABILITY_ISSUE_NO_VALID_PRICE,
    TRADABILITY_ISSUE_NO_VALID_VOLUME,
    TRADABILITY_ISSUE_ST_FILTERED,
    TRADABILITY_ISSUE_SUSPENDED,
    TRADABILITY_ISSUE_WARNING,
    AShareTradabilityConfig,
    AShareTradabilityMask,
    ExecutionTransitionAuditRecord,
    FactorExecutionFeasibilityReport,
    FactorTradabilityAuditReport,
    FactorTradabilityIssue,
    _FLOAT_TOLERANCE,
    _as_bool_value,
    _coerce_metadata,
    _empty_bool_matrix,
    _empty_issue_tensor,
    _issue_severity,
    _issue_sort_key,
    _parse_iso_date,
    _record_sort_key,
    _to_finite_float,
    _validate_matrix_shape,
    make_execution_feasibility_report_id,
    make_execution_transition_record_id,
    make_tradability_audit_report_id,
    make_tradability_config_id,
    make_tradability_issue_id,
    make_tradability_mask_id,
)

from quant_investor.factors.tradability_rendering import (
    render_execution_feasibility_markdown,
    render_tradability_audit_markdown,
)
from quant_investor.versioning import (
    FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION,
    FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
)


def get_matrix_field_optional(
    bundle: MatrixDataBundle,
    field_name: str,
) -> list[list[Any]] | None:
    if not bundle.has_field(field_name):
        return None
    return bundle.get_field(field_name)


def normalize_bool_matrix(
    values: Sequence[Sequence[Any]] | None,
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
    default: bool,
) -> list[list[bool]]:
    if values is None:
        return _empty_bool_matrix(symbols, dates, default)
    _validate_matrix_shape(values, symbols=symbols, dates=dates, field_name="bool_matrix")
    return [
        [_as_bool_value(value, default=default) for value in row]
        for row in values
    ]


def normalize_float_matrix(
    values: Sequence[Sequence[Any]] | None,
    *,
    symbols: Sequence[str],
    dates: Sequence[str],
) -> list[list[float | None]]:
    if values is None:
        return [[None for _ in dates] for _ in symbols]
    _validate_matrix_shape(values, symbols=symbols, dates=dates, field_name="float_matrix")
    return [[_to_finite_float(value) for value in row] for row in values]


def build_valid_price_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[bool]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    if not config.require_valid_price:
        return _empty_bool_matrix(symbols, dates, True)
    explicit = get_matrix_field_optional(bundle, config.valid_price_field)
    if explicit is not None:
        return normalize_bool_matrix(explicit, symbols=symbols, dates=dates, default=False)
    price_matrix = normalize_float_matrix(
        get_matrix_field_optional(bundle, config.price_field),
        symbols=symbols,
        dates=dates,
    )
    if config.price_field == FIELD_VWAP and all(
        value is None for row in price_matrix for value in row
    ):
        amount_matrix = normalize_float_matrix(
            get_matrix_field_optional(bundle, config.amount_field),
            symbols=symbols,
            dates=dates,
        )
        volume_matrix = normalize_float_matrix(
            get_matrix_field_optional(bundle, config.volume_field),
            symbols=symbols,
            dates=dates,
        )
        return [
            [
                amount is not None
                and volume is not None
                and amount > 0.0
                and volume > 0.0
                and amount / volume > 0.0
                for amount, volume in zip(amount_row, volume_row)
            ]
            for amount_row, volume_row in zip(amount_matrix, volume_matrix)
        ]
    return [[value is not None and value > 0.0 for value in row] for row in price_matrix]


def build_valid_volume_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[bool]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    if not config.require_valid_volume:
        return _empty_bool_matrix(symbols, dates, True)
    explicit = get_matrix_field_optional(bundle, config.valid_volume_field)
    if explicit is not None:
        return normalize_bool_matrix(explicit, symbols=symbols, dates=dates, default=False)
    volume_matrix = normalize_float_matrix(
        get_matrix_field_optional(bundle, config.volume_field),
        symbols=symbols,
        dates=dates,
    )
    return [[value is not None and value > 0.0 for value in row] for row in volume_matrix]


def build_listing_days_matrix(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[list[int | None]]:
    symbols = bundle.contract.symbols
    dates = bundle.contract.dates
    listing_days_values = get_matrix_field_optional(bundle, config.listing_days_field)
    if listing_days_values is not None:
        numeric = normalize_float_matrix(listing_days_values, symbols=symbols, dates=dates)
        return [[int(value) if value is not None else None for value in row] for row in numeric]

    listing_date_values = get_matrix_field_optional(bundle, config.listing_date_field)
    if listing_date_values is not None:
        _validate_matrix_shape(
            listing_date_values,
            symbols=symbols,
            dates=dates,
            field_name=config.listing_date_field,
        )
        output: list[list[int | None]] = []
        for row in listing_date_values:
            output_row: list[int | None] = []
            for current_date_text, listing_date_value in zip(dates, row):
                current_date = _parse_iso_date(current_date_text)
                listed_date = _parse_iso_date(listing_date_value)
                if current_date is None or listed_date is None:
                    output_row.append(None)
                else:
                    output_row.append((current_date - listed_date).days)
            output.append(output_row)
        return output

    metadata_candidates = [
        bundle.metadata.get(config.listing_date_field),
        bundle.metadata.get("listing_dates"),
        bundle.metadata.get("listing_date_by_symbol"),
    ]
    for candidate in metadata_candidates:
        if isinstance(candidate, Mapping):
            output = []
            for symbol in symbols:
                listed_date = _parse_iso_date(candidate.get(symbol))
                symbol_listing_days: list[int | None] = []
                for current_date_text in dates:
                    current_date = _parse_iso_date(current_date_text)
                    if current_date is None or listed_date is None:
                        symbol_listing_days.append(None)
                    else:
                        symbol_listing_days.append((current_date - listed_date).days)
                output.append(symbol_listing_days)
            return output
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            if len(candidate) == len(symbols):
                output = []
                for listed_date_value in candidate:
                    listed_date = _parse_iso_date(listed_date_value)
                    row = []
                    for current_date_text in dates:
                        current_date = _parse_iso_date(current_date_text)
                        if current_date is None or listed_date is None:
                            row.append(None)
                        else:
                            row.append((current_date - listed_date).days)
                    output.append(row)
                return output
    return [[None for _ in dates] for _ in symbols]


def _default_tradability_config() -> AShareTradabilityConfig:
    config = AShareTradabilityConfig(config_id="placeholder")
    config.config_id = make_tradability_config_id(config)
    return config


def _missing_tradability_fields(
    bundle: MatrixDataBundle,
    config: AShareTradabilityConfig,
) -> list[str]:
    field_names = {
        config.suspended_field,
        config.limit_up_field,
        config.limit_down_field,
        config.is_st_field,
        config.delisted_field,
    }
    missing = {field for field in field_names if not bundle.has_field(field)}
    has_listing_metadata = any(
        key in bundle.metadata
        for key in (
            config.listing_date_field,
            "listing_dates",
            "listing_date_by_symbol",
        )
    )
    if (
        not bundle.has_field(config.listing_days_field)
        and not bundle.has_field(config.listing_date_field)
        and not has_listing_metadata
    ):
        missing.add(config.listing_days_field)
        missing.add(config.listing_date_field)
    if (
        config.min_amount is not None
        and not bundle.has_field(config.amount_field)
        and not bundle.has_field(FIELD_LOW_LIQUIDITY)
    ):
        missing.add(config.amount_field)
        missing.add(FIELD_LOW_LIQUIDITY)
    if config.require_valid_price:
        has_explicit_price_flag = bundle.has_field(config.valid_price_field)
        has_price = bundle.has_field(config.price_field)
        has_derivable_vwap = (
            config.price_field == FIELD_VWAP
            and bundle.has_field(config.amount_field)
            and bundle.has_field(config.volume_field)
        )
        if not (has_explicit_price_flag or has_price or has_derivable_vwap):
            missing.add(config.valid_price_field)
            missing.add(config.price_field)
    if (
        config.require_valid_volume
        and not bundle.has_field(config.valid_volume_field)
        and not bundle.has_field(config.volume_field)
    ):
        missing.add(config.valid_volume_field)
        missing.add(config.volume_field)
    return sorted(missing)


def _append_issue(codes: list[str], issue_code: str) -> None:
    if issue_code not in codes:
        codes.append(issue_code)


def build_ashare_tradability_mask(
    bundle: MatrixDataBundle,
    *,
    config: AShareTradabilityConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AShareTradabilityMask:
    resolved_config = config or _default_tradability_config()
    symbols = list(bundle.contract.symbols)
    dates = list(bundle.contract.dates)
    can_trade = _empty_bool_matrix(symbols, dates, True)
    can_buy = _empty_bool_matrix(symbols, dates, True)
    can_sell = _empty_bool_matrix(symbols, dates, True)
    can_hold = _empty_bool_matrix(symbols, dates, True)
    research_eligible = _empty_bool_matrix(symbols, dates, True)
    issue_codes = _empty_issue_tensor(symbols, dates)

    suspended = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.suspended_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    limit_up = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.limit_up_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    limit_down = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.limit_down_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    is_st = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.is_st_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    delisted = normalize_bool_matrix(
        get_matrix_field_optional(bundle, resolved_config.delisted_field),
        symbols=symbols,
        dates=dates,
        default=False,
    )
    valid_price = build_valid_price_matrix(bundle, resolved_config)
    valid_volume = build_valid_volume_matrix(bundle, resolved_config)
    listing_days = build_listing_days_matrix(bundle, resolved_config)
    amount = normalize_float_matrix(
        get_matrix_field_optional(bundle, resolved_config.amount_field),
        symbols=symbols,
        dates=dates,
    )
    low_liquidity = normalize_bool_matrix(
        get_matrix_field_optional(bundle, FIELD_LOW_LIQUIDITY),
        symbols=symbols,
        dates=dates,
        default=False,
    )

    for row_index, _symbol in enumerate(symbols):
        for column_index, _date in enumerate(dates):
            cell_issues = issue_codes[row_index][column_index]
            if suspended[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                if resolved_config.suspension_filter:
                    research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_SUSPENDED)
            if delisted[row_index][column_index] and resolved_config.delisted_filter:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                can_hold[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_DELISTED)
            if is_st[row_index][column_index] and resolved_config.st_filter:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_ST_FILTERED)
            if limit_up[row_index][column_index] and resolved_config.limit_up_blocks_buy:
                can_buy[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED)
            if limit_down[row_index][column_index] and resolved_config.limit_down_blocks_sell:
                can_sell[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED)
            listed_days = listing_days[row_index][column_index]
            if listed_days is not None and listed_days < resolved_config.min_listing_days:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NEW_LISTING)
            if not valid_price[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NO_VALID_PRICE)
            if not valid_volume[row_index][column_index]:
                can_trade[row_index][column_index] = False
                can_buy[row_index][column_index] = False
                can_sell[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_NO_VALID_VOLUME)
            amount_value = amount[row_index][column_index]
            low_amount = (
                low_liquidity[row_index][column_index]
                or (
                    resolved_config.min_amount is not None
                    and amount_value is not None
                    and amount_value < resolved_config.min_amount
                )
            )
            if low_amount:
                can_buy[row_index][column_index] = False
                research_eligible[row_index][column_index] = False
                _append_issue(cell_issues, TRADABILITY_ISSUE_LOW_AMOUNT)
            issue_codes[row_index][column_index] = sorted(cell_issues)

    resolved_metadata = _coerce_metadata(metadata)
    missing_fields = _missing_tradability_fields(bundle, resolved_config)
    resolved_metadata.update(
        {
            "factor_tradability_audit_schema_version": FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "missing_tradability_fields": missing_fields,
        }
    )
    return AShareTradabilityMask(
        mask_id=make_tradability_mask_id(
            symbols=symbols,
            dates=dates,
            config_id=resolved_config.config_id,
        ),
        symbols=symbols,
        dates=dates,
        can_trade_mask=can_trade,
        can_buy_mask=can_buy,
        can_sell_mask=can_sell,
        can_hold_mask=can_hold,
        research_eligible_mask=research_eligible,
        issue_codes_by_cell=issue_codes,
        config=resolved_config,
        metadata=resolved_metadata,
    )


def _make_issue(
    *,
    symbol: str | None,
    date_value: str | None,
    issue_code: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorTradabilityIssue:
    return FactorTradabilityIssue(
        issue_id=make_tradability_issue_id(
            symbol=symbol,
            date=date_value,
            issue_code=issue_code,
            message=message,
        ),
        symbol=symbol,
        date=date_value,
        issue_code=issue_code,
        severity=_issue_severity(issue_code),
        message=message,
        metadata=dict(metadata or {}),
    )


def _issues_from_mask(mask: AShareTradabilityMask) -> list[FactorTradabilityIssue]:
    issues: list[FactorTradabilityIssue] = []
    for row_index, symbol in enumerate(mask.symbols):
        for column_index, date_value in enumerate(mask.dates):
            for issue_code in mask.issue_codes_by_cell[row_index][column_index]:
                issues.append(
                    _make_issue(
                        symbol=symbol,
                        date_value=date_value,
                        issue_code=issue_code,
                        message=f"{issue_code} observed for {symbol} on {date_value}.",
                        metadata={
                            "mask_id": mask.mask_id,
                            "row_index": row_index,
                            "column_index": column_index,
                        },
                    )
                )
    for field_name in mask.metadata.get("missing_tradability_fields", []) or []:
        issues.append(
            _make_issue(
                symbol=None,
                date_value=None,
                issue_code=TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD,
                message=f"tradability field missing: {field_name}.",
                metadata={"field_name": field_name, "mask_id": mask.mask_id},
            )
        )
    by_id: dict[str, FactorTradabilityIssue] = {}
    for issue in issues:
        by_id.setdefault(issue.issue_id, issue)
    return sorted(by_id.values(), key=_issue_sort_key)


def _issue_summary(issues: Sequence[FactorTradabilityIssue]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for issue in issues:
        summary[issue.issue_code] = summary.get(issue.issue_code, 0) + 1
    return dict(sorted(summary.items()))


def _verdict_from_issues(issues: Sequence[FactorTradabilityIssue]) -> str:
    if any(issue.severity == TRADABILITY_ISSUE_BLOCKER for issue in issues):
        return TRADABILITY_AUDIT_FAIL
    if any(issue.severity == TRADABILITY_ISSUE_WARNING for issue in issues):
        return TRADABILITY_AUDIT_WARN
    return TRADABILITY_AUDIT_PASS


def build_tradability_audit_report(
    mask: AShareTradabilityMask,
    *,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorTradabilityAuditReport:
    issues = _issues_from_mask(mask)
    cell_count = len(mask.symbols) * len(mask.dates)
    tradable_count = sum(1 for row in mask.can_trade_mask for value in row if value)
    buy_blocked_count = sum(1 for row in mask.can_buy_mask for value in row if not value)
    sell_blocked_count = sum(1 for row in mask.can_sell_mask for value in row if not value)
    research_eligible_count = sum(
        1 for row in mask.research_eligible_mask for value in row if value
    )
    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "factor_tradability_audit_schema_version": FACTOR_TRADABILITY_AUDIT_SCHEMA_VERSION,
            "non_runtime_impact": True,
            "config": mask.config.to_dict(),
        }
    )
    return FactorTradabilityAuditReport(
        report_id=make_tradability_audit_report_id(
            mask_id=mask.mask_id,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        mask_id=mask.mask_id,
        symbols_count=len(mask.symbols),
        dates_count=len(mask.dates),
        tradable_cell_count=tradable_count,
        blocked_cell_count=cell_count - tradable_count,
        buy_blocked_cell_count=buy_blocked_count,
        sell_blocked_cell_count=sell_blocked_count,
        research_eligible_cell_count=research_eligible_count,
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_INFO),
        issue_summary=_issue_summary(issues),
        issues=issues,
        verdict=_verdict_from_issues(issues),
        metadata=resolved_metadata,
    )


def _alignment_value(alignment: Any, field_name: str) -> Any:
    if isinstance(alignment, Mapping):
        return alignment.get(field_name)
    return getattr(alignment, field_name, None)


def _execution_alignments(
    *,
    weight_matrix: FactorWeightMatrix,
    alignments: Sequence[Any] | None,
    run: SingleFactorBacktestRun | None,
) -> list[dict[str, str]]:
    if alignments is not None:
        output: list[dict[str, str]] = []
        for alignment in alignments:
            signal_date = _alignment_value(alignment, "signal_date")
            execution_date = (
                _alignment_value(alignment, "execution_start_date")
                or _alignment_value(alignment, "execution_date")
            )
            if signal_date is None or execution_date is None:
                continue
            output.append(
                {
                    "signal_date": str(signal_date),
                    "execution_date": str(execution_date),
                }
            )
        return sorted(output, key=lambda item: (item["execution_date"], item["signal_date"]))
    if run is not None:
        return [
            {
                "signal_date": record.signal_date,
                "execution_date": record.execution_start_date,
            }
            for record in sorted(run.daily_records, key=lambda record: record.execution_start_date)
        ]
    output = []
    for index, signal_date in enumerate(weight_matrix.dates[:-1]):
        output.append({"signal_date": signal_date, "execution_date": weight_matrix.dates[index + 1]})
    return output


def _weight_at(
    weight_matrix: FactorWeightMatrix,
    row_index: int,
    column_index: int,
) -> float:
    value = weight_matrix.net_weights[row_index][column_index]
    number = _to_finite_float(value)
    return number if number is not None else 0.0


def _trade_direction(trade_weight: float) -> str:
    if trade_weight > _FLOAT_TOLERANCE:
        return TRADE_DIRECTION_BUY
    if trade_weight < -_FLOAT_TOLERANCE:
        return TRADE_DIRECTION_SELL
    return TRADE_DIRECTION_HOLD


def _mask_cell(
    tradability_mask: AShareTradabilityMask,
    *,
    row_index: int,
    execution_date: str,
) -> tuple[bool, bool, bool, bool, list[str], dict[str, Any]]:
    try:
        column_index = tradability_mask.dates.index(execution_date)
    except ValueError:
        return (
            False,
            False,
            False,
            False,
            [TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH],
            {"execution_date_in_mask": False},
        )
    return (
        tradability_mask.can_buy_mask[row_index][column_index],
        tradability_mask.can_sell_mask[row_index][column_index],
        tradability_mask.can_trade_mask[row_index][column_index],
        tradability_mask.can_hold_mask[row_index][column_index],
        list(tradability_mask.issue_codes_by_cell[row_index][column_index]),
        {"execution_date_in_mask": True, "mask_column_index": column_index},
    )


def _record_issues(record: ExecutionTransitionAuditRecord) -> list[FactorTradabilityIssue]:
    issues: list[FactorTradabilityIssue] = []
    for issue_code in record.issue_codes:
        message = f"{issue_code} affects {record.symbol} on {record.execution_date}."
        if issue_code == TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION:
            message = "buy transition blocked on execution date."
        elif issue_code == TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION:
            message = "sell transition blocked on execution date."
        issues.append(
            _make_issue(
                symbol=record.symbol,
                date_value=record.execution_date,
                issue_code=issue_code,
                message=message,
                metadata={
                    "record_id": record.record_id,
                    "signal_date": record.signal_date,
                    "trade_direction": record.trade_direction,
                    "trade_weight": record.trade_weight,
                },
            )
        )
    return issues


def _mask_has_warning_issue(mask: AShareTradabilityMask) -> bool:
    for row in mask.issue_codes_by_cell:
        for codes in row:
            if any(_issue_severity(code) == TRADABILITY_ISSUE_WARNING for code in codes):
                return True
    return False


def _has_short_leg(weight_matrix: FactorWeightMatrix) -> bool:
    if str(weight_matrix.metadata.get("mode", "")).lower() == "long_short":
        return True
    for row in weight_matrix.short_weights:
        for value in row:
            number = _to_finite_float(value)
            if number is not None and abs(number) > _FLOAT_TOLERANCE:
                return True
    return False


def audit_factor_weight_execution_feasibility(
    *,
    weight_matrix: FactorWeightMatrix,
    tradability_mask: AShareTradabilityMask,
    alignments: Sequence[Any] | None = None,
    run: SingleFactorBacktestRun | None = None,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> FactorExecutionFeasibilityReport:
    if list(weight_matrix.symbols) != list(tradability_mask.symbols):
        raise ValueError("weight_matrix symbols must match tradability_mask symbols.")
    date_to_index = {date_value: index for index, date_value in enumerate(weight_matrix.dates)}
    resolved_alignments = _execution_alignments(
        weight_matrix=weight_matrix,
        alignments=alignments,
        run=run,
    )
    previous_weights = {symbol: 0.0 for symbol in weight_matrix.symbols}
    records: list[ExecutionTransitionAuditRecord] = []

    for alignment in resolved_alignments:
        signal_date = alignment["signal_date"]
        execution_date = alignment["execution_date"]
        signal_index = date_to_index.get(signal_date)
        if signal_index is None:
            continue
        for row_index, symbol in enumerate(weight_matrix.symbols):
            previous_weight = previous_weights[symbol]
            target_weight = _weight_at(weight_matrix, row_index, signal_index)
            trade_weight = target_weight - previous_weight
            direction = _trade_direction(trade_weight)
            can_buy, can_sell, can_trade, can_hold, issue_codes, mask_metadata = _mask_cell(
                tradability_mask,
                row_index=row_index,
                execution_date=execution_date,
            )
            status = EXECUTION_AUDIT_STATUS_FEASIBLE
            if direction == TRADE_DIRECTION_BUY:
                if not can_buy:
                    _append_issue(issue_codes, TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION)
                if not can_buy or not can_trade:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            elif direction == TRADE_DIRECTION_SELL:
                if not can_sell:
                    _append_issue(issue_codes, TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION)
                if not can_sell or not can_trade:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            else:
                if not can_hold:
                    status = EXECUTION_AUDIT_STATUS_BLOCKED
            issue_codes = sorted(set(issue_codes))
            records.append(
                ExecutionTransitionAuditRecord(
                    record_id=make_execution_transition_record_id(
                        symbol=symbol,
                        signal_date=signal_date,
                        execution_date=execution_date,
                        target_weight=target_weight,
                    ),
                    symbol=symbol,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    previous_weight=previous_weight,
                    target_weight=target_weight,
                    trade_weight=trade_weight,
                    trade_direction=direction,
                    can_buy=can_buy,
                    can_sell=can_sell,
                    can_trade=can_trade,
                    status=status,
                    issue_codes=issue_codes,
                    metadata={
                        "mask_id": tradability_mask.mask_id,
                        "signal_index": signal_index,
                        "row_index": row_index,
                        **mask_metadata,
                    },
                )
            )
            previous_weights[symbol] = target_weight

    records = sorted(records, key=_record_sort_key)
    issue_by_id: dict[str, FactorTradabilityIssue] = {}
    for record in records:
        for issue in _record_issues(record):
            issue_by_id.setdefault(issue.issue_id, issue)
    issues = sorted(issue_by_id.values(), key=_issue_sort_key)
    blocked_count = sum(
        1 for record in records if record.status == EXECUTION_AUDIT_STATUS_BLOCKED
    )
    verdict = TRADABILITY_AUDIT_PASS
    if blocked_count:
        verdict = TRADABILITY_AUDIT_FAIL
    elif _mask_has_warning_issue(tradability_mask):
        verdict = TRADABILITY_AUDIT_WARN

    resolved_metadata = _coerce_metadata(metadata)
    resolved_metadata.update(
        {
            "factor_execution_feasibility_audit_schema_version": (
                FACTOR_EXECUTION_FEASIBILITY_AUDIT_SCHEMA_VERSION
            ),
            "non_runtime_impact": True,
            "no_pnl_adjustment": True,
        }
    )
    if _has_short_leg(weight_matrix):
        resolved_metadata["short_leg_is_research_analytic_not_cash_equity_short"] = True
    return FactorExecutionFeasibilityReport(
        report_id=make_execution_feasibility_report_id(
            backtest_run_id=run.run_id if run is not None else weight_matrix.metadata.get("run_id"),
            weight_matrix_id=weight_matrix.weights_id,
            generated_at=generated_at,
        ),
        generated_at=generated_at,
        factor_matrix_id=weight_matrix.factor_matrix_id,
        backtest_run_id=run.run_id if run is not None else weight_matrix.metadata.get("run_id"),
        weight_matrix_id=weight_matrix.weights_id,
        mask_id=tradability_mask.mask_id,
        total_transitions=len(records),
        feasible_transitions=sum(
            1 for record in records if record.status == EXECUTION_AUDIT_STATUS_FEASIBLE
        ),
        blocked_transitions=blocked_count,
        partially_feasible_transitions=sum(
            1 for record in records
            if record.status == EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE
        ),
        blocked_buy_count=sum(
            1 for record in records
            if TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION in record.issue_codes
        ),
        blocked_sell_count=sum(
            1 for record in records
            if TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION in record.issue_codes
        ),
        blocked_symbols=sorted(
            {
                record.symbol for record in records
                if record.status == EXECUTION_AUDIT_STATUS_BLOCKED
            }
        ),
        issue_count=len(issues),
        blocker_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_BLOCKER),
        warning_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_WARNING),
        info_count=sum(1 for issue in issues if issue.severity == TRADABILITY_ISSUE_INFO),
        transition_records=records,
        issues=issues,
        verdict=verdict,
        metadata=resolved_metadata,
    )


__all__ = [
    "TRADABILITY_AUDIT_PASS",
    "TRADABILITY_AUDIT_WARN",
    "TRADABILITY_AUDIT_FAIL",
    "TRADABILITY_ISSUE_INFO",
    "TRADABILITY_ISSUE_WARNING",
    "TRADABILITY_ISSUE_BLOCKER",
    "TRADABILITY_ISSUE_SUSPENDED",
    "TRADABILITY_ISSUE_LIMIT_UP_BUY_BLOCKED",
    "TRADABILITY_ISSUE_LIMIT_DOWN_SELL_BLOCKED",
    "TRADABILITY_ISSUE_ST_FILTERED",
    "TRADABILITY_ISSUE_DELISTED",
    "TRADABILITY_ISSUE_NEW_LISTING",
    "TRADABILITY_ISSUE_NO_VALID_PRICE",
    "TRADABILITY_ISSUE_NO_VALID_VOLUME",
    "TRADABILITY_ISSUE_LOW_AMOUNT",
    "TRADABILITY_ISSUE_BLOCKED_BUY_TRANSITION",
    "TRADABILITY_ISSUE_BLOCKED_SELL_TRANSITION",
    "TRADABILITY_ISSUE_MISSING_TRADABILITY_FIELD",
    "TRADABILITY_ISSUE_MASK_SHAPE_MISMATCH",
    "EXECUTION_AUDIT_STATUS_FEASIBLE",
    "EXECUTION_AUDIT_STATUS_PARTIALLY_FEASIBLE",
    "EXECUTION_AUDIT_STATUS_BLOCKED",
    "TRADE_DIRECTION_BUY",
    "TRADE_DIRECTION_SELL",
    "TRADE_DIRECTION_HOLD",
    "FIELD_SUSPENDED",
    "FIELD_LIMIT_UP",
    "FIELD_LIMIT_DOWN",
    "FIELD_IS_ST",
    "FIELD_DELISTED",
    "FIELD_LISTING_DAYS",
    "FIELD_LISTING_DATE",
    "FIELD_VALID_PRICE",
    "FIELD_VALID_VOLUME",
    "FIELD_LOW_LIQUIDITY",
    "FIELD_AMOUNT",
    "FIELD_VOLUME",
    "FIELD_OPEN",
    "FIELD_CLOSE",
    "FIELD_VWAP",
    "DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR",
    "DEFAULT_TRADABILITY_MASKS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME",
    "TRADABILITY_AUDIT_NON_RUNTIME_IMPACT_NOTE",
    "EXECUTION_FEASIBILITY_NON_RUNTIME_IMPACT_NOTE",
    "AShareTradabilityConfig",
    "FactorTradabilityIssue",
    "AShareTradabilityMask",
    "ExecutionTransitionAuditRecord",
    "FactorExecutionFeasibilityReport",
    "FactorTradabilityAuditReport",
    "make_tradability_config_id",
    "make_tradability_issue_id",
    "make_tradability_mask_id",
    "make_execution_transition_record_id",
    "make_execution_feasibility_report_id",
    "make_tradability_audit_report_id",
    "get_matrix_field_optional",
    "normalize_bool_matrix",
    "normalize_float_matrix",
    "build_valid_price_matrix",
    "build_valid_volume_matrix",
    "build_listing_days_matrix",
    "build_ashare_tradability_mask",
    "build_tradability_audit_report",
    "audit_factor_weight_execution_feasibility",
    "render_tradability_audit_markdown",
    "render_execution_feasibility_markdown",
]
