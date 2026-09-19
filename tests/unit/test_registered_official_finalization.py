"""Pure registered-row finalization; source registration and Store CAS are separate."""

import builtins
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from quant_investor.strategy_records import performance as perf
from quant_investor.strategy_records.store import StrategyRecordStoreError
from test_strategy_record_performance import _projection_catalog


def case(amount=Decimal("0")):
    catalog = _projection_catalog()
    normalized, _, _ = perf.normalize_registered_projection(catalog)
    baseline = perf.build_seed_rows(normalized, catalog=catalog)
    source = {
        "record": "20260104_1000",
        "source_record": baseline[-1]["record_id"],
        "data_date": "2026-01-04",
        "execution_kind": "applied_effective_ledger",
        "execution_status": "owner_declared_manual_execution_applied",
        "official_valuation": False,
        "manual_manifest_sha256": "a" * 64,
        "ledger_sha256": "b" * 64,
        "financial_state_sha256": "c" * 64,
        "positions": [
            {"symbol": "000001.SZ", "shares": 10000, "avg_cost": 50, "cost_basis": 500000}
        ],
        "accounting": {
            "cash_after": Decimal("520000") + amount,
            "market_value_after": Decimal("500000"),
            "total_value_after": Decimal("1020000") + amount,
            "portfolio_pnl_after": Decimal("20000") + amount,
        },
    }
    units = None
    if amount:
        units, _ = perf.apply_flow_neutral_unitization(
            pre_nav=baseline[-1]["raw_nav_cny"],
            pre_units=baseline[-1]["unit_count"],
            pre_unit_nav=baseline[-1]["unit_nav"],
            amount=amount,
        )
    rows = perf.extend_performance_rows(
        baseline,
        strict_record=source,
        manual_manifest_sha256=source["manual_manifest_sha256"],
        ledger_parquet_sha256=source["ledger_sha256"],
        financial_state_sha256=source["financial_state_sha256"],
        post_flow_unit_count=units,
        external_flow_amount=amount,
    )
    target = {
        **deepcopy(source),
        "record": "20260104_200001-b01",
        "source_record": source["record"],
        "execution_kind": "carry_forward",
        "execution_status": "no_action_carry_forward_official_valuation",
        "official_valuation": True,
        "valuation_completeness_passed": True,
        "valuation_status": "OFFICIAL_STRICT_MARKET_CLOSE_COMPLETE",
        "price_basis": "strict_parquet_market_close_hash_bound",
        "publication_class": perf.BATCH_CATCH_UP_OFFICIAL_VALUATION,
        "funding": None,
        "funding_correction": None,
        "manual_manifest_sha256": "d" * 64,
        "ledger_sha256": "e" * 64,
        "financial_state_sha256": "f" * 64,
        "accounting": {
            **source["accounting"],
            "market_value_after": Decimal("550000"),
            "total_value_after": Decimal("1070000") + amount,
            "portfolio_pnl_after": Decimal("70000") + amount,
        },
    }
    kwargs = {
        "strict_record": target,
        "manual_manifest_sha256": target["manual_manifest_sha256"],
        "ledger_parquet_sha256": target["ledger_sha256"],
        "financial_state_sha256": target["financial_state_sha256"],
        "official_close_source": source,
    }
    return rows, kwargs


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("100000"), Decimal("-100000")])
def test_official_close_replaces_intraday_and_preserves_applied_flow(tmp_path, amount):
    rows, kwargs = case(amount)
    before = deepcopy((rows, kwargs))
    result = perf.extend_performance_rows(rows, **kwargs)
    assert (rows, kwargs) == before
    assert result[:-1] == rows[:-1]
    assert len(result) == len(rows)
    closed = result[-1]
    assert closed["record_id"] == kwargs["strict_record"]["record"]
    assert closed["valuation_at"] == "2026-01-04T07:00:00Z"
    assert closed["evidence_kind"] == "REGISTERED_OFFICIAL_FINANCIAL_STATE"
    assert closed["unit_count"] == rows[-1]["unit_count"]
    assert closed["excluded_external_flow_cny"] == rows[-1]["excluded_external_flow_cny"]
    assert closed["adjusted_nav_cny"] == Decimal("1070000.0000")
    expected_return = Decimal("50000") / (Decimal("1020000") + amount)
    assert abs(closed["interval_return"] - expected_return) <= perf.UNIT_TOLERANCE
    assert closed["unit_nav"] == perf.unit_decimal(
        (Decimal("1070000") + amount) / rows[-1]["unit_count"], label="expected unit NAV"
    )
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    assert perf.write_deterministic_parquet(result, first) == perf.write_deterministic_parquet(
        result, second
    )
    assert first.read_bytes() == second.read_bytes()
    assert perf.read_performance_parquet(first) == result


def test_same_day_still_conflicts_without_exact_source():
    rows, kwargs = case()
    kwargs.pop("official_close_source")
    with pytest.raises(StrategyRecordStoreError, match="SAME_DATE"):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("record", "20260103_1100"),
        ("source_record", "20260101_1000"),
        ("data_date", "2026-01-05"),
        ("official_valuation", True),
        ("official_valuation", 0),
        ("execution_kind", "carry_forward"),
        ("funding_correction", {"reversed_amount": 1}),
        ("manual_manifest_sha256", "0" * 64),
        ("ledger_sha256", "0" * 64),
        ("financial_state_sha256", None),
    ],
)
def test_source_identity_drift_rejects_without_mutation(field, value):
    rows, kwargs = case()
    kwargs["official_close_source"][field] = value
    before = deepcopy((rows, kwargs))
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)
    assert (rows, kwargs) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("record", "20260104_1600"),
        ("record", "20260103_200001-b01"),
        ("source_record", "20260103_1000"),
        ("data_date", "2026-01-05"),
        ("official_valuation", False),
        ("official_valuation", 1),
        ("valuation_completeness_passed", False),
        ("execution_kind", "applied_effective_ledger"),
        ("execution_status", "no_action_carry_forward"),
        ("valuation_status", "INTRADAY"),
        ("price_basis", "quote"),
        ("publication_class", "CORRECTION"),
        ("funding", {"amount": 0, "offsetting": True}),
        ("funding_correction", {"amount": 0}),
        ("manual_manifest_sha256", "0" * 64),
        ("ledger_sha256", "0" * 64),
        ("financial_state_sha256", "0" * 64),
    ],
)
def test_target_profile_drift_rejects(field, value):
    rows, kwargs = case()
    kwargs["strict_record"][field] = value
    before = deepcopy((rows, kwargs))
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)
    assert (rows, kwargs) == before


@pytest.mark.parametrize("field", ["funding", "funding_correction"])
def test_missing_final_funding_declaration_is_not_none(field):
    rows, kwargs = case()
    del kwargs["strict_record"][field]
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("allow_same_date_correction", True),
        ("post_flow_unit_count", Decimal("1")),
        ("external_flow_amount", Decimal("0.00000001")),
        ("external_flow_amount", True),
    ],
)
def test_no_second_flow_or_correction_mode(field, value):
    rows, kwargs = case(Decimal("100000"))
    kwargs[field] = value
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize(
    "field", ["cash_after", "market_value_after", "total_value_after", "portfolio_pnl_after"]
)
def test_source_accounting_must_match_registered_row(field):
    rows, kwargs = case()
    kwargs["official_close_source"]["accounting"][field] += 1
    with pytest.raises(StrategyRecordStoreError, match="source accounting"):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize("side", ["strict_record", "official_close_source"])
@pytest.mark.parametrize("fault", ["duplicate", "fractional", "nonfinite", "missing", "bool"])
def test_invalid_positions_reject(side, fault):
    rows, kwargs = case()
    positions = kwargs[side]["positions"]
    if fault == "duplicate":
        positions.append(deepcopy(positions[0]))
    elif fault == "missing":
        positions.clear()
    else:
        positions[0]["shares"] = {"fractional": "0.1", "nonfinite": "NaN", "bool": True}[fault]
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize("field", ["shares", "avg_cost", "cost_basis", "symbol"])
def test_even_subcent_position_identity_changes_reject(field):
    rows, kwargs = case()
    row = kwargs["strict_record"]["positions"][0]
    row[field] = (
        "000002.SZ" if field == "symbol" else Decimal(str(row[field])) + Decimal("0.00000001")
    )
    with pytest.raises(StrategyRecordStoreError):
        perf.extend_performance_rows(rows, **kwargs)


def test_even_subcent_cash_change_rejects():
    rows, kwargs = case()
    kwargs["strict_record"]["accounting"]["cash_after"] += Decimal("0.00000001")
    with pytest.raises(StrategyRecordStoreError, match="cash changed"):
        perf.extend_performance_rows(rows, **kwargs)


@pytest.mark.parametrize("kind", ["REGISTERED_CORRECTION", "REGISTERED_OFFICIAL_FINANCIAL_STATE"])
def test_registered_official_or_correction_row_cannot_be_finalized_again(kind):
    rows, kwargs = case()
    rows[-1]["evidence_kind"] = kind
    with pytest.raises(StrategyRecordStoreError, match="intraday source"):
        perf.extend_performance_rows(rows, **kwargs)


def test_finalization_is_pure(monkeypatch):
    rows, kwargs = case()

    def forbidden(*args, **kwargs):
        raise AssertionError("finalization attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert perf.extend_performance_rows(rows, **kwargs)[-1]["record_id"] == "20260104_200001-b01"


def test_existing_explicit_correction_remains_a_separate_mode():
    rows, kwargs = case()
    kwargs.pop("official_close_source")
    result = perf.extend_performance_rows(rows, **kwargs, allow_same_date_correction=True)
    assert result[-1]["evidence_kind"] == "REGISTERED_CORRECTION"
    assert result[:-1] == rows[:-1]
    assert len(result) == len(rows)


def test_finalization_cannot_replace_the_only_baseline():
    rows, kwargs = case()
    single = deepcopy(rows[-1])
    single["sequence_no"] = 1
    for field in ("interval_return", "cumulative_return", "drawdown"):
        single[field] = Decimal("0")
    perf.validate_performance_rows([single])
    with pytest.raises(StrategyRecordStoreError, match="mode or parent"):
        perf.extend_performance_rows([single], **kwargs)


def test_result_identity_and_metadata_cannot_override_target_shas():
    rows, kwargs = case()
    kwargs["ledger_parquet_sha256"] = "a" * 64
    with pytest.raises(StrategyRecordStoreError, match="target SHA"):
        perf.extend_performance_rows(rows, **kwargs)
