"""Native frozen risk segment; full EOD/Decision admission remains an explicit seam."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from _native_morning_threshold_fixture import build
from _native_corporate_fixture import put
from test_unified_morning_strategy import _quote_capture
from quant_investor.intelligence.morning import validate_sina_quote_capture
from quant_investor.intelligence.morning_threshold_review import (
    build_threshold_review,
    validate_threshold_review,
    review_hash,
)
from quant_investor.operations.morning_risk_sources import MorningRiskSources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.strategy_records.risk_policy_contract import validate_initial_stop_policy


def fixture(root, **options):
    f = build(root, **options)
    recorded = {
        "native_inputs_ref": put(root, "fixtures/morning/native-inputs.json", f["native_inputs"]),
        "node_terminal_refs": f["terminal_refs"],
        "synthetic": True,
        "native_validation_completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ref = put(root, "results/operations/daily_production/CN/20260827/completion.v1.json", recorded)

    def read():
        return MorningRiskSources(
            workspace=root,
            recorded=recorded,
            completion_ref=ref,
            policy_refs=f["policy_refs"],
            quote_requested_at="2026-08-28T01:45:00Z",
        )

    path, sha = _quote_capture(
        root,
        run_date="20260828",
        request_time="2026-08-28T01:45:00Z",
        response_time="2026-08-28T01:45:01Z",
    )
    quote = json.loads((root / path).read_bytes())
    from quant_investor.intelligence.sina_quotes import parse_sina_quote_response

    quote["quote_rows"] = parse_sina_quote_response(
        (root / quote["raw_ref"]["path"]).read_bytes(), quote["symbol_mapping"]
    )
    sha = put(root, path, quote)["sha256"]
    validate_sina_quote_capture(
        quote, raw=(root / quote["raw_ref"]["path"]).read_bytes(), run_date="20260828"
    )
    args = {
        "quote": quote,
        "quote_capture_ref": {"path": path, "sha256": sha},
        "quote_raw_ref": {k: quote["raw_ref"][k] for k in ("path", "sha256")},
        "decision_result_ref": put(
            root, "fixtures/morning/decision-seam.json", {"synthetic_decision_admission_seam": True}
        ),
        "evidence_mode": "REPLAY_ONLY",
    }
    return f, read, args


def inventory(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_native_sources_and_review_remain_frozen_when_heads_disappear(tmp_path):
    f, read, args = fixture(tmp_path)
    sources = read()
    before = inventory(tmp_path)
    review = build_threshold_review(sources=sources, **args)
    assert review["summary_state"] == "COMPLETE_RESEARCH_REVIEW"
    assert review["rows"][0]["eod_risk"]["peak_price"] == "25.0"
    assert review["rows"][0]["eod_risk"]["moving_take_profit_review_price"] == "22.00"
    assert inventory(tmp_path) == before
    for path in (
        f["book"].root / "_record_store/current.v1.json",
        f["book"].root / "_event_store/current.v1.json",
        tmp_path / "data/parquet/cn/_latest.json",
    ):
        path.unlink()
    before = inventory(tmp_path)
    assert build_threshold_review(sources=read(), **args) == review
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "price,expected",
    [
        ("9.00", "WARNING_NOT_BREACH"),
        ("8.99", "WARNING_NOT_BREACH"),
        ("9.01", "CLEAR"),
        ("0", "QUOTE_UNAVAILABLE"),
        ("-1", "QUOTE_UNAVAILABLE"),
    ],
)
def test_intraday_quote_cannot_change_eod_peak_or_confirm_stop(tmp_path, price, expected):
    _, read, args = fixture(tmp_path)
    # Projection boundary: caller's native quote admission is separately exercised above.
    args["quote"]["quote_rows"][0]["price"] = price
    row = build_threshold_review(sources=read(), **args)["rows"][0]
    assert row["quote_observation"]["initial_stop_comparison"] == expected
    assert row["eod_risk"]["owner_stop_trigger"] == "CLEAR"
    assert row["eod_risk"]["peak_price"] == "25.0"
    assert row["actions"] == [] and row["executable"] is False


def test_missing_trailing_anchor_does_not_erase_independent_stop(tmp_path):
    _, read, args = fixture(tmp_path)
    sources = read()
    sources.rows[0]["anchor"] = None
    row = build_threshold_review(sources=sources, **args)["rows"][0]
    assert row["policy_binding"]["trailing"]["state"] == "NOT_CONFIGURED"
    assert row["policy_binding"]["initial_stop"]["state"] == "BOUND"
    assert row["eod_risk"]["owner_stop_price"] == "9.00"


@pytest.mark.parametrize("adjustment_change", [False, True])
def test_legacy_fixed_stop_uses_no_retired_trailing_or_quote_metadata(tmp_path, adjustment_change):
    from test_initial_stop_policy_compatibility import legacy_row

    f, read, args = fixture(tmp_path, adjustment_change=adjustment_change)
    policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
    policy["stops"][0] = legacy_row(policy["stops"][0])
    policy["stops"][0]["initial_stop_price_cny"] = "35.32"
    policy["stops"][0]["trigger"]["price_cny"] = "35.32"
    f["policy_refs"]["initial_stop"] = put(tmp_path, "fixtures/morning/legacy-stop.json", policy)
    sources = read()
    # Native source/window validation above; absence of a trailing anchor is an
    # explicit projection input seam, as in the existing independent-stop test.
    sources.rows[0]["anchor"] = None
    row = build_threshold_review(sources=sources, **args)["rows"][0]
    assert row["policy_binding"]["trailing"]["state"] == "NOT_CONFIGURED"
    assert row["policy_binding"]["initial_stop"]["configured_price"] == "35.32"
    risk = row["eod_risk"]
    for field in (
        "moving_take_profit_review_price",
        "moving_take_profit_reduce_price",
        "moving_stop_price",
        "peak_price",
        "peak_date",
        "tracking_start_date",
    ):
        assert risk[field] is None
    if adjustment_change:
        assert row["policy_binding"]["initial_stop"]["state"] == "UNCONFIRMED"
        assert "OWNER_STOP_CORPORATE_ACTION_REVIEW" in risk["owner_stop_blockers"]
        assert row["quote_observation"]["initial_stop_comparison"] == "UNCONFIRMED"
    else:
        assert row["policy_binding"]["initial_stop"]["state"] == "BOUND"
        assert risk["owner_stop_price"] == "35.32"
        assert risk["owner_stop_trigger"] == "BREACH"
        assert row["quote_observation"]["price"] == "49.50"
        assert row["quote_observation"]["initial_stop_comparison"] == "CLEAR"
    assert row["executable"] is False and row["actions"] == []
    policy["stops"][0]["retired_unexecutable_trailing_stop_cny"] = "999.00"
    policy["stops"][0]["sina_price_cny_at_20260828_143135"] = "1.00"
    f["policy_refs"]["initial_stop"] = put(
        tmp_path, "fixtures/morning/changed-metadata.json", policy
    )
    changed = read()
    changed.rows[0]["anchor"] = None
    again = build_threshold_review(sources=changed, **args)["rows"][0]
    assert again["eod_risk"] == risk
    assert again["quote_observation"] == row["quote_observation"]


def test_new_trailing_policy_cannot_reset_prior_eod_reconciliation(tmp_path):
    f, read, args = fixture(tmp_path)
    policy = json.loads((tmp_path / f["policy_refs"]["trailing"]["path"]).read_bytes())
    policy["policy_id"] = "synthetic-owner-revision"
    f["policy_refs"]["trailing"] = put(tmp_path, "fixtures/morning/revised-trailing.json", policy)
    review = build_threshold_review(sources=read(), **args)
    row = review["rows"][0]
    assert row["policy_binding"]["trailing"]["state"] == "NOT_RECONCILED_AT_PRIOR_EOD"
    assert row["quote_observation"]["trailing_review_comparison"] == "UNCONFIRMED"
    assert row["policy_binding"]["initial_stop"]["state"] == "BOUND"
    assert review["summary_state"] == "PARTIAL_RESEARCH_REVIEW"


def test_new_morning_stop_is_disclosed_without_backdated_breach(tmp_path):
    f, read, args = fixture(tmp_path)
    policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
    policy["effective_from"] = policy["owner_confirmation_recorded_at"] = "2026-08-28T01:00:00Z"
    f["policy_refs"]["initial_stop"] = put(tmp_path, "fixtures/morning/new-stop.json", policy)
    row = build_threshold_review(sources=read(), **args)["rows"][0]
    assert row["policy_binding"]["initial_stop"]["configured_price"] == "9.00"
    assert row["policy_binding"]["initial_stop"]["state"] == "UNCONFIRMED"
    assert row["eod_risk"]["owner_stop_trigger"] != "BREACH"
    assert row["quote_observation"]["initial_stop_comparison"] == "UNCONFIRMED"


def test_later_policy_confirmation_cannot_govern_original_quote(tmp_path):
    f, _, _ = fixture(tmp_path)
    policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
    policy["owner_confirmation_recorded_at"] = "2026-08-28T06:30:00Z"
    with pytest.raises(ContractError, match="NOT_AVAILABLE_AT_QUOTE"):
        validate_initial_stop_policy(policy, as_of="2026-08-28T01:45:00Z")


@pytest.mark.parametrize("fault", ["authority", "comparison", "summary", "source", "quote_symbol"])
def test_even_rehashed_semantically_changed_review_is_rejected(tmp_path, fault):
    _, read, args = fixture(tmp_path)
    review = deepcopy(build_threshold_review(sources=read(), **args))
    if fault == "authority":
        review["authority"]["trade"] = True
    elif fault == "comparison":
        review["rows"][0]["quote_observation"]["initial_stop_comparison"] = "WARNING_NOT_BREACH"
    elif fault == "summary":
        review["summary_state"] = "PARTIAL_RESEARCH_REVIEW"
    elif fault == "source":
        review["source_refs"].remove(review["quote_raw_ref"])
    else:
        review["quote_only_symbols"] = ["fake"]
    review["content_sha256"] = review_hash(review)
    with pytest.raises(ContractError):
        validate_threshold_review(review)


def test_confirmed_eod_stop_breach_survives_recovered_morning_quote(tmp_path):
    f, read, args = fixture(tmp_path)
    policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
    policy["stops"][0]["initial_stop_price_cny"] = "26.00"
    policy["stops"][0]["trigger"]["price_cny"] = "26.00"
    f["policy_refs"]["initial_stop"] = put(tmp_path, "fixtures/morning/breached-stop.json", policy)
    row = build_threshold_review(sources=read(), **args)["rows"][0]
    assert row["eod_risk"]["owner_stop_trigger"] == "BREACH"
    assert row["eod_risk"]["owner_review_state"] == "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW"
    assert row["quote_observation"]["initial_stop_comparison"] == "CLEAR"
    assert row["executable"] is False and row["actions"] == []


def test_prior_window_corporate_change_blocks_both_threshold_branches(tmp_path):
    _, read, args = fixture(tmp_path, adjustment_change=True)
    review = build_threshold_review(sources=read(), **args)
    row = review["rows"][0]
    assert "NAMED_EVENT_EVIDENCE_MISSING" in row["eod_risk"]["trailing_blockers"]
    assert "OWNER_STOP_CORPORATE_ACTION_REVIEW" in row["eod_risk"]["owner_stop_blockers"]
    assert row["quote_observation"]["initial_stop_comparison"] == "UNCONFIRMED"
    assert row["quote_observation"]["trailing_review_comparison"] == "UNCONFIRMED"
    assert review["summary_state"] == "PARTIAL_RESEARCH_REVIEW"


@pytest.mark.parametrize("fault", ["policy_baseline", "profile", "frame_bytes"])
def test_native_reader_rejects_inconsistent_bound_sources(tmp_path, fault):
    f, read, _ = fixture(tmp_path)
    if fault == "policy_baseline":
        policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
        policy["store_binding"]["ledger_sha256"] = "f" * 64
        f["policy_refs"]["initial_stop"] = put(
            tmp_path, "fixtures/morning/bad-baseline.json", policy
        )
    elif fault == "profile":
        sources = read()
        sources.recorded["native_inputs_ref"] = put(
            tmp_path,
            "fixtures/morning/mixed-inputs.json",
            {**f["native_inputs"], "previous_trade_date": "20260825"},
        )
    else:
        (tmp_path / f["native_inputs"]["adjustment_market_refs"]["002463.SZ"]["path"]).write_bytes(
            b"corrupted native frame"
        )
    before = inventory(tmp_path)
    with pytest.raises((ContractError, ValueError, RuntimeError)):
        read()
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "effective,confirmed,clock,expected",
    [
        ("2026-08-27T06:00:00Z", "2026-08-27T06:00:00Z", True, None),
        ("2026-08-27T07:00:00Z", "2026-08-27T07:00:00Z", True, None),
        (
            "2026-08-27T07:00:01Z",
            "2026-08-27T07:00:01Z",
            True,
            "OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD",
        ),
        ("2026-08-27T06:00:00Z", "2026-08-27T06:00:00Z", False, "OWNER_STOP_EVIDENCE_UNCONFIRMED"),
        (
            "2026-08-27T06:00:00Z",
            "2026-08-28T01:00:00Z",
            True,
            "OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD",
        ),
    ],
)
def test_stop_activation_requires_both_owner_clocks_and_native_close_clock(
    effective, confirmed, clock, expected
):
    from types import SimpleNamespace

    source = SimpleNamespace(
        day="20260827",
        policies={
            "initial_stop": {
                "effective_from": effective,
                "owner_confirmation_recorded_at": confirmed,
            }
        },
        calendar={"timezone": "Asia/Shanghai", "session_close_local": "15:00:00"} if clock else {},
    )
    assert MorningRiskSources._stop_availability(source) == expected


def test_unconfigured_initial_stop_remains_partial_not_clear(tmp_path):
    f, read, args = fixture(tmp_path)
    policy = json.loads((tmp_path / f["policy_refs"]["initial_stop"]["path"]).read_bytes())
    policy["stops"] = []
    f["policy_refs"]["initial_stop"] = put(
        tmp_path, "fixtures/morning/no-configured-stops.json", policy
    )
    review = build_threshold_review(sources=read(), **args)
    assert review["summary_state"] == "PARTIAL_RESEARCH_REVIEW"
    assert review["rows"][0]["policy_binding"]["initial_stop"]["state"] == "NOT_CONFIGURED"
    assert review["rows"][0]["quote_observation"]["initial_stop_comparison"] == "NOT_CONFIGURED"


def test_calculated_threshold_cannot_become_null_by_rehashing(tmp_path):
    from quant_investor.strategy_records.research_risk import seal

    _, read, args = fixture(tmp_path)
    review = build_threshold_review(sources=read(), **args)
    risk = review["rows"][0]["eod_risk"]
    risk["moving_take_profit_review_price"] = None
    review["rows"][0]["eod_risk"] = seal(risk)
    review["content_sha256"] = review_hash(review)
    with pytest.raises(ContractError, match="CALCULATED_LEVEL_MISSING"):
        validate_threshold_review(review)
