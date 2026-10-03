"""Advisory attribution: call extraction from the decision log and the forward-return join."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "research_advisory_attribution", SCRIPTS / "research_advisory_attribution.py"
)
attribution = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = attribution
_SPEC.loader.exec_module(attribution)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("buy", 1),
        ("add_risk", 1),
        ("buy 002463.SZ 1000 @130.00; buy 688183.SH 500 @120.00", 1),
        ("reduce_risk", -1),
        ("clear_risk", -1),
        ("local/manual sell 002851.SZ 200 @169.49", -1),
        ("hold", 0),
        ("Maxwell指示继续", None),
        (None, None),
    ],
)
def test_actions_classify_into_long_short_hold_or_unknown(action, expected) -> None:
    assert attribution.classify_action(action) == expected


def test_calls_are_symbol_level_directional_and_deduplicated() -> None:
    rows = [
        {
            "event_id": "a",
            "event_type": "advisory",
            "trade_date": "2026-07-12",
            "symbol": "002463.SZ",
            "action": "reduce_risk",
            "answer_source": "codex_thread",
        },
        {
            "event_id": "b",
            "event_type": "advisory",
            "trade_date": "2026-07-12",
            "symbol": "002463.SZ",
            "action": "reduce_risk",
            "answer_source": "codex_thread",
        },
        {
            "event_id": "c",
            "event_type": "human_action",
            "trade_date": "2026-07-08",
            "symbol": "002463.SZ,688183.SH",
            "action": "buy 002463.SZ 1000; buy 688183.SH 500",
            "channel": "user_reported_post_review",
        },
        {
            "event_id": "d",
            "event_type": "advisory",
            "trade_date": "2026-07-13",
            "symbol": "PORTFOLIO",
            "action": "hold",
            "answer_source": "codex_thread",
        },
        {
            "event_id": "e",
            "event_type": "pipeline_proposal",
            "trade_date": "2026-07-13",
            "symbol": "002463.SZ",
            "action": "buy",
        },
    ]

    calls, skipped = attribution.extract_calls(rows)

    assert [(c.trade_date, c.symbol, c.direction, c.event_type) for c in calls] == [
        ("20260712", "002463.SZ", -1, "advisory"),
        ("20260708", "002463.SZ", 1, "human_action"),
        ("20260708", "688183.SH", 1, "human_action"),
    ]
    assert calls[0].event_id == "a" and calls[0].source == "codex_thread"
    assert {row["event_id"]: row["reason"] for row in skipped} == {
        "d": "hold_or_watch",
        "e": "not_an_advice_or_action",
    }


def test_forward_returns_enter_at_the_next_open_and_subtract_the_universe() -> None:
    dates = ["20260701", "20260702", "20260703", "20260706", "20260707", "20260708", "20260709"]
    bars = pd.DataFrame(
        {
            "ts_code": "002463.SZ",
            "trade_date": dates,
            "open": [10.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            "adj_close": [10.0, 11.0, 13.2, 13.2, 14.4, 15.0, 15.6],
            "adj_factor": 1.0,
        }
    )
    benchmark = pd.DataFrame({"equal_weight": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    call = attribution.Call("x", "advisory", "codex_thread", "20260702", "002463.SZ", "buy", 1)

    result = attribution.forward_returns(bars, benchmark, call)

    # Advice on 07-02 is actionable at the 07-03 open of 12.0. Five sessions in
    # the market end at the 07-09 close of 15.6 (+30%); the universe made 10%.
    assert result["entry_date"] == "20260703"
    assert result["stock_5"] == pytest.approx(0.30)
    assert result["excess_5"] == pytest.approx(0.20)
    assert result["excess_20"] is None

    too_late = attribution.Call("y", "advisory", "codex_thread", "20260709", "002463.SZ", "buy", 1)
    assert attribution.forward_returns(bars, benchmark, too_late) == {"entry_date": None}
