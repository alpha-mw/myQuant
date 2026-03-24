from __future__ import annotations

import pandas as pd

from quant_investor.monitoring.cn_aggressive_portfolio_tracker import (
    _map_symbol_to_quote_code,
    _parse_quote_payload,
    _score_full_market_metrics,
)


def test_map_symbol_to_quote_code():
    assert _map_symbol_to_quote_code("688519.SH") == "sh688519"
    assert _map_symbol_to_quote_code("002008.SZ") == "sz002008"
    assert _map_symbol_to_quote_code("sh000300") == "sh000300"


def test_parse_quote_payload_extracts_core_fields():
    payload = (
        'v_sh688519="1~南亚新材~688519~142.24~133.02~137.20~5136164~2821647~2314317~'
        '142.25~4~142.23~12~142.22~2~142.21~1~142.20~22~142.26~5~142.95~41~143.00~28~'
        '143.02~65~143.07~2~~20260324130921~9.22~6.93~147.62~132.00~";'
    )
    parsed = _parse_quote_payload(payload)

    assert parsed is not None
    assert parsed["quote_code"] == "sh688519"
    assert parsed["name"] == "南亚新材"
    assert parsed["current"] == 142.24
    assert parsed["prev_close"] == 133.02
    assert parsed["change_pct"] == 6.93
    assert parsed["time"] == "20260324130921"


def test_score_full_market_metrics_is_deterministic():
    metrics = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "ret1": 0.01,
                "ret5": 0.03,
                "ret20": 0.08,
                "ret60": 0.12,
                "close_vs_ma20": 0.05,
                "ma20_vs_ma60": 0.04,
                "ma60_vs_ma120": 0.03,
                "dd20": -0.02,
            },
            {
                "symbol": "BBB",
                "ret1": 0.04,
                "ret5": 0.06,
                "ret20": 0.14,
                "ret60": 0.18,
                "close_vs_ma20": 0.08,
                "ma20_vs_ma60": 0.06,
                "ma60_vs_ma120": 0.05,
                "dd20": -0.01,
            },
            {
                "symbol": "CCC",
                "ret1": -0.02,
                "ret5": -0.01,
                "ret20": 0.02,
                "ret60": 0.04,
                "close_vs_ma20": -0.01,
                "ma20_vs_ma60": 0.01,
                "ma60_vs_ma120": 0.0,
                "dd20": -0.06,
            },
        ]
    )

    scored = _score_full_market_metrics(metrics)

    assert scored["symbol"].tolist() == ["BBB", "AAA", "CCC"]
    assert scored["rank_full_market"].tolist() == [1, 2, 3]
    assert scored["score_full_market"].is_monotonic_decreasing
