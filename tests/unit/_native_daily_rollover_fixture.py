"""Next-day synthetic Market and real native history audit for rollover tests."""

import json
from pathlib import Path
import sys
import pandas as pd
from quant_investor.market.cn_history_audit import run_cn_history_audit


def prepare_market(root: Path, day: str | None = None) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs

    workspace = root / "factor-workspace"
    from datetime import date

    suffix = "" if day is None else "-" + day
    target = date.fromisoformat(day or "2026-08-25")
    fixture = NativeFactorInputs(workspace / "synthetic-successor-inputs", extra_future_sessions=3)
    offset = fixture.sessions[90:].index(target)
    inputs = fixture.day(offset, extra_history=9)
    strict_market_from_factor_inputs(workspace, inputs, snapshot_suffix="-complete-bars")
    dates = pd.read_parquet(inputs["exchange_calendar_path"])["open_session"].tolist()
    result, path = run_cn_history_audit(
        data_root=workspace / "data",
        output_root=workspace / "data/private/synthetic-history",
        days=100,
        end_date=inputs["as_of"],
        allow_online=False,
        trade_dates=[day.strftime("%Y%m%d") for day in dates[-100:]],
    )
    value = {
        "synthetic": True,
        "target": inputs["as_of"],
        "history": result,
        "audit_path": str(path),
    }
    (root / ("successor-market-audit" + suffix + ".json")).write_text(
        json.dumps(value, indent=2, default=str) + "\n"
    )
    return value


if __name__ == "__main__":
    print(json.dumps(prepare_market(Path(sys.argv[1])), indent=2, default=str))
