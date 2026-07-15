from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market import macro_mart
from tests.helpers.macro_fixture import bind_macro_generation


def _row() -> dict[str, object]:
    return {
        "trade_date": "2024-05-10",
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2024-05-10T08:00:00+00:00",
    }


@pytest.mark.parametrize("failure", ["hash_drift", "unreadable"])
def test_switched_recovery_rolls_back_when_required_table_is_invalid(
    tmp_path: Path,
    failure: str,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    macro_root = market_root / "macro_daily"
    catalog_path, _, _, _ = bind_macro_generation(
        macro_root,
        generation_id="required-closure",
        row=_row(),
    )
    daily_root = market_root / "daily_basic"
    daily_root.mkdir()
    daily_path = daily_root / "part.parquet"
    pd.DataFrame([{"trade_date": "20240510"}]).to_parquet(
        daily_path,
        index=False,
    )
    if failure == "unreadable":
        daily_path.write_bytes(b"not parquet")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["required_tables"] = ["daily_basic", "macro_daily"]
    catalog["tables"]["daily_basic"] = {
        "logical_table": "daily_basic",
        "path": "daily_basic/part.parquet",
        "table_root": "daily_basic",
        "sha256": hashlib.sha256(daily_path.read_bytes()).hexdigest(),
    }
    new_bytes = macro_mart._catalog_bytes(catalog)
    old_bytes = b'{"catalog":"exact-old"}\n'
    pointer_path = market_root / "_latest.json"
    pointer_bytes = b'{"snapshot_id":"stable"}\n'
    pointer_path.write_bytes(pointer_bytes)
    journal_path, _ = macro_mart._prepare_catalog_transaction(
        root=macro_root,
        run_id=f"required-{failure}",
        old_catalog_bytes=old_bytes,
        new_catalog_bytes=new_bytes,
        generation_id="required-closure",
        expected_market_pointer_sha256=hashlib.sha256(
            pointer_bytes
        ).hexdigest(),
    )
    macro_mart._atomic_write_bytes(catalog_path, new_bytes)
    if failure == "hash_drift":
        daily_path.write_bytes(daily_path.read_bytes() + b"drift")

    macro_mart._recover_catalog_transactions(
        root=macro_root,
        catalog_path=catalog_path,
    )

    assert catalog_path.read_bytes() == old_bytes
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["state"] == "rolled_back"
    expected_detail = (
        "macro_catalog_required_hash_mismatch:daily_basic"
        if failure == "hash_drift"
        else "macro_catalog_required_table_unreadable:daily_basic"
    )
    assert journal["detail"] == expected_detail
