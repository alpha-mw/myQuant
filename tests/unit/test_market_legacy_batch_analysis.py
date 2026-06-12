"""Legacy market batch-analysis compatibility boundary tests."""

from __future__ import annotations

import json
from pathlib import Path


def test_legacy_batch_module_saves_batch_result(tmp_path):
    from quant_investor.market.legacy_batch_analysis import save_batch_result

    output = save_batch_result(
        {
            "category": "hs300",
            "batch_id": 1,
            "timestamp": "20260612_122200",
            "stocks": ["000001.SZ"],
            "stock_count": 1,
        },
        market="CN",
        output_dir=str(tmp_path),
    )

    payload = json.loads(Path(output).read_text(encoding="utf-8"))

    assert Path(output).name == "batch_hs300_001_20260612_122200.json"
    assert payload["stocks"] == ["000001.SZ"]


def test_analyze_category_wrapper_preserves_monkeypatch_compatibility(
    monkeypatch,
    tmp_path,
):
    import quant_investor.market.analyze as analyze

    batch_calls: list[tuple[list[str], int]] = []
    saved_batches: list[int] = []

    monkeypatch.setattr(
        analyze,
        "get_all_local_symbols",
        lambda category, market="CN", data_dir=None: [
            "000001.SZ",
            "000002.SZ",
        ],
    )

    def _fake_analyze_batch(
        symbols,
        category,
        batch_id,
        **kwargs,
    ):
        batch_calls.append((list(symbols), int(batch_id)))
        return {
            "category": category,
            "batch_id": batch_id,
            "timestamp": f"20260612_12220{batch_id}",
            "stocks": list(symbols),
            "stock_count": len(symbols),
        }

    def _fake_save_batch_result(result, **kwargs):
        saved_batches.append(int(result["batch_id"]))
        return str(tmp_path / f"batch_{result['batch_id']}.json")

    monkeypatch.setattr(analyze, "analyze_batch", _fake_analyze_batch)
    monkeypatch.setattr(analyze, "save_batch_result", _fake_save_batch_result)

    results = analyze.analyze_category_full(
        "hs300",
        market="CN",
        batch_size=1,
        output_dir=str(tmp_path),
    )

    assert [batch_id for _, batch_id in batch_calls] == [1, 2]
    assert saved_batches == [1, 2]
    assert [result["stocks"] for result in results] == [
        ["000001.SZ"],
        ["000002.SZ"],
    ]
