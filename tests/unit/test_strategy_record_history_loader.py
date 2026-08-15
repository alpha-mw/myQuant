from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.strategy_records import history as history_module
from quant_investor.strategy_records.store import bootstrap_catalog


def test_registered_history_reads_real_strategy_root_catalog(tmp_path: Path) -> None:
    records_root = tmp_path / "strategy_records"
    strategy_root = records_root / "CN" / "aggressive_tech_manufacturing"
    run_dir = strategy_root / "20260809_0933"
    run_dir.mkdir(parents=True)
    bootstrap_catalog(
        strategy_root,
        records=[
            {
                "record_id": "20260809_0933",
                "relative_path": "20260809_0933",
                "state": "ONLINE",
                "evidence_status": "HASH_VERIFIED",
            },
            {
                "record_id": "20260808_0933",
                "relative_path": "archives/2026/08/20260808_0933",
                "state": "ARCHIVED",
                "evidence_status": "ARCHIVE_HASH_VERIFIED",
            },
        ],
        active_record_id="20260809_0933",
        generation_id="history-loader-test",
        published_at="2026-08-10T00:00:00Z",
    )

    runs = history_module.HistoryLoader(records_root).load_recent(
        market="CN",
        strategy="aggressive_tech_manufacturing",
    )

    assert [item["record_id"] for item in runs] == [
        "20260809_0933",
        "20260808_0933",
    ]
    assert runs[0]["record_dir"] == str(run_dir)
    assert runs[0]["storage_state"] == "ONLINE"
    assert runs[0]["evidence_status"] == "HASH_VERIFIED"
    assert runs[1]["record_dir"] == ""
    assert runs[1]["storage_state"] == "ARCHIVED"
    assert runs[1]["evidence_status"] == "ARCHIVE_HASH_VERIFIED"


def test_registered_history_uses_catalog_and_exposes_storage_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / "strategy_records"
    active_dir = store_root / "CN" / "aggressive_tech_manufacturing" / "20260808_0933"
    active_dir.mkdir(parents=True)
    (active_dir / "analysis_report.md").write_text("# registered report\n", encoding="utf-8")
    monkeypatch.setattr(
        history_module,
        "load_registered_catalog",
        lambda root: ({"catalog_path": "catalog.json"}, {"records": []}),
    )
    monkeypatch.setattr(
        history_module,
        "catalog_history_entries",
        lambda root: [
            {
                "market": "CN",
                "strategy": "aggressive_tech_manufacturing",
                "record_id": "20260808_0933",
                "date": "20260808",
                "timestamp": "20260808_093300",
                "record_dir": str(active_dir),
                "storage_state": "ONLINE",
                "evidence_status": "HASH_VERIFIED",
                "summary": {
                    "symbols": ["002463.SZ"],
                    "actions": ["hold"],
                    "latest_report_excerpt": "registered summary",
                },
            }
        ],
    )
    loader = history_module.HistoryLoader(store_root)
    assert not hasattr(loader, "_legacy_run_dirs")

    runs = loader.load_recent(
        market="CN",
        strategy="aggressive_tech_manufacturing",
        max_dates=5,
    )

    assert runs[0]["record_id"] == "20260808_0933"
    assert runs[0]["storage_state"] == "ONLINE"
    assert runs[0]["evidence_status"] == "HASH_VERIFIED"
    assert runs[0]["symbols"] == ["002463.SZ"]
    context = loader.build_recall_context(runs, market="CN")
    assert context["source"] == "strategy_record_catalog"
    assert context["records"][0]["record_id"] == "20260808_0933"
    assert context["records"][0]["storage_state"] == "ONLINE"
    rendered = loader.format_context_section(runs)
    assert "storage=`ONLINE`" in rendered
    assert "evidence=`HASH_VERIFIED`" in rendered
    assert (
        loader.load_last_report(
            market="CN",
            strategy="aggressive_tech_manufacturing",
        )
        == "# registered report\n"
    )


def test_registered_history_missing_catalog_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(history_module, "load_registered_catalog", lambda root: None)
    loader = history_module.HistoryLoader(tmp_path / "strategy_records")

    with pytest.raises(history_module.RecordStoreError, match="catalog missing"):
        loader.load_recent(
            market="CN",
            strategy="aggressive_tech_manufacturing",
        )


def test_unregistered_strategy_never_uses_direct_directory_read(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "strategy_records"
    run_dir = store_root / "US" / "simulated_portfolio_10000" / "20260402_2214"
    run_dir.mkdir(parents=True)
    (run_dir / "analysis_report.md").write_text(
        "# Legacy report\n- symbol: RKLB\n- action: watch\n",
        encoding="utf-8",
    )
    loader = history_module.HistoryLoader(store_root)

    with pytest.raises(history_module.RecordStoreError, match="catalog missing"):
        loader.load_recent(market="US", strategy="simulated_portfolio_10000")
