from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.branch_readiness import (
    BranchDataReadiness,
    BranchGovernanceReport,
    STATUS_PASS,
)
from quant_investor.market.data_governance import run_data_governance
from quant_investor.market.macro_mart import write_macro_mart
from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore
from tests.helpers.macro_readiness_fixture import (
    make_blocked_macro_readiness_runtime,
    make_macro_readiness_runtime,
)


@pytest.fixture(autouse=True)
def _keep_release_calendar_fixture_off_canonical_data(monkeypatch):
    import quant_investor.market.data_governance as governance_module

    monkeypatch.setattr(
        governance_module,
        "freeze_macro_readiness_runtime",
        lambda **kwargs: make_blocked_macro_readiness_runtime(
            macro_logical_date=str(kwargs.get("macro_logical_date") or ""),
            target_session_date=str(kwargs.get("target_session_date") or ""),
        ),
    )


def _daily_frame(symbol: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol, symbol],
            "trade_date": ["20240509", "20240510"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.0],
            "close": [10.1, 10.4],
            "volume": [1000.0, 1200.0],
            "amount": [10_000.0, 12_500.0],
        }
    )


def _write_parquet_market_data(root):
    data_root = root / "data"
    parquet_root = data_root / "parquet" / "cn"
    bars_root = parquet_root / "bars"
    serving_root = data_root / "parquet_serving" / "cn" / "bars"
    manifest_path = parquet_root / "_snapshots" / "fixture.json"
    frame = _daily_frame()
    (bars_root / "year=2024").mkdir(parents=True)
    (serving_root / "symbol=000001.SZ").mkdir(parents=True)
    frame.to_parquet(bars_root / "year=2024" / "part.parquet", index=False)
    frame.to_parquet(serving_root / "symbol=000001.SZ" / "bars.parquet", index=False)
    manifest_path.parent.mkdir(parents=True)
    expected_symbols = ["000001.SZ"]
    expected_scope_sha256 = hashlib.sha256("\n".join(expected_symbols).encode("utf-8")).hexdigest()
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v3",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": 1,
        "expected_scope_count": 1,
        "observed_bar_count": 1,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "coverage_trade_date": "20240510",
        "expected_scope_sha256": expected_scope_sha256,
        "suspended_symbols": [],
        "inactive_symbols": [],
        "verified_nontrading_bak_daily_zero_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": [],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
    }
    manifest_path.write_text(
        json.dumps({"snapshot_id": "fixture", "coverage": coverage}),
        encoding="utf-8",
    )
    (parquet_root / "_latest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "snapshot_id": "fixture",
                "latest_complete_trade_date": "20240510",
                "latest_trade_date": "20240510",
                "table_root": str(bars_root),
                "derived_serving_root": str(serving_root),
                "manifest_path": str(manifest_path),
                "coverage": coverage,
            }
        ),
        encoding="utf-8",
    )
    universe_root = data_root / "cn_universe"
    universe_root.mkdir(parents=True)
    (universe_root / "cn_index_components.json").write_text(
        json.dumps({"full_a": expected_symbols}),
        encoding="utf-8",
    )
    return data_root


def _write_full_a_scope_fixture(root: Path) -> Path:
    data_root = root / "data"
    parquet_root = data_root / "parquet" / "cn"
    bars_root = parquet_root / "bars"
    serving_root = data_root / "parquet_serving" / "cn" / "bars"
    manifest_path = parquet_root / "_snapshots" / "scope-fixture.json"
    expected_symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    symbol_frames = {
        "000001.SZ": _daily_frame("000001.SZ"),
        "000002.SZ": _daily_frame("000002.SZ"),
        "000003.SZ": _daily_frame("000003.SZ").assign(trade_date=["20240508", "20240509"]),
        "000999.SZ": _daily_frame("000999.SZ").assign(trade_date=["20200102", "20200103"]),
    }
    (bars_root / "year=2024").mkdir(parents=True)
    for symbol, frame in symbol_frames.items():
        symbol_root = serving_root / f"symbol={symbol}"
        symbol_root.mkdir(parents=True)
        frame.to_parquet(symbol_root / "bars.parquet", index=False)
    pd.concat(symbol_frames.values(), ignore_index=True).to_parquet(
        bars_root / "year=2024" / "part.parquet",
        index=False,
    )
    expected_scope_sha256 = hashlib.sha256(
        "\n".join(sorted(expected_symbols)).encode("utf-8")
    ).hexdigest()
    pit_path = parquet_root / "reference" / "stock_basic_membership.parquet"
    pit_path.parent.mkdir(parents=True)
    pd.DataFrame({"ts_code": expected_symbols}).to_parquet(pit_path, index=False)
    pit_sha256 = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v3",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": 3,
        "expected_scope_count": 3,
        "observed_bar_count": 2,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "coverage_trade_date": "20240510",
        "expected_scope_sha256": expected_scope_sha256,
        "suspended_symbols": [],
        "inactive_symbols": ["000003.SZ"],
        "verified_nontrading_bak_daily_zero_symbols": [],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": ["000003.SZ"],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
        "suspended_evidence_symbols": [],
        "inactive_evidence_symbols": ["000003.SZ"],
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": pit_sha256,
    }
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"snapshot_id": "scope-fixture", "coverage": coverage}),
        encoding="utf-8",
    )
    (parquet_root / "_latest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "snapshot_id": "scope-fixture",
                "latest_complete_trade_date": "20240510",
                "latest_trade_date": "20240510",
                "table_root": str(bars_root),
                "derived_serving_root": str(serving_root),
                "manifest_path": str(manifest_path),
                "coverage": coverage,
            }
        ),
        encoding="utf-8",
    )
    universe_root = data_root / "cn_universe"
    universe_root.mkdir(parents=True)
    (universe_root / "cn_index_components.json").write_text(
        json.dumps({"full_a": expected_symbols}),
        encoding="utf-8",
    )
    return data_root


def _mutate_bound_coverage(data_root: Path, mutation) -> None:
    pointer_path = data_root / "parquet" / "cn" / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    coverage = dict(pointer["coverage"])
    mutation(coverage)
    pointer["coverage"] = coverage
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    manifest_path = Path(pointer["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = coverage
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_fundamental(root):
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "availability_date": "2024-04-30",
                "source": "tushare_fina_indicator;forecast",
                "source_priority": "tushare_primary",
                "fin_roe": 0.13,
                "fin_roa": 0.06,
                "fin_debt_to_assets": 0.42,
                "fin_net_profit_yoy": 0.18,
                "fin_ocf_to_profit": 1.12,
                "fin_fcf_to_profit": 0.88,
                "fcf_to_price": 0.04,
                "forecast_revision": 0.05,
            }
        ]
    ).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps(
            {
                "provider_status": "tushare_primary",
                "source_priority": "tushare_primary",
                "storage_backend": "parquet_canonical",
            }
        ),
        encoding="utf-8",
    )


def test_data_governance_default_is_local_read_only(tmp_path):
    data_root = _write_parquet_market_data(tmp_path)
    fundamental_root = tmp_path / "cn_fundamental"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental(fundamental_root)
    write_macro_mart(
        {
            "trade_date": "20240510",
            "macro_score": 0.2,
            "liquidity_score": 0.4,
            "volatility_percentile": 45.0,
            "policy_signal": "neutral",
            "source": "tushare_macro",
            "source_priority": "tushare_primary",
        },
        data_root=macro_root,
        raw_snapshot_root=tmp_path / "snapshots" / "macro",
    )

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=fundamental_root,
        macro_root=macro_root,
        output_dir=tmp_path / "reports",
    )

    assert result["local_read_only"] is True
    assert result["status"] == "blocked"
    assert result["allow_live"] is False
    assert result["reports"][0]["readiness"]["macro"]["status"] == "block"
    assert "macro_catalog_missing" in result["reports"][0]["readiness"]["macro"]["blockers"]
    assert (
        (tmp_path / "reports")
        .joinpath(result["artifacts"]["full_a"]["json"].split("/")[-1])
        .exists()
    )


def test_full_a_governance_uses_snapshot_scope_not_historical_serving(tmp_path):
    data_root = _write_full_a_scope_fixture(tmp_path)

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    report = result["reports"][0]
    quant = report["readiness"]["quant"]
    scope = report["metadata"]["quant_scope"]
    assert quant["status"] == "pass"
    assert quant["coverage_ratio"] == 1.0
    assert quant["affected_symbols"] == []
    assert set(quant["metadata"]["latest_dates"]) == {"000001.SZ", "000002.SZ"}
    assert scope["policy"] == "snapshot_bound_current_full_a"
    assert scope["status"] == "passed"
    assert scope["blockers"] == []
    assert scope["expected_scope_count"] == 3
    assert scope["observed_bar_count"] == 2
    assert scope["readiness_symbol_count"] == 2
    assert scope["non_blocking_absent_symbols"] == ["000003.SZ"]
    assert scope["serving_inventory_count"] == 4


@pytest.mark.parametrize(
    ("case", "expected_blocker"),
    [
        ("coverage_date", "coverage_trade_date_mismatch"),
        ("category", "coverage_full_a_category_missing"),
        ("complete", "coverage_not_complete"),
        ("schema", "coverage_schema_version_unsupported"),
        ("provenance", "coverage_provenance_invalid"),
        ("components_hash", "coverage_expected_scope_sha256_mismatch"),
        ("classification_union", "coverage_non_blocking_absent_union_mismatch"),
        ("status_evidence", "coverage_inactive_evidence_mismatch"),
        ("pit_hash", "coverage_pit_membership_sha256_mismatch"),
        ("observed_count", "coverage_observed_bar_count_mismatch"),
        ("serving_presence", "readiness_symbols_missing_serving"),
    ],
)
def test_full_a_governance_scope_contract_mismatch_fails_closed(
    tmp_path,
    case,
    expected_blocker,
):
    data_root = _write_full_a_scope_fixture(tmp_path)
    if case == "coverage_date":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("coverage_trade_date", "20240509"),
        )
    elif case == "category":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("categories_checked", ["hs300"]),
        )
    elif case == "complete":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("complete", False),
        )
    elif case == "schema":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("coverage_schema_version", "legacy-coverage.v1"),
        )
    elif case == "provenance":
        pointer_path = data_root / "parquet" / "cn" / "_latest.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = Path(pointer["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["historical_scope_hash_backfilled"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif case == "components_hash":
        components_path = data_root / "cn_universe" / "cn_index_components.json"
        components_path.write_text(
            json.dumps({"full_a": ["000001.SZ", "000002.SZ"]}),
            encoding="utf-8",
        )
    elif case == "classification_union":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("non_blocking_absent_symbols", []),
        )
    elif case == "status_evidence":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("inactive_evidence_symbols", []),
        )
    elif case == "pit_hash":
        pointer = json.loads(
            (data_root / "parquet" / "cn" / "_latest.json").read_text(encoding="utf-8")
        )
        Path(pointer["coverage"]["pit_membership_path"]).write_bytes(b"drift")
    elif case == "observed_count":
        _mutate_bound_coverage(
            data_root,
            lambda coverage: coverage.__setitem__("observed_bar_count", 1),
        )
    else:
        (
            data_root / "parquet_serving" / "cn" / "bars" / "symbol=000002.SZ" / "bars.parquet"
        ).unlink()

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    quant = result["reports"][0]["readiness"]["quant"]
    scope = result["reports"][0]["metadata"]["quant_scope"]
    assert result["status"] == "blocked"
    assert quant["status"] == "block"
    assert "full_a_governance_scope_invalid" in quant["blockers"]
    assert any(expected_blocker in blocker for blocker in quant["blockers"])
    assert scope["status"] == "blocked"
    assert any(expected_blocker in blocker for blocker in scope["blockers"])


def test_full_a_governance_accepts_v2_without_nonblocking_absence(tmp_path):
    data_root = _write_parquet_market_data(tmp_path)
    _mutate_bound_coverage(
        data_root,
        lambda coverage: coverage.__setitem__("coverage_schema_version", "cn-full-a-coverage.v2"),
    )

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    report = result["reports"][0]
    assert report["metadata"]["quant_scope"]["status"] == "passed"
    assert report["readiness"]["quant"]["status"] == "pass"


def test_full_a_governance_accepts_coverage_bound_v4_pit_generation(
    tmp_path,
):
    data_root = _write_parquet_market_data(tmp_path)
    pointer_path = data_root / "parquet" / "cn" / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    immutable_root = (
        data_root
        / "parquet"
        / "cn"
        / "_snapshots"
        / str(pointer["snapshot_id"])
    )
    immutable_table = immutable_root / "table" / "bars"
    immutable_serving = immutable_root / "serving" / "bars"
    shutil.copytree(Path(pointer["table_root"]), immutable_table)
    shutil.copytree(Path(pointer["derived_serving_root"]), immutable_serving)
    pointer["table_root"] = str(immutable_table)
    pointer["derived_serving_root"] = str(immutable_serving)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    pit_store = PITUniverseStore(
        root_dir=data_root / "parquet" / "cn" / "reference",
        raw_root=data_root / "cn_universe" / "raw",
        compatibility_path=(
            data_root
            / "cn_universe"
            / "stock_basic_membership_latest.json"
        ),
    )
    generation = pit_store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                name="One",
                list_date="20200101",
                source_list_status="L",
                observed_at="2024-05-10T00:00:00Z",
                source_run_id="governance-v4",
            )
        ],
        observed_at="2024-05-10T00:00:00Z",
        source_run_id="governance-v4",
    )

    def _upgrade(coverage):
        coverage.update(
            {
                "coverage_schema_version": "cn-full-a-coverage.v4",
                "pit_membership_path": generation["canonical_path"],
                "pit_membership_sha256": generation["canonical_sha256"],
                "pit_generation_id": generation["generation_id"],
                "pit_generation_manifest_path": generation[
                    "generation_manifest_path"
                ],
                "pit_generation_manifest_sha256": generation[
                    "generation_manifest_sha256"
                ],
            }
        )

    _mutate_bound_coverage(data_root, _upgrade)
    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    scope = result["reports"][0]["metadata"]["quant_scope"]
    assert scope["status"] == "passed"
    assert scope["pit_generation_id"] == generation["generation_id"]


def test_non_full_a_unhealthy_snapshot_returns_structured_block(tmp_path):
    data_root = _write_parquet_market_data(tmp_path)
    _mutate_bound_coverage(
        data_root,
        lambda coverage: coverage.__setitem__("coverage_trade_date", "20240509"),
    )

    result = run_data_governance(
        market="CN",
        categories=["hs300"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    report = result["reports"][0]
    scope = report["metadata"]["quant_scope"]
    quant = report["readiness"]["quant"]
    assert result["status"] == "blocked"
    assert scope["status"] == "blocked"
    assert "market_category_scope_unavailable" in scope["blockers"][0]
    assert "category_governance_scope_invalid" in quant["blockers"]


def test_data_governance_allow_live_uses_explicit_maintenance_path(tmp_path, monkeypatch):
    data_root = _write_parquet_market_data(tmp_path)
    calls = {"fundamental": 0, "macro": 0}

    def _fake_fundamental(**kwargs):
        calls["fundamental"] += 1
        assert kwargs["allow_live"] is True
        assert kwargs["universes"] == "full_a"
        return {}

    def _fake_macro(**kwargs):
        calls["macro"] += 1
        assert kwargs["allow_live"] is True
        return {}

    monkeypatch.setattr(
        "quant_investor.market.fundamental_mart.run_cn_fundamental_maintenance", _fake_fundamental
    )
    monkeypatch.setattr("quant_investor.market.macro_mart.run_cn_macro_maintenance", _fake_macro)

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        allow_live=True,
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    assert calls == {"fundamental": 1, "macro": 1}
    assert result["local_read_only"] is False


def test_data_governance_rejects_retired_root_argument(tmp_path):
    with pytest.raises(TypeError):
        run_data_governance(
            intelligence_root=tmp_path / "retired",  # type: ignore[call-arg]
        )


def test_data_governance_deduplicates_categories(tmp_path, monkeypatch):
    data_root = _write_parquet_market_data(tmp_path)
    calls: list[str] = []

    def _read_once(*, market, category, as_of, data_dir=None):
        calls.append(category)
        return (
            {},
            {},
            type("Reader", (), {"snapshot": lambda self: {}})(),
            {},
            "",
        )

    monkeypatch.setattr(
        "quant_investor.market.data_governance._read_local_frames",
        _read_once,
    )

    result = run_data_governance(
        market="CN",
        categories=["full_a", "full_a"],
        category="full_a",
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    assert calls == ["full_a"]
    assert result["categories"] == ["full_a"]


def test_data_governance_freezes_and_reuses_one_macro_readiness_evidence(
    tmp_path,
    monkeypatch,
):
    import quant_investor.market.data_governance as governance_module

    runtime = make_macro_readiness_runtime(
        macro_logical_date="2024-05-10",
        target_session_date="2024-05-10",
    )
    calls = {"load": 0, "freeze": 0, "assess": 0}
    observed_evidence = []

    def _read_frames(*, market, category, as_of, data_dir=None):
        return {}, {}, object(), {"status": "passed"}, "20240510"

    def _load_macro(**kwargs):
        calls["load"] += 1
        return {"trade_date": "20240510"}, {"generation_id": "fixture"}

    def _freeze(**kwargs):
        calls["freeze"] += 1
        assert kwargs["macro_logical_date"] == "20240510"
        assert kwargs["target_session_date"] == "20240510"
        assert kwargs["calendar_root"] == tmp_path / "calendar"
        return runtime

    def _assess(**kwargs):
        calls["assess"] += 1
        observed_evidence.append(kwargs["pinned_macro_readiness_evidence"])
        assert kwargs["decision_cutoff_at"] == runtime.decision_cutoff_at
        return BranchGovernanceReport(
            run_id=str(kwargs["run_id"]),
            market="CN",
            category=str(kwargs["category"]),
            as_of="2024-05-10",
            readiness={
                branch: BranchDataReadiness(
                    branch=branch,
                    status=STATUS_PASS,
                    coverage_ratio=1.0,
                )
                for branch in ("quant", "fundamental", "macro")
            },
            blocked_symbols=[],
            quantifiable_universe=[],
            investable_universe=[],
            branch_data={},
        )

    monkeypatch.setattr(governance_module, "_read_local_frames", _read_frames)
    monkeypatch.setattr(governance_module, "load_macro_record", _load_macro)
    monkeypatch.setattr(
        governance_module,
        "freeze_macro_readiness_runtime",
        _freeze,
    )
    monkeypatch.setattr(
        governance_module,
        "assess_branch_data_readiness",
        _assess,
    )
    monkeypatch.setattr(
        governance_module,
        "write_branch_readiness_report",
        lambda report, output_dir: {
            "json": f"{report.category}.json",
            "md": f"{report.category}.md",
            "csv": f"{report.category}.csv",
        },
    )

    result = run_data_governance(
        categories=["full_a", "hs300"],
        as_of="20240510",
        output_dir=tmp_path / "reports",
        macro_release_calendar_root=tmp_path / "calendar",
    )

    assert calls == {"load": 1, "freeze": 1, "assess": 2}
    assert observed_evidence == [runtime.evidence, runtime.evidence]
    assert observed_evidence[0] is observed_evidence[1]
    expected_payload = json.loads(json.dumps(runtime.evidence.to_dict()))
    for report in result["reports"]:
        metadata = report["metadata"]["macro_readiness_runtime"]
        assert metadata["macro_readiness_evidence"] == expected_payload
        assert (
            metadata["macro_readiness_evidence_semantic_sha256"]
            == runtime.evidence.semantic_sha256
        )


def test_retired_market_mart_module_is_physically_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.market.intelligence_mart")
