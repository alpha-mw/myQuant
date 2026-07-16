from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.macro.nbs_pmi import (
    NBS_PMI_PARSER_CONTRACT_SHA256,
    NBS_PMI_PARSER_VERSION,
    NbsPmiCapture,
    NbsPmiTransientError,
    parse_nbs_cn_pmi_html,
)
from quant_investor.market import macro_mart
from quant_investor.macro.store import pointer_sha256
from tests.helpers.macro_fixture import write_ready_macro_observations


TARGET = "20240510"
CAPTURED_AT = datetime(2024, 5, 10, 8, 30, tzinfo=timezone.utc)
NBS_URL = "https://www.stats.gov.cn/sj/zxfb/202404/t20240430_1.html"


def _nbs_capture() -> NbsPmiCapture:
    body = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="ArticleTitle" content="2024年4月中国采购经理指数运行情况">
  <meta name="PubDate" content="2024/04/30 09:30">
  <title>2024年4月中国采购经理指数运行情况 - 国家统计局</title>
</head>
<body>
  <main><p>4月份，制造业采购经理指数（PMI）为50.4%。</p></main>
</body>
</html>
""".encode("utf-8")
    parsed = parse_nbs_cn_pmi_html(body, source_url=NBS_URL)
    return NbsPmiCapture(
        month=parsed.month,
        value=parsed.value,
        source_url=parsed.source_url,
        source_record_id=parsed.source_record_id,
        article_title=parsed.article_title,
        source_release_at=parsed.source_release_at,
        fetch_started_at=CAPTURED_AT.isoformat(),
        fetch_completed_at=CAPTURED_AT.isoformat(),
        content_type="text/html",
        charset="utf-8",
        body_bytes=body,
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_size_bytes=len(body),
        parser_version=NBS_PMI_PARSER_VERSION,
        parser_contract_sha256=NBS_PMI_PARSER_CONTRACT_SHA256,
        redirect_chain=(NBS_URL,),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_contract(market_root: Path) -> dict[str, object]:
    generation = market_root / "macro_daily" / "_generations" / "next"
    generation.mkdir(parents=True)
    table = generation / "part.parquet"
    pd.DataFrame([{"trade_date": "2024-05-10"}]).to_parquet(
        table,
        index=False,
    )
    manifest = generation / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    provider = generation / "provider_bundle.json"
    provider.write_text("{}\n", encoding="utf-8")
    return {
        "generation_id": "next",
        "as_of": TARGET,
        "row_count": 1,
        "parquet_sha256": _sha(table),
        "generation_manifest_sha256": _sha(manifest),
        "provider_bundle_sha256": _sha(provider),
    }


def _strict_catalog_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    market_root = tmp_path / "cn"
    required_root = market_root / "daily_basic"
    required_root.mkdir(parents=True)
    required_table = required_root / "part.parquet"
    pd.DataFrame([{"trade_date": TARGET}]).to_parquet(
        required_table,
        index=False,
    )
    catalog: dict[str, object] = {
        "schema_version": macro_mart.STRICT_CATALOG_SCHEMA,
        "required_tables": ["daily_basic", "macro_daily"],
        "tables": {
            "daily_basic": {
                "path": "daily_basic/part.parquet",
                "table_root": "daily_basic",
                "sha256": _sha(required_table),
            },
            "macro_daily": {"path": "macro_daily/part.parquet"},
        },
    }
    return market_root, catalog, _generation_contract(market_root)


def test_strict_catalog_requires_hash_bound_readable_closure(
    tmp_path: Path,
) -> None:
    market_root, catalog, generation = _strict_catalog_fixture(tmp_path)
    catalog["required_tables"] = [
        "daily_basic",
        "missing_required",
        "macro_daily",
    ]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_required_entry_missing:missing_required",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )

    catalog["required_tables"] = ["daily_basic", "macro_daily"]
    catalog["tables"]["daily_basic"].pop("sha256")  # type: ignore[index,union-attr]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_required_hash_missing:daily_basic",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )

    catalog["tables"]["daily_basic"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_required_hash_mismatch:daily_basic",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )

    outside = tmp_path / "outside.parquet"
    pd.DataFrame([{"trade_date": TARGET}]).to_parquet(outside, index=False)
    catalog["tables"]["daily_basic"]["path"] = str(outside)  # type: ignore[index]
    catalog["tables"]["daily_basic"]["sha256"] = _sha(outside)  # type: ignore[index]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_member_invalid:daily_basic:path",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )


def test_strict_catalog_rejects_unreadable_and_intelligence_residue(
    tmp_path: Path,
) -> None:
    market_root, catalog, generation = _strict_catalog_fixture(tmp_path)
    required_table = market_root / "daily_basic" / "part.parquet"
    required_table.write_bytes(b"not parquet")
    catalog["tables"]["daily_basic"]["sha256"] = _sha(required_table)  # type: ignore[index]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_required_table_unreadable:daily_basic",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )

    pd.DataFrame([{"trade_date": TARGET}]).to_parquet(
        required_table,
        index=False,
    )
    catalog["tables"]["daily_basic"]["sha256"] = _sha(required_table)  # type: ignore[index]
    catalog["tables"]["daily_basic"]["columns"] = ["intelligence_score"]  # type: ignore[index]
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_retired_intelligence_present",
    ):
        macro_mart._strict_catalog_payload(
            old_catalog=catalog,
            market_root=market_root,
            generation_manifest=generation,
        )


def test_recovery_rolls_back_when_market_pointer_changed(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "cn"
    macro_root = market_root / "macro_daily"
    macro_root.mkdir(parents=True)
    catalog_path = market_root / "_catalog.json"
    pointer_path = market_root / "_latest.json"
    old_bytes = b'{"catalog":"old"}\n'
    new_bytes = b'{"catalog":"new"}\n'
    expected_pointer = b'{"snapshot_id":"expected"}\n'
    catalog_path.write_bytes(old_bytes)
    pointer_path.write_bytes(expected_pointer)
    journal_path, journal = macro_mart._prepare_catalog_transaction(
        root=macro_root,
        run_id="pointer-crash",
        old_catalog_bytes=old_bytes,
        new_catalog_bytes=new_bytes,
        generation_id="pointer-crash",
        expected_market_pointer_sha256=hashlib.sha256(
            expected_pointer
        ).hexdigest(),
    )
    macro_mart._atomic_write_bytes(catalog_path, new_bytes)
    macro_mart._atomic_write_bytes(
        pointer_path,
        b'{"snapshot_id":"changed"}\n',
    )

    macro_mart._recover_catalog_transactions(
        root=macro_root,
        catalog_path=catalog_path,
    )

    assert catalog_path.read_bytes() == old_bytes
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["expected_market_pointer_sha256"]
    assert recovered["state"] == "rolled_back"
    assert recovered["detail"] == "macro_transaction_market_pointer_hash_mismatch"


def test_recovery_cleans_journalless_transaction_and_restores_switch(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "cn"
    macro_root = market_root / "macro_daily"
    transaction = macro_root / "_transactions" / "orphan"
    transaction.mkdir(parents=True)
    catalog_path = market_root / "_catalog.json"
    old_bytes = b'{"catalog":"old"}\n'
    new_bytes = b'{"catalog":"new"}\n'
    catalog_path.write_bytes(new_bytes)
    (transaction / "old_catalog.json").write_bytes(old_bytes)
    (transaction / "new_catalog.json").write_bytes(new_bytes)

    macro_mart._recover_catalog_transactions(
        root=macro_root,
        catalog_path=catalog_path,
    )

    assert catalog_path.read_bytes() == old_bytes
    assert not transaction.exists()

    empty = macro_root / "_transactions" / "empty-orphan"
    empty.mkdir()
    macro_mart._recover_catalog_transactions(
        root=macro_root,
        catalog_path=catalog_path,
    )
    assert not empty.exists()


def test_aborted_transaction_directory_can_be_reprepared_with_same_run_id(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "cn"
    macro_root = market_root / "macro_daily"
    macro_root.mkdir(parents=True)
    catalog_path = market_root / "_catalog.json"
    pointer_path = market_root / "_latest.json"
    old_bytes = b'{"catalog":"old"}\n'
    new_bytes = b'{"catalog":"new"}\n'
    pointer_bytes = b'{"snapshot_id":"stable"}\n'
    catalog_path.write_bytes(old_bytes)
    pointer_path.write_bytes(pointer_bytes)
    pointer_sha = hashlib.sha256(pointer_bytes).hexdigest()
    first_path, _ = macro_mart._prepare_catalog_transaction(
        root=macro_root,
        run_id="prepared-crash",
        old_catalog_bytes=old_bytes,
        new_catalog_bytes=new_bytes,
        generation_id="prepared-crash",
        expected_market_pointer_sha256=pointer_sha,
    )

    macro_mart._recover_catalog_transactions(
        root=macro_root,
        catalog_path=catalog_path,
    )
    assert json.loads(first_path.read_text(encoding="utf-8"))["state"] == "aborted"

    second_path, second = macro_mart._prepare_catalog_transaction(
        root=macro_root,
        run_id="prepared-crash",
        old_catalog_bytes=old_bytes,
        new_catalog_bytes=new_bytes,
        generation_id="prepared-crash",
        expected_market_pointer_sha256=pointer_sha,
    )
    assert second_path == first_path
    assert second["state"] == "prepared"


class _FakeTushare:
    @staticmethod
    def _frame(fields: tuple[str, ...]) -> pd.DataFrame:
        rows = []
        for index, month in enumerate(
            pd.period_range("2023-05", "2024-04", freq="M").strftime("%Y%m")
        ):
            row: dict[str, object] = {"month": month}
            for field in fields:
                row[field] = 8.6 if field == "m2_yoy" else 1.0 + index
            rows.append(row)
        return pd.DataFrame(rows)

    def cn_pmi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("PMI010000",)).rename(columns={"month": "MONTH"})

    def cn_cpi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("nt_yoy",))

    def cn_ppi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("ppi_yoy",))

    def sf_month(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("inc_month",))

    def cn_m(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("m1_yoy", "m2_yoy"))


def _refresh_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    market_root = tmp_path / "parquet" / "cn"
    macro_root = market_root / "macro_daily"
    bars_root = market_root / "bars"
    daily_root = market_root / "daily_basic"
    for directory in (macro_root, bars_root, daily_root):
        directory.mkdir(parents=True)
    dates = pd.bdate_range(end="2024-05-10", periods=300)
    rows = []
    for symbol_index in range(120):
        direction = 1.0 if symbol_index % 3 else -0.25
        for date_index, trade_date in enumerate(dates):
            rows.append(
                {
                    "ts_code": f"{symbol_index:06d}.SZ",
                    "trade_date": trade_date.strftime("%Y%m%d"),
                    "close": 10.0
                    + direction * date_index * 0.01
                    + math.sin((date_index + symbol_index) / 11.0) * 0.03,
                }
            )
    bars = pd.DataFrame(rows)
    for (year, month), frame in bars.groupby(
        [
            pd.to_datetime(bars["trade_date"]).dt.year,
            pd.to_datetime(bars["trade_date"]).dt.month,
        ]
    ):
        partition = bars_root / f"year={year}" / f"month={month:02d}"
        partition.mkdir(parents=True)
        frame.to_parquet(partition / "part.parquet", index=False)
    legacy_macro = macro_root / "part.parquet"
    pd.DataFrame([{"trade_date": "2024-05-09"}]).to_parquet(
        legacy_macro,
        index=False,
    )
    daily_table = daily_root / "part.parquet"
    pd.DataFrame([{"trade_date": TARGET}]).to_parquet(daily_table, index=False)
    catalog_path = market_root / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.LEGACY_CATALOG_SCHEMA,
                "required_tables": ["daily_basic", "macro_daily"],
                "tables": {
                    "daily_basic": {
                        "path": str(daily_table),
                        "table_root": str(daily_root),
                    },
                    "macro_daily": {
                        "path": str(legacy_macro),
                        "table_root": str(macro_root),
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pointer_path = market_root / "_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "snapshot_id": "fixture",
                "status": "OK",
                "table_root": str(bars_root),
                "latest_available_trade_date": TARGET,
                "latest_complete_trade_date": TARGET,
                "latest_trade_date": TARGET,
                "blockers": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_ready_macro_observations(
        market_root / "macro_observations",
        as_of="2024-05-10",
    )
    return macro_root, catalog_path, pointer_path


def test_same_run_retry_resumes_generation_without_second_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path = _refresh_workspace(tmp_path)
    nbs_capture = _nbs_capture()

    def _fetch_official(url: str) -> NbsPmiCapture:
        assert url == NBS_URL
        return nbs_capture

    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(macro_mart, "_build_tushare_client", _FakeTushare)
    monkeypatch.setattr(macro_mart, "fetch_nbs_cn_pmi", _fetch_official)
    original_write = macro_mart._write_primary_generation

    def _write_then_crash(**kwargs: object):
        original_write(**kwargs)
        raise RuntimeError("simulated crash after generation rename")

    monkeypatch.setattr(
        macro_mart,
        "_write_primary_generation",
        _write_then_crash,
    )
    call = {
        "market": "CN",
        "as_of": TARGET,
        "data_root": macro_root,
        "run_id": "same-run-retry",
        "expected_catalog_sha256": _sha(catalog_path),
        "expected_market_pointer_sha256": _sha(pointer_path),
        "macro_observations_root": macro_root.parent / "macro_observations",
        "expected_macro_observations_pointer_sha256": pointer_sha256(
            macro_root.parent / "macro_observations"
        ),
        "allow_live": True,
        "nbs_cn_pmi_url": NBS_URL,
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        macro_mart.refresh_cn_macro_mart(**call)

    landed_capture = (
        macro_root
        / "_generations"
        / "same-run-retry"
        / macro_mart._nbs_capture_relative_path(nbs_capture)
    )
    assert landed_capture.read_bytes() == nbs_capture.body_bytes
    assert _sha(landed_capture) == nbs_capture.body_sha256

    monkeypatch.setattr(macro_mart, "_write_primary_generation", original_write)

    def _provider_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("valid landed generation must be resumed")

    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        _provider_must_not_run,
    )
    monkeypatch.setattr(
        macro_mart,
        "fetch_nbs_cn_pmi",
        _provider_must_not_run,
    )

    mismatched_url_call = {
        **call,
        "nbs_cn_pmi_url": NBS_URL.replace("_1.html", "_2.html"),
    }
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_retry_generation_nbs_url_mismatch",
    ):
        macro_mart.refresh_cn_macro_mart(**mismatched_url_call)

    result = macro_mart.refresh_cn_macro_mart(**call)

    assert result["status"] == "promoted"
    assert result["run_id"] == "same-run-retry"
    assert Path(str(result["transaction_journal"])).exists()


def test_same_run_retry_cannot_reuse_fallback_without_matching_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path = _refresh_workspace(tmp_path)
    before_catalog = catalog_path.read_bytes()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(macro_mart, "_build_tushare_client", _FakeTushare)

    def _official_transient(_url: str) -> NbsPmiCapture:
        raise NbsPmiTransientError("nbs_pmi_network_unavailable")

    monkeypatch.setattr(macro_mart, "fetch_nbs_cn_pmi", _official_transient)
    original_write = macro_mart._write_primary_generation

    def _write_then_crash(**kwargs: object):
        original_write(**kwargs)
        raise RuntimeError("simulated crash after fallback generation rename")

    monkeypatch.setattr(
        macro_mart,
        "_write_primary_generation",
        _write_then_crash,
    )
    authorized_call = {
        "market": "CN",
        "as_of": TARGET,
        "data_root": macro_root,
        "run_id": "fallback-retry-authorization",
        "expected_catalog_sha256": _sha(catalog_path),
        "expected_market_pointer_sha256": _sha(pointer_path),
        "macro_observations_root": macro_root.parent / "macro_observations",
        "expected_macro_observations_pointer_sha256": pointer_sha256(
            macro_root.parent / "macro_observations"
        ),
        "allow_live": True,
        "nbs_cn_pmi_url": NBS_URL,
        "allow_tushare_fallback": True,
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        macro_mart.refresh_cn_macro_mart(**authorized_call)

    monkeypatch.setattr(macro_mart, "_write_primary_generation", original_write)

    def _provider_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retry policy mismatch must fail before provider I/O")

    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        _provider_must_not_run,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_retry_generation_fallback_authorization_mismatch",
    ):
        macro_mart.refresh_cn_macro_mart(
            **{
                **authorized_call,
                "allow_tushare_fallback": False,
            }
        )

    assert catalog_path.read_bytes() == before_catalog
