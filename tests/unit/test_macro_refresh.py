from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.market import macro_mart


TARGET = "20240510"
CAPTURED_AT = datetime(2024, 5, 10, 8, 30, tzinfo=timezone.utc)


class _FakeTushare:
    def __init__(self, *, empty_endpoint: str = "") -> None:
        self.empty_endpoint = empty_endpoint

    @staticmethod
    def _months() -> list[str]:
        return list(
            pd.period_range("2023-05", "2024-04", freq="M").strftime("%Y%m")
        )

    def _frame(self, endpoint: str, fields: tuple[str, ...]) -> pd.DataFrame:
        if endpoint == self.empty_endpoint:
            return pd.DataFrame()
        rows = []
        for index, month in enumerate(self._months()):
            row: dict[str, object] = {"month": month}
            for field in fields:
                row[field] = 8.6 if field == "m2_yoy" else 1.0 + index
            rows.append(row)
        return pd.DataFrame(rows)

    def cn_pmi(self, **_kwargs: object) -> pd.DataFrame:
        frame = self._frame("cn_pmi", ("PMI010000",))
        if not frame.empty:
            frame = frame.rename(columns={"month": "MONTH"})
        return frame

    def cn_cpi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame("cn_cpi", ("nt_yoy",))

    def cn_ppi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame("cn_ppi", ("ppi_yoy",))

    def sf_month(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame("sf_month", ("inc_month",))

    def cn_m(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame("cn_m", ("m1_yoy", "m2_yoy"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    market_root = tmp_path / "parquet" / "cn"
    macro_root = market_root / "macro_daily"
    bars_root = market_root / "bars"
    macro_root.mkdir(parents=True)
    bars_root.mkdir(parents=True)

    dates = pd.bdate_range(end="2024-05-10", periods=300)
    rows: list[dict[str, object]] = []
    for symbol_index in range(120):
        symbol = f"{symbol_index:06d}.SZ"
        direction = 1.0 if symbol_index % 3 else -0.25
        for date_index, trade_date in enumerate(dates):
            close = (
                10.0
                + direction * date_index * 0.01
                + math.sin((date_index + symbol_index) / 11.0) * 0.03
            )
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date.strftime("%Y%m%d"),
                    "close": close,
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
    pd.DataFrame(
        [{"trade_date": "2024-05-09", "macro_score": -0.1}]
    ).to_parquet(legacy_macro, index=False)
    other_root = market_root / "daily_basic"
    other_root.mkdir()
    other_table = other_root / "part.parquet"
    pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": TARGET}]).to_parquet(
        other_table,
        index=False,
    )
    catalog = {
        "schema_version": macro_mart.LEGACY_CATALOG_SCHEMA,
        "required_tables": ["daily_basic", "macro_daily"],
        "tables": {
            "daily_basic": {
                "logical_table": "daily_basic",
                "path": str(other_table),
                "table_root": str(other_root),
                "date_column": "trade_date",
            },
            "macro_daily": {
                "logical_table": "macro_daily",
                "path": str(legacy_macro),
                "table_root": str(macro_root),
                "date_column": "trade_date",
            },
        },
    }
    catalog_path = market_root / "_catalog.json"
    catalog_path.write_text(json.dumps(catalog, sort_keys=True), encoding="utf-8")
    pointer = {
        "snapshot_id": "fixture",
        "status": "OK",
        "table_root": str(bars_root),
        "latest_available_trade_date": TARGET,
        "latest_complete_trade_date": TARGET,
        "latest_trade_date": TARGET,
        "blockers": [],
    }
    pointer_path = market_root / "_latest.json"
    pointer_path.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    return macro_root, catalog_path, pointer_path, bars


def _refresh(
    macro_root: Path,
    catalog_path: Path,
    pointer_path: Path,
    *,
    run_id: str = "macro-refresh-fixture",
) -> dict[str, object]:
    return macro_mart.refresh_cn_macro_mart(
        market="CN",
        as_of=TARGET,
        data_root=macro_root,
        run_id=run_id,
        expected_catalog_sha256=_sha(catalog_path),
        expected_market_pointer_sha256=_sha(pointer_path),
        allow_live=True,
    )


def test_refresh_promotes_hash_bound_strict_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, bars = _workspace(tmp_path)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )

    result = _refresh(macro_root, catalog_path, pointer_path)

    assert result["status"] == "promoted"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["schema_version"] == macro_mart.STRICT_CATALOG_SCHEMA
    assert catalog["tables"]["daily_basic"]["path"] == (
        "daily_basic/part.parquet"
    )
    frame, manifest = macro_mart.read_macro_mart(data_root=macro_root)
    assert manifest["transform_version"] == macro_mart.TRANSFORM_VERSION
    assert manifest["historical_replay_eligible"] is False
    assert frame.iloc[0]["policy_signal"] == "neutral"

    ordered = bars.sort_values(["ts_code", "trade_date"]).copy()
    ordered["return"] = ordered.groupby("ts_code")["close"].pct_change(
        fill_method=None
    )
    returns = ordered.dropna(subset=["return"])
    recent = returns.groupby("ts_code")["return"].tail(20)
    expected_symbol_mean = recent.groupby(returns.loc[recent.index, "ts_code"]).mean()
    expected_macro = float(np.clip(expected_symbol_mean.mean() * 20.0, -1.0, 1.0))
    expected_breadth = float(expected_symbol_mean.gt(0.0).mean())
    assert frame.iloc[0]["macro_score"] == pytest.approx(expected_macro)
    assert frame.iloc[0]["liquidity_score"] == pytest.approx(expected_breadth)
    journal = json.loads(
        Path(str(result["transaction_journal"])).read_text(encoding="utf-8")
    )
    assert journal["state"] == "committed"


def test_refresh_requires_explicit_live_without_writes(tmp_path: Path) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    before = catalog_path.read_bytes()
    with pytest.raises(macro_mart.MacroMartPromotionError, match="live_not_authorized"):
        macro_mart.refresh_cn_macro_mart(
            market="CN",
            data_root=macro_root,
            run_id="no-live",
            expected_catalog_sha256=_sha(catalog_path),
            expected_market_pointer_sha256=_sha(pointer_path),
            allow_live=False,
        )
    assert catalog_path.read_bytes() == before
    assert not (macro_root / "_generations").exists()


def test_provider_failure_preserves_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    before = catalog_path.read_bytes()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(empty_endpoint="cn_m"),
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_response_empty:cn_m",
    ):
        _refresh(macro_root, catalog_path, pointer_path)
    assert catalog_path.read_bytes() == before
    assert not (macro_root / "_generations").exists()


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("duplicate", "macro_provider_month_duplicate:cn_cpi"),
        ("future", "macro_provider_future_month_rejected:cn_cpi"),
        ("missing", "macro_provider_schema_invalid:cn_cpi"),
        ("nonfinite", "macro_provider_value_invalid:cn_cpi:nt_yoy"),
    ],
)
def test_provider_contract_rejects_malformed_rows_before_any_write(
    mutation: str,
    blocker: str,
) -> None:
    client = _FakeTushare()
    base = client.cn_cpi

    def _malformed(**kwargs: object) -> pd.DataFrame:
        frame = base(**kwargs)
        if mutation == "duplicate":
            return pd.concat([frame, frame.tail(1)], ignore_index=True)
        if mutation == "future":
            frame.loc[frame.index[-1], "month"] = "202406"
        elif mutation == "missing":
            frame = frame.drop(columns=["nt_yoy"])
        elif mutation == "nonfinite":
            frame.loc[frame.index[-1], "nt_yoy"] = np.inf
        return frame

    client.cn_cpi = _malformed  # type: ignore[method-assign]
    with pytest.raises(macro_mart.MacroMartPromotionError, match=blocker):
        macro_mart._fetch_provider_bundle(
            client=client,
            trade_date=TARGET,
            captured_at=CAPTURED_AT,
        )


def test_market_pointer_aba_is_rejected_before_catalog_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    before_catalog = catalog_path.read_bytes()
    pointer_bytes = pointer_path.read_bytes()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    original = macro_mart._write_primary_generation

    def _write_then_aba(**kwargs: object):
        result = original(**kwargs)
        macro_mart._atomic_write_bytes(pointer_path, pointer_bytes)
        return result

    monkeypatch.setattr(macro_mart, "_write_primary_generation", _write_then_aba)
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_market_pointer_cas_mismatch",
    ):
        _refresh(macro_root, catalog_path, pointer_path)
    assert catalog_path.read_bytes() == before_catalog


def test_refresh_is_idempotent_and_recovers_switched_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    first = _refresh(macro_root, catalog_path, pointer_path)
    journal_path = Path(str(first["transaction_journal"]))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    journal_path.write_text(json.dumps(journal, sort_keys=True), encoding="utf-8")

    def _no_second_provider() -> object:
        raise AssertionError("idempotent refresh must not call provider")

    monkeypatch.setattr(macro_mart, "_build_tushare_client", _no_second_provider)
    second = _refresh(
        macro_root,
        catalog_path,
        pointer_path,
        run_id="unused-second-run",
    )
    assert second["status"] == "already_current"
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "committed"


def test_recovery_restores_exact_old_catalog_when_switched_generation_is_bad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    result = _refresh(macro_root, catalog_path, pointer_path)
    journal_path = Path(str(result["transaction_journal"]))
    transaction = journal_path.parent
    expected_old = (transaction / "old_catalog.json").read_bytes()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    journal_path.write_text(json.dumps(journal, sort_keys=True), encoding="utf-8")
    provider_path = Path(
        str(result["manifest"]["resolved_provider_bundle"])
    )
    provider_path.write_bytes(provider_path.read_bytes() + b"tampered")

    with macro_mart._catalog_writer_lock(macro_root.parent):
        macro_mart._recover_catalog_transactions(
            root=macro_root,
            catalog_path=catalog_path,
        )

    assert catalog_path.read_bytes() == expected_old
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "rolled_back"
