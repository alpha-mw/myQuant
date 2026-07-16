from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.macro.nbs_pmi import (
    NBS_PMI_PARSER_CONTRACT_SHA256,
    NBS_PMI_PARSER_VERSION,
    NbsPmiCapture,
    NbsPmiPermanentError,
    NbsPmiTransientError,
    parse_nbs_cn_pmi_html,
)
from quant_investor.market import macro_mart
from tests.helpers.macro_fixture import bind_macro_generation


TARGET = "20240510"
CAPTURED_AT = datetime(2024, 5, 10, 8, 30, tzinfo=timezone.utc)
NBS_URL = "https://www.stats.gov.cn/sj/zxfb/202404/t20240430_1.html"


def _nbs_body() -> bytes:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="ArticleTitle" content="2024年4月中国采购经理指数运行情况">
  <meta name="PubDate" content="2024/04/30 09:30">
  <title>2024年4月中国采购经理指数运行情况 - 国家统计局</title>
</head>
<body>
  <main>
    <p>4月份，制造业采购经理指数（PMI）为50.4%。</p>
  </main>
</body>
</html>
""".encode("utf-8")


def _nbs_capture() -> NbsPmiCapture:
    body = _nbs_body()
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


def _patch_official_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _nbs_capture()

    def _fetch(url: str) -> NbsPmiCapture:
        assert url == NBS_URL
        return capture

    monkeypatch.setattr(
        macro_mart,
        "fetch_nbs_cn_pmi",
        _fetch,
    )


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
        nbs_cn_pmi_url=NBS_URL,
    )


def test_refresh_promotes_hash_bound_strict_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, bars = _workspace(tmp_path)
    clock_calls = iter(
        CAPTURED_AT + timedelta(seconds=index) for index in range(100)
    )
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clock_calls))
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)

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
    assert manifest["source"] == macro_mart.SOURCE_OFFICIAL_FIRST
    assert manifest["source_priority"] == macro_mart.SOURCE_OFFICIAL
    assert manifest["provider_fallback_used"] is False
    assert frame.iloc[0]["policy_signal"] == "neutral"

    provider_path = Path(str(manifest["resolved_provider_bundle"]))
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    capture_entry = provider["endpoints"]["cn_pmi"]["raw_capture"]
    capture_path = provider_path.parent / capture_entry["path"]
    assert provider["schema_version"] == macro_mart.PROVIDER_BUNDLE_SCHEMA
    assert provider["source_policy"] == macro_mart.PROVIDER_SOURCE_POLICY
    assert provider["fallback_used"] is False
    assert provider["official_attempts"][0]["requested_url"] == NBS_URL
    assert provider["endpoints"]["cn_pmi"]["source_system"] == "nbs_official"
    assert capture_path.read_bytes() == _nbs_body()
    assert capture_entry["sha256"] == _sha(capture_path)
    assert capture_entry["size_bytes"] == capture_path.stat().st_size
    assert manifest["provider_capture_files"] == [
        {
            "endpoint": "cn_pmi",
            "path": capture_entry["path"],
            "sha256": capture_entry["sha256"],
            "size_bytes": capture_entry["size_bytes"],
        }
    ]
    catalog_entry = catalog["tables"]["macro_daily"]
    assert catalog_entry["provider_capture_files_sha256"] == (
        manifest["provider_capture_files_sha256"]
    )
    assert manifest["primary_provenance"][
        "provider_capture_files_sha256"
    ] == manifest["provider_capture_files_sha256"]

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
    source_release_at = pd.Timestamp(
        provider["selected_inputs"]["cn_pmi"]["source_release_at"]
    )
    fetch_completed_at = pd.Timestamp(
        capture_entry["fetch_completed_at"]
    )
    decision_cutoff_at = pd.Timestamp(provider["decision_cutoff_at"])
    committed_at = pd.Timestamp(journal["committed_at"])
    assert source_release_at < fetch_completed_at <= decision_cutoff_at
    assert decision_cutoff_at <= committed_at
    assert (
        capture_entry["source_release_at"]
        != capture_entry["fetch_completed_at"]
    )


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
            nbs_cn_pmi_url=NBS_URL,
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
    _patch_official_fetch(monkeypatch)
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
            nbs_cn_pmi_url=NBS_URL,
            nbs_fetcher=lambda _url: _nbs_capture(),
        )


def test_tushare_endpoint_retries_anomalous_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeTushare()
    original = client.cn_cpi
    calls = 0

    def _empty_once(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame()
        return original(**kwargs)

    client.cn_cpi = _empty_once  # type: ignore[method-assign]
    sleeps: list[float] = []
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)

    entry, chosen, completed = macro_mart._fetch_tushare_endpoint(
        client=client,
        endpoint="cn_cpi",
        spec=macro_mart._ENDPOINT_SPECS["cn_cpi"],
        start_month="202305",
        end_month="202405",
        source_system="tushare_primary",
        source_role="configured_primary",
        sleeper=sleeps.append,
    )

    assert entry["attempt_count"] == 2
    assert chosen["month"] == "202404"
    assert completed == CAPTURED_AT
    assert sleeps == [0.25]


def test_provider_cutoff_preserves_subsecond_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_completed = CAPTURED_AT + timedelta(milliseconds=500)
    tushare_completed = CAPTURED_AT + timedelta(milliseconds=700)
    capture = replace(
        _nbs_capture(),
        fetch_completed_at=official_completed.isoformat(),
    )
    monkeypatch.setattr(
        macro_mart,
        "_utc_now",
        lambda: tushare_completed,
    )

    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: capture,
    )

    cutoff = pd.Timestamp(result.bundle["decision_cutoff_at"])
    assert cutoff == pd.Timestamp(tushare_completed)
    assert cutoff > pd.Timestamp(official_completed)
    assert all(
        pd.Timestamp(value["observed_available_at"]) <= cutoff
        for value in result.bundle["selected_inputs"].values()
    )


def test_transient_official_failure_requires_explicit_tushare_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)

    def _transient(_url: str) -> NbsPmiCapture:
        raise NbsPmiTransientError("nbs_pmi_dns_unavailable")

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_official_provider_transient:cn_pmi",
    ):
        macro_mart._fetch_provider_bundle(
            client=_FakeTushare(),
            trade_date=TARGET,
            captured_at=CAPTURED_AT,
            nbs_cn_pmi_url=NBS_URL,
            nbs_fetcher=_transient,
        )

    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        allow_tushare_fallback=True,
        nbs_fetcher=_transient,
    )

    assert result.captures == {}
    assert result.bundle["source"] == macro_mart.SOURCE_TUSHARE
    assert result.bundle["source_priority"] == macro_mart.SOURCE_TUSHARE
    assert result.bundle["fallback_authorized"] is True
    assert result.bundle["fallback_used"] is True
    assert result.bundle["official_release_timestamps_claimed"] is False
    attempt = result.bundle["official_attempts"][0]
    assert attempt["endpoint"] == "cn_pmi"
    assert attempt["status"] == "transient_failure"
    assert attempt["source_system"] == "nbs_official"
    assert attempt["requested_url"] == NBS_URL
    assert attempt["trigger_category"] == "transport_transient"
    assert attempt["fallback_provider"] == "tushare_pro"
    assert attempt["reason"] == "nbs_pmi_dns_unavailable"
    assert pd.Timestamp(attempt["attempt_started_at"]) <= pd.Timestamp(
        attempt["attempt_completed_at"]
    )
    pmi = result.bundle["endpoints"]["cn_pmi"]
    selected = result.bundle["selected_inputs"]["cn_pmi"]
    assert pmi["source_system"] == "tushare_fallback"
    assert pmi["source_role"] == "explicit_transport_fallback"
    assert "raw_capture" not in pmi
    assert selected["source_system"] == "tushare_fallback"
    assert selected["official_release_timestamp_known"] is False


def test_permanent_official_failure_never_uses_tushare_fallback() -> None:
    def _permanent(_url: str) -> NbsPmiCapture:
        raise NbsPmiPermanentError("nbs_pmi_article_title_invalid")

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_official_provider_invalid:cn_pmi",
    ):
        macro_mart._fetch_provider_bundle(
            client=object(),
            trade_date=TARGET,
            captured_at=CAPTURED_AT,
            nbs_cn_pmi_url=NBS_URL,
            allow_tushare_fallback=True,
            nbs_fetcher=_permanent,
        )


def test_v2_validator_rejects_cn_pmi_laundered_as_tushare_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: _nbs_capture(),
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_OFFICIAL_FIRST,
        "source_priority": macro_mart.SOURCE_OFFICIAL,
        "provider_fallback_used": False,
    }
    macro_mart._validate_provider_bundle(bundle, manifest=manifest)
    bundle["endpoints"]["cn_pmi"]["source_system"] = "tushare_primary"
    bundle["endpoints"]["cn_pmi"]["source_role"] = "configured_primary"
    bundle["selected_inputs"]["cn_pmi"][
        "source_system"
    ] = "tushare_primary"
    bundle["selected_inputs"]["cn_pmi"][
        "source_role"
    ] = "configured_primary"

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_bundle_endpoint_source_invalid",
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


def test_v2_validator_rejects_non_nbs_intermediate_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: _nbs_capture(),
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_OFFICIAL_FIRST,
        "source_priority": macro_mart.SOURCE_OFFICIAL,
        "provider_fallback_used": False,
    }
    raw = bundle["endpoints"]["cn_pmi"]["raw_capture"]
    raw["redirect_chain"] = [
        NBS_URL,
        "https://evil.example/t20240430_1.html",
        NBS_URL,
    ]

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_bundle_official_evidence_invalid",
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


def test_v2_validator_replays_official_redirect_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: _nbs_capture(),
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_OFFICIAL_FIRST,
        "source_priority": macro_mart.SOURCE_OFFICIAL,
        "provider_fallback_used": False,
    }
    bundle["endpoints"]["cn_pmi"]["raw_capture"]["redirect_chain"] = [
        NBS_URL
    ] * (macro_mart.NBS_PMI_MAX_REDIRECTS + 2)

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_bundle_official_evidence_invalid",
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            "endpoint_completion_mismatch",
            "macro_provider_bundle_endpoint_completion_invalid",
        ),
        (
            "cutoff_after_all_completions",
            "macro_provider_bundle_completion_cutoff_mismatch",
        ),
    ],
)
def test_v2_validator_binds_endpoint_completion_to_exact_global_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    blocker: str,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: _nbs_capture(),
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_OFFICIAL_FIRST,
        "source_priority": macro_mart.SOURCE_OFFICIAL,
        "provider_fallback_used": False,
    }
    macro_mart._validate_provider_bundle(bundle, manifest=manifest)

    if mutation == "endpoint_completion_mismatch":
        bundle["endpoints"]["cn_cpi"]["fetch_completed_at"] = (
            CAPTURED_AT + timedelta(days=1)
        ).isoformat()
    else:
        later_cutoff = (CAPTURED_AT + timedelta(seconds=1)).isoformat()
        bundle["fetched_at"] = later_cutoff
        bundle["decision_cutoff_at"] = later_cutoff

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match=blocker,
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


@pytest.mark.parametrize(
    ("field", "blocker"),
    [
        ("decision_cutoff_at", "macro_provider_bundle_cutoff_invalid"),
        ("fetched_at", "macro_provider_bundle_cutoff_invalid"),
        (
            "endpoint_fetch_completed_at",
            "macro_provider_bundle_endpoint_completion_invalid",
        ),
        (
            "selected_observed_available_at",
            "macro_provider_bundle_selected_input_after_cutoff",
        ),
        ("attempt_started_at", "macro_provider_bundle_attempt_clock_invalid"),
        ("attempt_completed_at", "macro_provider_bundle_attempt_clock_invalid"),
        (
            "source_release_at",
            "macro_provider_bundle_official_evidence_invalid",
        ),
        (
            "raw_fetch_started_at",
            "macro_provider_bundle_official_evidence_invalid",
        ),
        (
            "raw_fetch_completed_at",
            "macro_provider_bundle_official_evidence_invalid",
        ),
    ],
)
def test_v2_validator_rejects_naive_provenance_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    blocker: str,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        nbs_fetcher=lambda _url: _nbs_capture(),
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_OFFICIAL_FIRST,
        "source_priority": macro_mart.SOURCE_OFFICIAL,
        "provider_fallback_used": False,
    }
    naive = CAPTURED_AT.replace(tzinfo=None).isoformat()
    raw = bundle["endpoints"]["cn_pmi"]["raw_capture"]
    if field in {"decision_cutoff_at", "fetched_at"}:
        bundle[field] = naive
    elif field == "endpoint_fetch_completed_at":
        bundle["endpoints"]["cn_cpi"]["fetch_completed_at"] = naive
    elif field == "selected_observed_available_at":
        bundle["selected_inputs"]["cn_cpi"][
            "observed_available_at"
        ] = naive
    elif field in {"attempt_started_at", "attempt_completed_at"}:
        bundle["official_attempts"][0][field] = naive
    elif field == "source_release_at":
        bundle["selected_inputs"]["cn_pmi"]["source_release_at"] = naive
    elif field == "raw_fetch_started_at":
        raw["fetch_started_at"] = naive
    else:
        raw["fetch_completed_at"] = naive

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match=blocker,
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


def test_v2_validator_requires_official_failure_before_fallback_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)

    def _transient(_url: str) -> NbsPmiCapture:
        raise NbsPmiTransientError("nbs_pmi_dns_unavailable")

    result = macro_mart._fetch_provider_bundle(
        client=_FakeTushare(),
        trade_date=TARGET,
        captured_at=CAPTURED_AT,
        nbs_cn_pmi_url=NBS_URL,
        allow_tushare_fallback=True,
        nbs_fetcher=_transient,
    )
    bundle = json.loads(json.dumps(result.bundle))
    manifest = {
        "as_of": "2024-05-10",
        "provider_bundle_schema_version": macro_mart.PROVIDER_BUNDLE_SCHEMA,
        "source_policy": macro_mart.PROVIDER_SOURCE_POLICY,
        "source": macro_mart.SOURCE_TUSHARE,
        "source_priority": macro_mart.SOURCE_TUSHARE,
        "provider_fallback_used": True,
    }
    macro_mart._validate_provider_bundle(bundle, manifest=manifest)

    before_attempt = (CAPTURED_AT - timedelta(seconds=1)).isoformat()
    bundle["endpoints"]["cn_pmi"][
        "fetch_completed_at"
    ] = before_attempt
    bundle["selected_inputs"]["cn_pmi"][
        "observed_available_at"
    ] = before_attempt

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_bundle_fallback_evidence_invalid",
    ):
        macro_mart._validate_provider_bundle(bundle, manifest=manifest)


def test_legacy_v1_generation_is_not_current_equivalent(tmp_path: Path) -> None:
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    root.mkdir(parents=True)
    bind_macro_generation(
        root,
        generation_id="legacy-v1",
        row={
            "trade_date": "2024-05-10",
            "macro_score": 0.1,
            "liquidity_score": 0.2,
            "volatility_percentile": 0.3,
            "policy_signal": "neutral",
            "source": macro_mart.SOURCE_TUSHARE,
            "source_priority": macro_mart.SOURCE_TUSHARE,
            "pit_status": "market_point_in_time",
            "fetched_at": "2024-05-10T08:00:00+00:00",
        },
    )

    equivalent = macro_mart._current_macro_is_equivalent(
        root=root,
        trade_date=TARGET,
        market_pointer_sha256="1" * 64,
        current_at=CAPTURED_AT,
    )

    assert equivalent is None


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
    _patch_official_fetch(monkeypatch)
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
    _patch_official_fetch(monkeypatch)
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


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_sidecar_failure_is_fail_closed_and_recovery_restores_old_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)
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
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    capture_path = (
        provider_path.parent
        / provider["endpoints"]["cn_pmi"]["raw_capture"]["path"]
    )
    if mutation == "missing":
        capture_path.unlink()
    else:
        capture_path.write_bytes(capture_path.read_bytes() + b"tampered")

    with pytest.raises(macro_mart.MacroMartPromotionError):
        macro_mart.read_macro_mart(data_root=macro_root)

    with macro_mart._catalog_writer_lock(macro_root.parent):
        macro_mart._recover_catalog_transactions(
            root=macro_root,
            catalog_path=catalog_path,
        )

    assert catalog_path.read_bytes() == expected_old
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "rolled_back"
