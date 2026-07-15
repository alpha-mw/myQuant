from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market import macro_mart


TARGET = "20240510"
STARTED_AT = datetime(2024, 5, 10, 8, 0, tzinfo=timezone.utc)


class _MonthlyProvider:
    def __init__(self, *, latest_month: str) -> None:
        self.latest_month = latest_month

    def _frame(self, fields: tuple[str, ...]) -> pd.DataFrame:
        months = pd.period_range(
            end=pd.Period(self.latest_month, freq="M"),
            periods=12,
            freq="M",
        ).strftime("%Y%m")
        return pd.DataFrame(
            [
                {
                    "month": month,
                    **{
                        field: 8.6 if field == "m2_yoy" else 1.0 + index
                        for field in fields
                    },
                }
                for index, month in enumerate(months)
            ]
        )

    def cn_pmi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("PMI010000",))

    def cn_cpi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("nt_yoy",))

    def cn_ppi(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("ppi_yoy",))

    def sf_month(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("inc_month",))

    def cn_m(self, **_kwargs: object) -> pd.DataFrame:
        return self._frame(("m1_yoy", "m2_yoy"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    market_root = tmp_path / "parquet" / "cn"
    macro_root = market_root / "macro_daily"
    bars_root = market_root / "bars"
    macro_root.mkdir(parents=True)
    bars_root.mkdir()
    catalog_path = market_root / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.LEGACY_CATALOG_SCHEMA,
                "required_tables": [],
                "tables": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pointer_path = market_root / "_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "snapshot_id": "freshness-fixture",
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
    return macro_root, catalog_path, pointer_path


def _refresh(
    macro_root: Path,
    catalog_path: Path,
    pointer_path: Path,
) -> dict[str, object]:
    return macro_mart.refresh_cn_macro_mart(
        market="CN",
        as_of=TARGET,
        data_root=macro_root,
        run_id="freshness-fixture",
        expected_catalog_sha256=_sha(catalog_path),
        expected_market_pointer_sha256=_sha(pointer_path),
        allow_live=True,
    )


def test_provider_io_completion_crossing_capture_window_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path = _minimal_workspace(tmp_path)
    catalog_before = catalog_path.read_bytes()
    expired = datetime(2024, 5, 13, 7, 0, 1, tzinfo=timezone.utc)
    clocks = iter((STARTED_AT, expired, expired))
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clocks))
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _MonthlyProvider(latest_month="202404"),
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_capture_window_expired",
    ):
        _refresh(macro_root, catalog_path, pointer_path)

    assert catalog_path.read_bytes() == catalog_before
    assert not (macro_root / "_generations").exists()


def test_provider_bundle_uses_post_io_completion_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_at = datetime(2024, 5, 10, 9, 2, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: completed_at)

    bundle = macro_mart._fetch_provider_bundle(
        client=_MonthlyProvider(latest_month="202404"),
        trade_date=TARGET,
        captured_at=STARTED_AT,
    )

    assert bundle["fetched_at"] == completed_at.isoformat()
    assert bundle["decision_cutoff_at"] == completed_at.isoformat()
    assert {
        selected["observed_available_at"]
        for selected in bundle["selected_inputs"].values()
    } == {completed_at.isoformat()}


def test_provider_latest_month_from_2020_is_stale_for_2024_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: STARTED_AT)

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_latest_month_stale:cn_cpi",
    ):
        macro_mart._fetch_provider_bundle(
            client=_MonthlyProvider(latest_month="202012"),
            trade_date=TARGET,
            captured_at=STARTED_AT,
        )


def test_current_cutoff_requires_june_pmi_under_endpoint_lag_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    assert macro_mart._expected_latest_provider_month(
        cutoff,
        max_release_lag_days=15,
    ) == "202606"
    assert macro_mart._expected_latest_provider_month(
        cutoff,
        max_release_lag_days=45,
    ) == "202605"
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: cutoff)

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_latest_month_stale:cn_pmi",
    ):
        macro_mart._fetch_provider_bundle(
            client=_MonthlyProvider(latest_month="202605"),
            trade_date="20260714",
            captured_at=cutoff,
        )


def test_capture_window_is_revalidated_immediately_before_catalog_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path = _minimal_workspace(tmp_path)
    before_switch = datetime(2024, 5, 13, 7, 0, 1, tzinfo=timezone.utc)
    clocks = iter((STARTED_AT, STARTED_AT, STARTED_AT, before_switch))
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clocks))
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _MonthlyProvider(latest_month="202404"),
    )
    monkeypatch.setattr(
        macro_mart,
        "_load_market_inputs",
        lambda *_args, **_kwargs: (pd.DataFrame(), [], "0" * 64),
    )
    monkeypatch.setattr(
        macro_mart,
        "_derive_macro_frame",
        lambda *_args, **_kwargs: (
            pd.DataFrame([{"trade_date": TARGET}]),
            {"symbol_count": 1},
        ),
    )
    monkeypatch.setattr(
        macro_mart,
        "_write_primary_generation",
        lambda **_kwargs: ({"generation_id": "freshness-fixture"}, object()),
    )
    monkeypatch.setattr(
        macro_mart,
        "_strict_catalog_payload",
        lambda **_kwargs: {},
    )

    def _unexpected_publish(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("expired refresh must not switch the catalog")

    monkeypatch.setattr(
        macro_mart,
        "_publish_catalog_generation",
        _unexpected_publish,
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_capture_window_expired",
    ):
        _refresh(macro_root, catalog_path, pointer_path)
