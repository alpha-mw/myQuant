from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.macro.nbs_pmi import (
    NbsPmiCapture,
    parse_nbs_cn_pmi_html,
)
from quant_investor.market import macro_mart
from quant_investor.macro.store import pointer_sha256
from tests.helpers.macro_fixture import write_ready_macro_observations


TARGET = "20260714"
LANDED_AT = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)
AFTER_DEADLINE = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
NBS_CN_PMI_URL = (
    "https://www.stats.gov.cn/test/202605/t20260531_0000001.html"
)
NBS_CN_PMI_BODY = """<!doctype html>
<html><head>
<meta name="ArticleTitle" content="2026年5月中国采购经理指数运行情况">
<meta name="PubDate" content="2026/05/31 09:30">
</head><body>
<p>5月份，中国制造业采购经理指数（PMI）为49.5%</p>
</body></html>""".encode("utf-8")


def _official_capture() -> NbsPmiCapture:
    parsed = parse_nbs_cn_pmi_html(
        NBS_CN_PMI_BODY,
        source_url=NBS_CN_PMI_URL,
    )
    captured_at = LANDED_AT.replace(microsecond=0).isoformat()
    return NbsPmiCapture(
        month=parsed.month,
        value=parsed.value,
        source_url=parsed.source_url,
        source_record_id=parsed.source_record_id,
        article_title=parsed.article_title,
        source_release_at=parsed.source_release_at,
        fetch_started_at=captured_at,
        fetch_completed_at=captured_at,
        content_type="text/html",
        charset="utf-8",
        body_bytes=NBS_CN_PMI_BODY,
        body_sha256=parsed.body_sha256,
        body_size_bytes=parsed.body_size_bytes,
        parser_version=parsed.parser_version,
        parser_contract_sha256=parsed.parser_contract_sha256,
        redirect_chain=(NBS_CN_PMI_URL,),
    )


def _fetch_official(url: str) -> NbsPmiCapture:
    assert url == NBS_CN_PMI_URL
    return _official_capture()


class _MonthlyProvider:
    @staticmethod
    def _frame(fields: tuple[str, ...]) -> pd.DataFrame:
        months = pd.period_range(
            end=pd.Period("202605", freq="M"),
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


def _market_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-14", periods=120)
    rows: list[dict[str, object]] = []
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
    return pd.DataFrame(rows)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _observation_args(macro_root: Path) -> dict[str, object]:
    root = macro_root.parent / "macro_observations"
    return {
        "macro_observations_root": root,
        "expected_macro_observations_pointer_sha256": pointer_sha256(root),
    }


def _land_retry_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    market_root = tmp_path / "parquet" / "cn"
    macro_root = market_root / "macro_daily"
    bars_root = market_root / "bars"
    macro_root.mkdir(parents=True)
    bars_root.mkdir()
    market = _market_frame()
    bar_path = bars_root / "part.parquet"
    market.to_parquet(bar_path, index=False)
    evidence = [
        {
            "path": "part.parquet",
            "size_bytes": bar_path.stat().st_size,
            "sha256": _sha(bar_path),
        }
    ]
    evidence_sha = macro_mart._canonical_json_sha256({"files": evidence})
    observation_root = market_root / "macro_observations"
    observation_pointer_sha = write_ready_macro_observations(
        observation_root,
        as_of="2026-07-14",
    )
    macro_snapshot, observation_generation = (
        macro_mart._load_v15_macro_snapshot(
            observations_root=observation_root,
            expected_pointer_sha256=observation_pointer_sha,
            as_of=TARGET,
        )
    )

    monkeypatch.setattr(macro_mart, "_utc_now", lambda: LANDED_AT)
    monkeypatch.setattr(macro_mart, "fetch_nbs_cn_pmi", _fetch_official)
    provider_fetch = macro_mart._fetch_provider_bundle(
        client=_MonthlyProvider(),
        trade_date=TARGET,
        captured_at=LANDED_AT,
        nbs_cn_pmi_url=NBS_CN_PMI_URL,
        nbs_fetcher=_fetch_official,
    )
    provider_bundle = provider_fetch.bundle
    assert provider_bundle["selected_inputs"]["cn_pmi"]["month"] == "202605"
    assert (
        provider_bundle["selected_inputs"]["cn_pmi"][
            "expected_latest_month_lower_bound"
        ]
        == "202605"
    )
    frame, formula_universe, v15_controls = (
        macro_mart._derive_v15_macro_frame(
            market,
            trade_date=TARGET,
            provider_bundle=provider_bundle,
            macro_snapshot=macro_snapshot,
            observation_generation=observation_generation,
        )
    )

    pointer_path = market_root / "_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "snapshot_id": "retry-freshness",
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
    macro_mart._write_primary_generation(
        root=macro_root,
        run_id="landed-before-deadline",
        frame=frame,
        provider_bundle=provider_bundle,
        provider_captures=provider_fetch.captures,
        market_pointer_sha256=_sha(pointer_path),
        market_input_evidence=evidence,
        market_input_files_sha256=evidence_sha,
        market_formula_universe=formula_universe,
        macro_snapshot=macro_snapshot,
        v15_controls=v15_controls,
        observation_generation=observation_generation,
    )

    legacy_table = macro_root / "part.parquet"
    pd.DataFrame([{"trade_date": "2026-07-13"}]).to_parquet(
        legacy_table,
        index=False,
    )
    catalog_path = market_root / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.LEGACY_CATALOG_SCHEMA,
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "path": str(legacy_table),
                        "table_root": str(macro_root),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generation = macro_root / "_generations" / "landed-before-deadline"
    return macro_root, catalog_path, pointer_path, generation


def test_retry_after_month_deadline_requires_new_run_id_and_keeps_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, generation = (
        _land_retry_generation(tmp_path, monkeypatch)
    )
    catalog_before = catalog_path.read_bytes()
    generation_before = _tree_hashes(generation)
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: AFTER_DEADLINE)

    def _provider_must_not_run() -> object:
        raise AssertionError("stale landed generation cannot masquerade as fresh")

    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        _provider_must_not_run,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_stale_new_run_id_required:cn_pmi",
    ):
        macro_mart.refresh_cn_macro_mart(
            market="CN",
            as_of=TARGET,
            data_root=macro_root,
            run_id="landed-before-deadline",
            expected_catalog_sha256=_sha(catalog_path),
            expected_market_pointer_sha256=_sha(pointer_path),
            nbs_cn_pmi_url=NBS_CN_PMI_URL,
            allow_live=True,
            **_observation_args(macro_root),
        )

    assert catalog_path.read_bytes() == catalog_before
    assert _tree_hashes(generation) == generation_before
    assert not (macro_root / "_transactions").exists()


def test_stale_current_generation_is_not_already_current_after_month_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, generation = (
        _land_retry_generation(tmp_path, monkeypatch)
    )
    manifest_path = generation / "manifest.json"
    provider_path = generation / "provider_bundle.json"
    table_path = generation / "part.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.STRICT_CATALOG_SCHEMA,
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "path": table_path.relative_to(
                            macro_root.parent
                        ).as_posix(),
                        "table_root": generation.relative_to(
                            macro_root.parent
                        ).as_posix(),
                        "generation_manifest": manifest_path.relative_to(
                            macro_root.parent
                        ).as_posix(),
                        "provider_bundle": provider_path.relative_to(
                            macro_root.parent
                        ).as_posix(),
                        "generation_id": "landed-before-deadline",
                        "parquet_sha256": manifest["parquet_sha256"],
                        "sha256": manifest["parquet_sha256"],
                        "generation_manifest_sha256": _sha(manifest_path),
                        "provider_bundle_sha256": manifest[
                            "provider_bundle_sha256"
                        ],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_before = catalog_path.read_bytes()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: AFTER_DEADLINE)
    provider_called = False

    def _new_provider_required() -> object:
        nonlocal provider_called
        provider_called = True
        raise RuntimeError("new provider snapshot required")

    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        _new_provider_required,
    )
    with pytest.raises(RuntimeError, match="new provider snapshot required"):
        macro_mart.refresh_cn_macro_mart(
            market="CN",
            as_of=TARGET,
            data_root=macro_root,
            run_id="fresh-after-deadline",
            expected_catalog_sha256=_sha(catalog_path),
            expected_market_pointer_sha256=_sha(pointer_path),
            nbs_cn_pmi_url=NBS_CN_PMI_URL,
            allow_live=True,
            **_observation_args(macro_root),
        )

    assert provider_called is True
    assert catalog_path.read_bytes() == catalog_before


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
                "snapshot_id": "switch-freshness",
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
        as_of="2026-07-14",
    )
    return macro_root, catalog_path, pointer_path


def test_catalog_switch_rechecks_current_month_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path = _minimal_workspace(tmp_path)
    catalog_before = catalog_path.read_bytes()
    clocks = iter((*([LANDED_AT] * 6), AFTER_DEADLINE))
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clocks))
    monkeypatch.setattr(macro_mart, "_build_tushare_client", _MonthlyProvider)
    monkeypatch.setattr(macro_mart, "fetch_nbs_cn_pmi", _fetch_official)
    monkeypatch.setattr(
        macro_mart,
        "_load_market_inputs",
        lambda *_args, **_kwargs: (pd.DataFrame(), [], "0" * 64),
    )
    monkeypatch.setattr(
        macro_mart,
        "_derive_v15_macro_frame",
        lambda *_args, **_kwargs: (
            pd.DataFrame([{"trade_date": TARGET}]),
            {"placeholder": True},
            {"schema_version": "cn-macro-controls.v15.v1"},
        ),
    )
    monkeypatch.setattr(
        macro_mart,
        "_write_primary_generation",
        lambda **_kwargs: ({"generation_id": "crossed-deadline"}, object()),
    )
    monkeypatch.setattr(
        macro_mart,
        "_strict_catalog_payload",
        lambda **_kwargs: {},
    )

    def _unexpected_publish(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("stale bundle must not reach catalog publication")

    monkeypatch.setattr(
        macro_mart,
        "_publish_catalog_generation",
        _unexpected_publish,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_provider_stale_new_run_id_required:cn_pmi",
    ):
        macro_mart.refresh_cn_macro_mart(
            market="CN",
            as_of=TARGET,
            data_root=macro_root,
            run_id="crossed-deadline",
            expected_catalog_sha256=_sha(catalog_path),
            expected_market_pointer_sha256=_sha(pointer_path),
            nbs_cn_pmi_url=NBS_CN_PMI_URL,
            allow_live=True,
            **_observation_args(macro_root),
        )

    assert catalog_path.read_bytes() == catalog_before
    assert not (macro_root / "_transactions").exists()
