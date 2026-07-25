from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
from quant_investor.macro.release_calendar import (
    CriticalEventGapEvaluation,
    IssuerCoverage,
    ReleaseCalendarCASMismatch,
    ReleaseCalendarEvidence,
    ReleaseCalendarGenerationProof,
    ReleaseCalendarIdentity,
    ReleaseReadinessEvaluation,
    SessionLagEvaluation,
)
from quant_investor.market import macro_mart
from quant_investor.market.branch_readiness import load_macro_record
from quant_investor.market.dag.context import (
    _validated_pinned_macro_controls,
)
from quant_investor.macro.store import pointer_sha256
from tests.helpers.macro_fixture import (
    bind_macro_generation,
    write_ready_macro_observations,
)


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


def _patch_release_calendar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    root_name: str = "macro-release-calendar",
    ready_lag: int = 0,
) -> tuple[Path, Path, str, dict[str, str]]:
    root = tmp_path / root_name
    root.mkdir(mode=0o700)
    pointer_path = root / "_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "macro-release-calendar-pointer.v1",
                "generation_id": "release-calendar-20240510",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pointer_sha = _sha(pointer_path)
    proof = ReleaseCalendarGenerationProof(
        generation_id="release-calendar-20240510",
        pointer_sha256=pointer_sha,
        manifest_sha256="2" * 64,
        semantic_sha256="3" * 64,
        registry_sha256="4" * 64,
        critical_policy_sha256="5" * 64,
        plan_sha256="6" * 64,
        capture_manifest_sha256="7" * 64,
        market_open_days_sha256="8" * 64,
    )
    evidence = ReleaseCalendarEvidence(
        identity=ReleaseCalendarIdentity(
            pointer_path=str(pointer_path),
            pointer_sha256=pointer_sha,
            generation_id=proof.generation_id,
            generation_path=str(root / "_generations" / proof.generation_id),
            manifest_sha256=proof.manifest_sha256,
            semantic_sha256=proof.semantic_sha256,
            parent_generation_id="",
            parent_pointer_sha256="",
            parent_manifest_sha256="",
            parent_semantic_sha256="",
        ),
        registry_version="fixture",
        registry_sha256=proof.registry_sha256,
        critical_policy_version="fixture",
        critical_policy_sha256=proof.critical_policy_sha256,
        plan_sha256=proof.plan_sha256,
        capture_manifest_sha256=proof.capture_manifest_sha256,
        market_open_days_sha256=proof.market_open_days_sha256,
        captured_at=CAPTURED_AT.isoformat(),
        open_dates=("2024-05-08", "2024-05-09", "2024-05-10"),
        issuer_coverage=(
            IssuerCoverage(
                issuer="nbs_official",
                through_at=(CAPTURED_AT + timedelta(days=1)).isoformat(),
                source_ids=(),
            ),
            IssuerCoverage(
                issuer="pbc_official",
                through_at=(CAPTURED_AT + timedelta(days=1)).isoformat(),
                source_ids=(),
            ),
        ),
        source_artifacts=(),
        events=(),
        resolutions=(),
        validated_ancestry=(proof,),
    )

    def _load_calendar(
        *,
        canonical_root: str | Path,
        expected_pointer_sha256: str,
    ) -> ReleaseCalendarEvidence:
        assert Path(canonical_root).resolve(strict=True) == root.resolve(
            strict=True
        )
        if (
            expected_pointer_sha256 != pointer_sha
            or _sha(pointer_path) != pointer_sha
        ):
            raise ReleaseCalendarCASMismatch(
                "release_calendar_pointer_cas_mismatch"
            )
        return evidence

    monkeypatch.setattr(macro_mart, "load_release_calendar", _load_calendar)

    def _ready_evaluation(
        _evidence: ReleaseCalendarEvidence,
        *,
        macro_logical_date: str,
        target_session_date: str,
        decision_cutoff_at: str,
    ) -> ReleaseReadinessEvaluation:
        macro_date = datetime.strptime(
            macro_logical_date,
            "%Y%m%d",
        ).date()
        target_date = datetime.strptime(
            target_session_date,
            "%Y%m%d",
        ).date()
        return ReleaseReadinessEvaluation(
            ready=True,
            session_lag=SessionLagEvaluation(
                ready=True,
                session_lag=ready_lag,
                macro_logical_date=macro_date.isoformat(),
                target_session_date=target_date.isoformat(),
                blockers=(),
            ),
            critical_event_gap=CriticalEventGapEvaluation(
                ready=True,
                window_start_exclusive=(
                    f"{macro_date.isoformat()}T07:00:00+00:00"
                ),
                window_end_inclusive=decision_cutoff_at,
                relevant_event_ids=(),
                resolved_event_ids=(),
                blocking_event_ids=(),
                blockers=(),
            ),
            blockers=(),
        )

    monkeypatch.setattr(
        macro_mart,
        "evaluate_release_readiness",
        _ready_evaluation,
    )
    binding = {
        "macro_release_calendar_generation_id": proof.generation_id,
        "pointer_sha256": proof.pointer_sha256,
        "manifest_sha256": proof.manifest_sha256,
        "semantic_sha256": proof.semantic_sha256,
        "registry_sha256": proof.registry_sha256,
        "plan_sha256": proof.plan_sha256,
        "capture_manifest_sha256": proof.capture_manifest_sha256,
        "market_open_days_sha256": proof.market_open_days_sha256,
        "critical_policy_sha256": proof.critical_policy_sha256,
    }
    return root, pointer_path, pointer_sha, binding


def _workspace(
    tmp_path: Path,
    *,
    macro_as_of: str = TARGET,
    macro_decision_cutoff_at: str | None = None,
) -> tuple[Path, Path, Path, pd.DataFrame]:
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
    write_ready_macro_observations(
        market_root / "macro_observations",
        as_of=macro_as_of,
        decision_cutoff_at=macro_decision_cutoff_at,
    )
    return macro_root, catalog_path, pointer_path, bars


def test_release_decision_cutoff_is_session_bound_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, pointer_sha, _ = _patch_release_calendar(
        tmp_path,
        monkeypatch,
    )
    evidence = macro_mart.load_release_calendar(
        canonical_root=root,
        expected_pointer_sha256=pointer_sha,
    )

    assert macro_mart._macro_release_decision_cutoff(
        evidence,
        target_session_date=TARGET,
        provider_capture_cutoff_at=CAPTURED_AT.isoformat(),
    ) == "2024-05-10T07:00:00+00:00"

    same_day = replace(
        evidence,
        issuer_coverage=tuple(
            replace(item, through_at=CAPTURED_AT.isoformat())
            for item in evidence.issuer_coverage
        ),
    )
    assert macro_mart._macro_release_decision_cutoff(
        same_day,
        target_session_date=TARGET,
        provider_capture_cutoff_at=CAPTURED_AT.isoformat(),
    ) == CAPTURED_AT.isoformat()

    before_target = replace(
        evidence,
        issuer_coverage=tuple(
            replace(
                item,
                through_at=(CAPTURED_AT - timedelta(days=1)).isoformat(),
            )
            for item in evidence.issuer_coverage
        ),
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_release_calendar_coverage_before_target",
    ):
        macro_mart._macro_release_decision_cutoff(
            before_target,
            target_session_date=TARGET,
            provider_capture_cutoff_at=CAPTURED_AT.isoformat(),
        )

    before_close = replace(
        evidence,
        issuer_coverage=tuple(
            replace(item, through_at="2024-05-10T06:59:59+00:00")
            for item in evidence.issuer_coverage
        ),
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_release_calendar_coverage_before_market_close",
    ):
        macro_mart._macro_release_decision_cutoff(
            before_close,
            target_session_date=TARGET,
            provider_capture_cutoff_at=CAPTURED_AT.isoformat(),
        )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_release_calendar_coverage_in_future",
    ):
        macro_mart._macro_release_decision_cutoff(
            same_day,
            target_session_date=TARGET,
            provider_capture_cutoff_at="2024-05-10T08:29:59+00:00",
        )


def _valid_release_readiness_evidence() -> dict[str, Any]:
    return {
        "schema_version": "macro-release-readiness-evaluation.v1",
        "macro_logical_date": TARGET,
        "target_session_date": TARGET,
        "decision_cutoff_at": "2024-05-10T07:00:00+00:00",
        "evaluation": {
            "ready": True,
            "session_lag": {
                "ready": True,
                "session_lag": 0,
                "macro_logical_date": "2024-05-10",
                "target_session_date": "2024-05-10",
                "blockers": [],
            },
            "critical_event_gap": {
                "ready": True,
                "window_start_exclusive": (
                    "2024-05-10T07:00:00+00:00"
                ),
                "window_end_inclusive": "2024-05-10T07:00:00+00:00",
                "relevant_event_ids": [],
                "resolved_event_ids": [],
                "blocking_event_ids": [],
                "blockers": [],
            },
            "blockers": [],
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "lag_out_of_range",
        "hidden_blocking_event",
        "nested_date_mismatch",
        "window_mismatch",
        "noncanonical_cutoff",
        "unclassified_relevant_event",
        "resolved_relevant_event",
        "positive_lag_relevant_event",
        "nested_semantic_sha",
        "unsafe_event_id",
    ],
)
def test_release_readiness_evidence_rejects_nested_tampering(
    mutation: str,
) -> None:
    payload = _valid_release_readiness_evidence()
    evaluation = payload["evaluation"]
    lag = evaluation["session_lag"]
    gap = evaluation["critical_event_gap"]
    if mutation == "lag_out_of_range":
        lag["session_lag"] = 999
    elif mutation == "hidden_blocking_event":
        gap["ready"] = False
        gap["relevant_event_ids"] = ["event-1"]
        gap["blocking_event_ids"] = ["event-1"]
        gap["blockers"] = ["hidden"]
    elif mutation == "nested_date_mismatch":
        lag["macro_logical_date"] = "2024-05-09"
    elif mutation == "window_mismatch":
        gap["window_start_exclusive"] = "2024-05-09T07:00:00+00:00"
    elif mutation == "noncanonical_cutoff":
        payload["decision_cutoff_at"] = "2024-05-10T15:00:00+08:00"
        gap["window_end_inclusive"] = "2024-05-10T15:00:00+08:00"
    elif mutation == "unclassified_relevant_event":
        gap["relevant_event_ids"] = ["event-1"]
    elif mutation == "resolved_relevant_event":
        gap["relevant_event_ids"] = ["event-1"]
        gap["resolved_event_ids"] = ["event-1"]
    elif mutation == "positive_lag_relevant_event":
        payload["macro_logical_date"] = "20240509"
        lag["session_lag"] = 1
        lag["macro_logical_date"] = "2024-05-09"
        gap["window_start_exclusive"] = "2024-05-09T07:00:00+00:00"
        gap["relevant_event_ids"] = ["event-1"]
        gap["resolved_event_ids"] = ["event-1"]
    elif mutation == "nested_semantic_sha":
        gap["semantic_sha256"] = "0" * 64
    elif mutation == "unsafe_event_id":
        gap["relevant_event_ids"] = [" event-1"]
        gap["resolved_event_ids"] = [" event-1"]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_release_readiness_evidence_invalid",
    ):
        macro_mart._validate_macro_release_readiness_evidence(payload)


def _refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    macro_root: Path,
    catalog_path: Path,
    pointer_path: Path,
    *,
    run_id: str = "macro-refresh-fixture",
) -> dict[str, object]:
    release_root, _, release_pointer_sha, _ = _patch_release_calendar(
        tmp_path,
        monkeypatch,
        root_name=f"macro-release-calendar-{run_id}",
    )
    stage = macro_mart.stage_cn_macro_authoritative_refresh(
        market="CN",
        as_of=TARGET,
        canonical_root=macro_root,
        staging_root=tmp_path / f"macro-stage-{run_id}",
        run_id=run_id,
        expected_catalog_sha256=_sha(catalog_path),
        expected_market_pointer_sha256=_sha(pointer_path),
        macro_observations_root=macro_root.parent / "macro_observations",
        expected_macro_observations_pointer_sha256=pointer_sha256(
            macro_root.parent / "macro_observations"
        ),
        macro_release_calendar_root=release_root,
        expected_macro_release_calendar_pointer_sha256=(
            release_pointer_sha
        ),
        allow_live=True,
        nbs_cn_pmi_url=NBS_URL,
    )
    return macro_mart.promote_staged_macro_generation(
        staging_root=stage["staging_root"],
        canonical_root=macro_root,
        expected_catalog_sha256=_sha(catalog_path),
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

    result = _refresh(
        tmp_path,
        monkeypatch,
        macro_root,
        catalog_path,
        pointer_path,
    )

    assert result["status"] == "promoted"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["schema_version"] == macro_mart.STRICT_CATALOG_SCHEMA
    assert catalog["tables"]["daily_basic"]["path"] == (
        "daily_basic/part.parquet"
    )
    frame, manifest = macro_mart.read_macro_mart(data_root=macro_root)
    assert manifest["transform_version"] == macro_mart.V15_TRANSFORM_VERSION
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
    assert frame.iloc[0]["macro_score"] == pytest.approx(0.0)
    assert frame.iloc[0]["liquidity_score"] == pytest.approx(0.0)
    assert frame.iloc[0]["macro_score"] != pytest.approx(expected_macro)
    assert frame.iloc[0]["liquidity_score"] != pytest.approx(expected_breadth)
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


def test_v15_stage_accepts_readiness_proven_observation_lag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _bars = _workspace(
        tmp_path,
        macro_as_of="20240509",
        macro_decision_cutoff_at="2024-05-10T07:00:00+00:00",
    )
    release_root, _release_pointer_path, release_pointer_sha, _binding = (
        _patch_release_calendar(
            tmp_path,
            monkeypatch,
            root_name="macro-release-calendar-lagged-observations",
            ready_lag=1,
        )
    )
    clock_calls = iter(
        CAPTURED_AT + timedelta(seconds=index) for index in range(200)
    )
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clock_calls))
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)

    stage = macro_mart.stage_cn_macro_authoritative_refresh(
        market="CN",
        as_of=TARGET,
        canonical_root=macro_root,
        staging_root=tmp_path / "macro-stage-lagged-observations",
        run_id="v15-stage-lagged-observations",
        expected_catalog_sha256=_sha(catalog_path),
        expected_market_pointer_sha256=_sha(pointer_path),
        macro_observations_root=macro_root.parent / "macro_observations",
        expected_macro_observations_pointer_sha256=pointer_sha256(
            macro_root.parent / "macro_observations"
        ),
        macro_release_calendar_root=release_root,
        expected_macro_release_calendar_pointer_sha256=release_pointer_sha,
        allow_live=True,
        nbs_cn_pmi_url=NBS_URL,
    )

    assert stage["row"]["trade_date"] == "2024-05-10"
    snapshot = json.loads(
        Path(stage["manifest"]["resolved_macro_snapshot"]).read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["as_of"] == "20240509"
    readiness = stage["manifest"]["macro_release_readiness_evidence"]
    assert readiness["macro_logical_date"] == "20240509"
    assert readiness["target_session_date"] == "20240510"
    assert readiness["evaluation"]["session_lag"]["session_lag"] == 1

    promoted = macro_mart.promote_staged_macro_generation(
        staging_root=stage["staging_root"],
        canonical_root=macro_root,
        expected_catalog_sha256=_sha(catalog_path),
    )
    assert promoted["status"] == "promoted"


def test_observation_pointer_change_during_provider_fetch_blocks_catalog_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _bars = _workspace(tmp_path)
    catalog_before = catalog_path.read_bytes()
    observations_root = macro_root.parent / "macro_observations"
    observation_pointer_path = observations_root / "_latest.json"
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
    original_fetch = macro_mart._fetch_provider_bundle

    def _fetch_then_advance_pointer(**kwargs: Any):
        result = original_fetch(**kwargs)
        advanced_pointer = json.loads(
            observation_pointer_path.read_text(encoding="utf-8")
        )
        advanced_pointer["generation_id"] = (
            "macro-observations-provider-race-g2"
        )
        observation_pointer_path.write_text(
            json.dumps(
                advanced_pointer,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        macro_mart,
        "_fetch_provider_bundle",
        _fetch_then_advance_pointer,
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_observation_pointer_cas_mismatch",
    ):
        _refresh(
            tmp_path,
            monkeypatch,
            macro_root,
            catalog_path,
            pointer_path,
            run_id="macro-provider-pointer-race",
        )

    assert catalog_path.read_bytes() == catalog_before


def test_v15_stage_promote_reader_and_dag_controls_are_one_hash_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _bars = _workspace(tmp_path)
    before_catalog = catalog_path.read_bytes()
    clock_calls = iter(
        CAPTURED_AT + timedelta(seconds=index) for index in range(200)
    )
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: next(clock_calls))
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)
    (
        release_calendar_root,
        release_calendar_pointer_path,
        release_calendar_pointer_sha,
        release_calendar_binding,
    ) = _patch_release_calendar(tmp_path, monkeypatch)
    observations_root = macro_root.parent / "macro_observations"
    stage = macro_mart.stage_cn_macro_authoritative_refresh(
        market="CN",
        as_of=TARGET,
        canonical_root=macro_root,
        staging_root=tmp_path / "macro-stage",
        run_id="v15-stage-promote",
        expected_catalog_sha256=_sha(catalog_path),
        expected_market_pointer_sha256=_sha(pointer_path),
        macro_observations_root=observations_root,
        expected_macro_observations_pointer_sha256=pointer_sha256(
            observations_root
        ),
        macro_release_calendar_root=release_calendar_root,
        expected_macro_release_calendar_pointer_sha256=(
            release_calendar_pointer_sha
        ),
        allow_live=True,
        nbs_cn_pmi_url=NBS_URL,
    )

    assert stage["status"] == "staged"
    assert stage["promoted"] is False
    assert catalog_path.read_bytes() == before_catalog
    receipt_path = Path(str(stage["staging_receipt"]))
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["macro_observations_root"] == str(
        observations_root.resolve(strict=True)
    )
    assert receipt["macro_release_calendar_root"] == str(
        release_calendar_root.resolve(strict=True)
    )
    assert receipt["macro_release_calendar_generation"] == (
        release_calendar_binding
    )
    assert stage["manifest"]["macro_release_calendar_generation"] == (
        release_calendar_binding
    )
    release_calendar_flat_binding = (
        macro_mart._macro_release_calendar_flat_binding(
            release_calendar_binding
        )
    )
    for field_name, value in release_calendar_flat_binding.items():
        assert receipt[field_name] == value
        assert stage["manifest"][field_name] == value
    assert "macro_readiness_evidence_semantic_sha256" not in (
        stage["manifest"]
    )
    release_readiness_evidence = stage["manifest"][
        "macro_release_readiness_evidence"
    ]
    assert release_readiness_evidence["decision_cutoff_at"] == (
        "2024-05-10T07:00:00+00:00"
    )
    assert stage["manifest"]["macro_release_decision_cutoff_at"] == (
        release_readiness_evidence["decision_cutoff_at"]
    )
    assert receipt["macro_release_readiness_evidence"] == (
        release_readiness_evidence
    )
    provider_bundle = json.loads(
        Path(
            str(stage["manifest"]["resolved_provider_bundle"])
        ).read_text(encoding="utf-8")
    )
    assert provider_bundle["decision_cutoff_at"] != (
        release_readiness_evidence["decision_cutoff_at"]
    )
    controls_path = Path(
        str(stage["manifest"]["resolved_v15_controls"])
    )
    assert _sha(controls_path) == stage["manifest"]["v15_controls_sha256"]
    assert stage["row"]["macro_score"] == pytest.approx(0.0)
    assert stage["row"]["liquidity_score"] == pytest.approx(0.0)

    untrusted_receipt = dict(receipt)
    untrusted_receipt["macro_observations_root"] = str(tmp_path)
    receipt_path.write_text(
        json.dumps(untrusted_receipt, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_observations_root_untrusted",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    receipt_path.write_bytes(receipt_bytes)

    tampered_receipt = json.loads(receipt_bytes.decode("utf-8"))
    tampered_receipt["macro_release_calendar_generation"][
        "semantic_sha256"
    ] = "9" * 64
    receipt_path.write_text(
        json.dumps(tampered_receipt, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_staging_release_calendar_binding_invalid",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    receipt_path.write_bytes(receipt_bytes)

    release_calendar_pointer_before = (
        release_calendar_pointer_path.read_bytes()
    )
    advanced_release_pointer = json.loads(
        release_calendar_pointer_before.decode("utf-8")
    )
    advanced_release_pointer["generation_id"] = (
        "release-calendar-20240510-race"
    )
    release_calendar_pointer_path.write_text(
        json.dumps(advanced_release_pointer, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="release_calendar_pointer_cas_mismatch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    assert release_calendar_pointer_path.read_bytes() != (
        release_calendar_pointer_before
    )
    release_calendar_pointer_path.write_bytes(
        release_calendar_pointer_before
    )

    observation_pointer_path = observations_root / "_latest.json"
    observation_pointer_before = observation_pointer_path.read_bytes()
    advanced_pointer = json.loads(
        observation_pointer_before.decode("utf-8")
    )
    advanced_pointer["generation_id"] = "macro-observations-race-g2"
    observation_pointer_path.write_text(
        json.dumps(advanced_pointer, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_observation_pointer_cas_mismatch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    assert not (
        macro_root / "_generations" / "v15-stage-promote"
    ).exists()
    observation_pointer_path.write_bytes(observation_pointer_before)

    original_atomic_write = macro_mart._atomic_write_bytes
    concurrent_release_pointer = json.dumps(
        {
            "schema_version": "macro-release-calendar-pointer.v1",
            "generation_id": "release-calendar-concurrent-advance",
        },
        sort_keys=True,
    ).encode("utf-8")

    def _write_catalog_then_advance_release_pointer(
        path: Path,
        payload: bytes,
    ) -> None:
        original_atomic_write(path, payload)
        if path == catalog_path and payload != before_catalog:
            release_calendar_pointer_path.write_bytes(
                concurrent_release_pointer
            )

    monkeypatch.setattr(
        macro_mart,
        "_atomic_write_bytes",
        _write_catalog_then_advance_release_pointer,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_release_calendar_pointer_changed_during_switch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    assert release_calendar_pointer_path.read_bytes() == (
        concurrent_release_pointer
    )
    monkeypatch.setattr(
        macro_mart,
        "_atomic_write_bytes",
        original_atomic_write,
    )
    release_calendar_pointer_path.write_bytes(
        release_calendar_pointer_before
    )
    concurrent_observation_pointer = json.dumps(
        {
            "schema_version": "macro-observation-pointer.v2",
            "generation_id": "macro-observations-concurrent-advance",
        },
        sort_keys=True,
    ).encode("utf-8")

    def _write_catalog_then_advance_observation_pointer(
        path: Path,
        payload: bytes,
    ) -> None:
        original_atomic_write(path, payload)
        if path == catalog_path and payload != before_catalog:
            observation_pointer_path.write_bytes(
                concurrent_observation_pointer
            )

    monkeypatch.setattr(
        macro_mart,
        "_atomic_write_bytes",
        _write_catalog_then_advance_observation_pointer,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_observation_pointer_changed_during_switch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=stage["staging_root"],
            canonical_root=macro_root,
            expected_catalog_sha256=_sha(catalog_path),
        )
    assert catalog_path.read_bytes() == before_catalog
    assert observation_pointer_path.read_bytes() == (
        concurrent_observation_pointer
    )
    monkeypatch.setattr(
        macro_mart,
        "_atomic_write_bytes",
        original_atomic_write,
    )
    observation_pointer_path.write_bytes(observation_pointer_before)
    promoted = macro_mart.promote_staged_macro_generation(
        staging_root=stage["staging_root"],
        canonical_root=macro_root,
        expected_catalog_sha256=_sha(catalog_path),
    )

    assert promoted["status"] == "promoted"
    assert promoted["orphan_generation_reused"] is True
    frame, manifest = macro_mart.read_macro_mart(data_root=macro_root)
    record, loaded_manifest = load_macro_record(
        as_of=TARGET,
        root=macro_root,
    )
    dag_controls = _validated_pinned_macro_controls(loaded_manifest)
    assert frame.iloc[0]["macro_score"] == dag_controls["macro_score"]
    assert record["liquidity_score"] == dag_controls["liquidity_score"]
    assert manifest["v15_controls_sha256"] == _sha(
        Path(str(manifest["resolved_v15_controls"]))
    )
    assert dag_controls["observation_generation"]["pointer_sha256"] == (
        pointer_sha256(observations_root)
    )
    assert manifest["macro_release_calendar_generation"] == (
        release_calendar_binding
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["tables"]["macro_daily"][
        "macro_release_calendar_generation"
    ] == release_calendar_binding
    journal = json.loads(
        Path(promoted["transaction_journal"]).read_text(encoding="utf-8")
    )
    assert journal["macro_release_calendar_generation"] == (
        release_calendar_binding
    )
    for field_name, value in release_calendar_flat_binding.items():
        assert catalog["tables"]["macro_daily"][field_name] == value
        assert journal[field_name] == value
    assert catalog["tables"]["macro_daily"][
        "macro_release_readiness_evidence"
    ] == release_readiness_evidence
    assert journal["macro_release_readiness_evidence"] == (
        release_readiness_evidence
    )
    promoted_catalog_bytes = catalog_path.read_bytes()
    tampered_catalog = json.loads(promoted_catalog_bytes.decode("utf-8"))
    tampered_catalog["tables"]["macro_daily"][
        "macro_release_calendar_generation"
    ]["manifest_sha256"] = "a" * 64
    catalog_path.write_text(
        json.dumps(tampered_catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_catalog_release_calendar_binding_mismatch",
    ):
        macro_mart.read_macro_mart(data_root=macro_root)
    catalog_path.write_bytes(promoted_catalog_bytes)


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
            macro_observations_root=macro_root.parent / "macro_observations",
            expected_macro_observations_pointer_sha256=pointer_sha256(
                macro_root.parent / "macro_observations"
            ),
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
        _refresh(
            tmp_path,
            monkeypatch,
            macro_root,
            catalog_path,
            pointer_path,
        )
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
        macro_observation_pointer_sha256="2" * 64,
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
    original = macro_mart._copy_or_reuse_staged_generation

    def _copy_then_aba(**kwargs: Any):
        result = original(**kwargs)
        macro_mart._atomic_write_bytes(pointer_path, pointer_bytes)
        return result

    monkeypatch.setattr(
        macro_mart,
        "_copy_or_reuse_staged_generation",
        _copy_then_aba,
    )
    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_market_pointer_cas_mismatch",
    ):
        _refresh(
            tmp_path,
            monkeypatch,
            macro_root,
            catalog_path,
            pointer_path,
        )
    assert catalog_path.read_bytes() == before_catalog


def test_authoritative_recovery_commits_valid_switched_journal(
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
    first = _refresh(
        tmp_path,
        monkeypatch,
        macro_root,
        catalog_path,
        pointer_path,
    )
    journal_path = Path(str(first["transaction_journal"]))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    journal_path.write_text(json.dumps(journal, sort_keys=True), encoding="utf-8")

    release_root = Path(str(first["macro_release_calendar_root"]))
    observations_root = macro_root.parent / "macro_observations"
    assert journal["macro_observations_root"] == str(observations_root)
    assert journal[
        "expected_macro_observations_pointer_sha256"
    ] == pointer_sha256(observations_root)
    with (
        macro_mart._market_writer_lock(macro_root.parent),
        macro_mart._catalog_writer_lock(macro_root.parent),
        macro_mart._macro_observation_writer_lock(observations_root),
        macro_mart._macro_release_calendar_writer_lock(release_root),
    ):
        macro_mart._recover_catalog_transactions(
            root=macro_root,
            catalog_path=catalog_path,
            locked_macro_observations_root=observations_root,
            locked_macro_release_calendar_root=release_root,
        )
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "committed"


def test_promotion_recovers_catalog_before_release_calendar_stage_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    old_catalog = catalog_path.read_bytes()
    old_catalog_sha = hashlib.sha256(old_catalog).hexdigest()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)
    run_id = "release-drift-after-switch"
    result = _refresh(
        tmp_path,
        monkeypatch,
        macro_root,
        catalog_path,
        pointer_path,
        run_id=run_id,
    )
    journal_path = Path(str(result["transaction_journal"]))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    macro_mart._atomic_json(journal_path, journal)
    release_pointer = (
        Path(str(result["macro_release_calendar_root"])) / "_latest.json"
    )
    advanced_release_pointer = (
        b'{"generation_id":"advanced-release-calendar"}\n'
    )
    macro_mart._atomic_write_bytes(
        release_pointer,
        advanced_release_pointer,
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="release_calendar_pointer_cas_mismatch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=(
                tmp_path / f"macro-stage-{run_id}" / run_id
            ),
            canonical_root=macro_root,
            expected_catalog_sha256=old_catalog_sha,
        )

    assert catalog_path.read_bytes() == old_catalog
    assert release_pointer.read_bytes() == advanced_release_pointer
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "rolled_back"


def test_promotion_recovery_rolls_back_on_observation_pointer_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro_root, catalog_path, pointer_path, _ = _workspace(tmp_path)
    old_catalog = catalog_path.read_bytes()
    old_catalog_sha = hashlib.sha256(old_catalog).hexdigest()
    monkeypatch.setattr(macro_mart, "_utc_now", lambda: CAPTURED_AT)
    monkeypatch.setattr(
        macro_mart,
        "_build_tushare_client",
        lambda: _FakeTushare(),
    )
    _patch_official_fetch(monkeypatch)
    run_id = "observation-drift-after-switch"
    result = _refresh(
        tmp_path,
        monkeypatch,
        macro_root,
        catalog_path,
        pointer_path,
        run_id=run_id,
    )
    journal_path = Path(str(result["transaction_journal"]))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    macro_mart._atomic_json(journal_path, journal)
    observations_pointer = (
        macro_root.parent / "macro_observations" / "_latest.json"
    )
    advanced_observations_pointer = (
        b'{"generation_id":"advanced-observations"}\n'
    )
    macro_mart._atomic_write_bytes(
        observations_pointer,
        advanced_observations_pointer,
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_observation_pointer_cas_mismatch",
    ):
        macro_mart.promote_staged_macro_generation(
            staging_root=(
                tmp_path / f"macro-stage-{run_id}" / run_id
            ),
            canonical_root=macro_root,
            expected_catalog_sha256=old_catalog_sha,
        )

    assert catalog_path.read_bytes() == old_catalog
    assert observations_pointer.read_bytes() == (
        advanced_observations_pointer
    )
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "rolled_back"


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
    result = _refresh(
        tmp_path,
        monkeypatch,
        macro_root,
        catalog_path,
        pointer_path,
    )
    journal_path = Path(str(result["transaction_journal"]))
    transaction = journal_path.parent
    expected_old = (transaction / "old_catalog.json").read_bytes()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "switched"
    journal_path.write_text(json.dumps(journal, sort_keys=True), encoding="utf-8")
    manifest = result["manifest"]
    assert isinstance(manifest, dict)
    provider_path = Path(str(manifest["resolved_provider_bundle"]))
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

    release_root = Path(str(journal["macro_release_calendar_root"]))
    observations_root = macro_root.parent / "macro_observations"
    with (
        macro_mart._market_writer_lock(macro_root.parent),
        macro_mart._catalog_writer_lock(macro_root.parent),
        macro_mart._macro_observation_writer_lock(observations_root),
        macro_mart._macro_release_calendar_writer_lock(release_root),
    ):
        macro_mart._recover_catalog_transactions(
            root=macro_root,
            catalog_path=catalog_path,
            locked_macro_observations_root=observations_root,
            locked_macro_release_calendar_root=release_root,
        )

    assert catalog_path.read_bytes() == expected_old
    recovered = json.loads(journal_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "rolled_back"
