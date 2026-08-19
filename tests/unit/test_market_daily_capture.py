from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.market_daily_capture import (
    MarketDailyCaptureBlocked,
    build_private_market_candidate,
    capture_market_daily,
    publish_market_daily_capture,
    replay_market_daily_capture,
    shadow_market_daily_capture,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.pit_universe import (
    PITUniverseRecord,
    PITUniverseStore,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _authority(
    tmp_path: Path,
    target: str = "20260818",
    *,
    open_trade_dates: list[str] | None = None,
) -> tuple[Path, str]:
    payload = {
        "schema_version": "cn-close-session-receipt.v1",
        "status": "AUTHORIZED",
        "target_trade_date": target,
    }
    if open_trade_dates is not None:
        payload["open_trade_dates"] = open_trade_dates
    path = _write_json(
        tmp_path / "authority.json",
        payload,
    )
    return path, _sha256(path)


def _authority_from_raw_calendar(
    tmp_path: Path,
    *,
    target: str,
    open_trade_dates: list[str],
) -> tuple[Path, str]:
    raw_path = _write_json(
        tmp_path / "close-session.raw.json",
        {
            "code": 0,
            "data": {
                "count": len(open_trade_dates),
                "fields": [
                    "exchange",
                    "cal_date",
                    "is_open",
                    "pretrade_date",
                ],
                "has_more": False,
                "items": [["SSE", trade_date, 1, ""] for trade_date in open_trade_dates],
            },
            "detail": "",
            "msg": "",
            "request_id": "unit-request",
        },
    )
    authority_path = _write_json(
        tmp_path / "authority.json",
        {
            "schema_version": "cn-close-session-receipt.v1",
            "status": "TARGET_AUTHORIZED",
            "target_date": target,
            "raw_response_path": str(raw_path),
            "raw_response_sha256": _sha256(raw_path),
        },
    )
    return authority_path, _sha256(authority_path)


def _scope(tmp_path: Path) -> tuple[Path, str]:
    path = _write_json(
        tmp_path / "scope.json",
        {"full_a": ["000001.SZ", "000002.SZ"]},
    )
    return path, _sha256(path)


def _pit(tmp_path: Path) -> dict[str, str]:
    manifest = _write_json(tmp_path / "pit" / "manifest.json", {"ok": True})
    membership = _write_json(
        tmp_path / "pit" / "membership.json",
        {"records": ["000001.SZ", "000002.SZ"]},
    )
    return {
        "generation_id": "pit-unit",
        "generation_manifest_path": str(manifest),
        "generation_manifest_sha256": _sha256(manifest),
        "canonical_path": str(membership),
        "canonical_sha256": _sha256(membership),
    }


class _Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.daily_rows = [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260818",
                "open": 10.0,
                "high": 11.0,
                "low": 9.8,
                "close": 10.8,
                "pre_close": 10.0,
                "change": 0.8,
                "pct_chg": 8.0,
                "vol": 1000.0,
                "amount": 10000.0,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260818",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.2,
                "pre_close": 20.0,
                "change": 0.2,
                "pct_chg": 1.0,
                "vol": 2000.0,
                "amount": 20000.0,
            },
        ]
        self.basic_rows = [
            {
                "ts_code": symbol,
                "trade_date": "20260818",
                "turnover_rate": 1.0,
                "volume_ratio": 1.0,
                "pe": 10.0,
                "pb": 1.0,
                "total_mv": 100000.0,
                "circ_mv": 90000.0,
            }
            for symbol in ("000001.SZ", "000002.SZ")
        ]
        self.adj_rows = [
            {
                "ts_code": symbol,
                "trade_date": "20260818",
                "adj_factor": 1.0,
            }
            for symbol in ("000001.SZ", "000002.SZ")
        ]

    def daily(self, **_kwargs):
        self.calls.append("daily")
        return pd.DataFrame(self.daily_rows)

    def daily_basic(self, **_kwargs):
        self.calls.append("daily_basic")
        return pd.DataFrame(self.basic_rows)

    def adj_factor(self, **_kwargs):
        self.calls.append("adj_factor")
        return pd.DataFrame(self.adj_rows)


class _WindowProvider(_Provider):
    def _dated(self, rows, trade_date):
        return pd.DataFrame([{**row, "trade_date": trade_date} for row in rows])

    def daily(self, trade_date=None, **_kwargs):
        self.calls.append(f"daily:{trade_date}")
        return self._dated(self.daily_rows, trade_date)

    def daily_basic(self, trade_date=None, **_kwargs):
        self.calls.append(f"daily_basic:{trade_date}")
        return self._dated(self.basic_rows, trade_date)

    def adj_factor(self, trade_date=None, **_kwargs):
        self.calls.append(f"adj_factor:{trade_date}")
        return self._dated(self.adj_rows, trade_date)


class _Store:
    def __init__(self, *, latest_complete_trade_date: str = "") -> None:
        self.frame: pd.DataFrame | None = None
        self.kwargs: dict[str, object] = {}
        self.latest_complete_trade_date = latest_complete_trade_date

    def upsert_bars(self, frame: pd.DataFrame, **kwargs):
        self.frame = frame.copy()
        self.kwargs = dict(kwargs)
        return {
            "status": "OK",
            "snapshot_id": "candidate",
            "latest_complete_trade_date": kwargs["target_trade_date"],
        }

    def validate_latest(self):
        return {
            "status": "passed",
            "snapshot_id": "candidate",
            "latest_complete_trade_date": self.latest_complete_trade_date,
        }


def _capture_args(tmp_path: Path, provider: _Provider) -> dict[str, object]:
    authority, authority_sha = _authority(tmp_path)
    scope, scope_sha = _scope(tmp_path)
    return {
        "provider": provider,
        "capture_root": tmp_path / "capture",
        "target_authority_path": authority,
        "expected_target_authority_sha256": authority_sha,
        "scope_path": scope,
        "expected_scope_sha256": scope_sha,
        "pit_generation_binding": _pit(tmp_path),
        "expected_market_pointer_sha256": "a" * 64,
    }


def test_capture_replay_and_publish_never_refetch_provider(tmp_path):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)

    capture = capture_market_daily(**args)

    assert capture["status"] == "CAPTURED"
    assert provider.calls == ["daily", "daily_basic", "adj_factor"]
    manifest = json.loads(Path(capture["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["provider_accounting"] == {
        "calls": 3,
        "failed": 0,
        "malformed": 0,
        "has_more": False,
    }
    assert manifest["classification"]["counts"]["true_missing"] == 0
    assert manifest["endpoints"]["daily"]["row_count"] == 2

    store = _Store(latest_complete_trade_date="20260816")
    result = publish_market_daily_capture(
        capture_manifest_path=capture["manifest_path"],
        expected_capture_manifest_sha256=capture["manifest_sha256"],
        data_root=tmp_path / "unused-data-root",
        scope_path=args["scope_path"],
        expected_scope_sha256=args["expected_scope_sha256"],
        pit_generation_binding=args["pit_generation_binding"],
        expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
        store=store,
    )

    assert result["status"] == "PUBLISHED"
    assert result["provider_refetched"] is False
    assert provider.calls == ["daily", "daily_basic", "adj_factor"]
    assert set(store.frame["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert store.kwargs["expected_latest_pointer_sha256"] == "a" * 64


def test_two_session_catch_up_is_sealed_and_published_in_order(tmp_path):
    provider = _WindowProvider()
    args = _capture_args(tmp_path, provider)
    authority, authority_sha = _authority_from_raw_calendar(
        tmp_path,
        target="20260818",
        open_trade_dates=["20260816", "20260817", "20260818"],
    )
    args["target_authority_path"] = authority
    args["expected_target_authority_sha256"] = authority_sha
    args["target_trade_dates"] = ["20260817", "20260818"]
    args["parent_latest_complete_trade_date"] = "20260816"

    capture = capture_market_daily(**args)

    assert capture["target_trade_dates"] == ["20260817", "20260818"]
    assert provider.calls == [
        "daily:20260817",
        "daily_basic:20260817",
        "adj_factor:20260817",
        "daily:20260818",
        "daily_basic:20260818",
        "adj_factor:20260818",
    ]
    manifest = json.loads(Path(capture["manifest_path"]).read_text(encoding="utf-8"))
    assert [session["trade_date"] for session in manifest["sessions"]] == [
        "20260817",
        "20260818",
    ]
    mismatched_store = _Store(latest_complete_trade_date="20260815")
    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="publication_parent_trade_date_mismatch",
    ):
        publish_market_daily_capture(
            capture_manifest_path=capture["manifest_path"],
            expected_capture_manifest_sha256=capture["manifest_sha256"],
            data_root=tmp_path / "unused-mismatch",
            scope_path=args["scope_path"],
            expected_scope_sha256=args["expected_scope_sha256"],
            pit_generation_binding=args["pit_generation_binding"],
            expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
            store=mismatched_store,
        )
    assert mismatched_store.frame is None
    store = _Store(latest_complete_trade_date="20260816")
    result = publish_market_daily_capture(
        capture_manifest_path=capture["manifest_path"],
        expected_capture_manifest_sha256=capture["manifest_sha256"],
        data_root=tmp_path / "unused",
        scope_path=args["scope_path"],
        expected_scope_sha256=args["expected_scope_sha256"],
        pit_generation_binding=args["pit_generation_binding"],
        expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
        store=store,
    )
    assert result["target_trade_dates"] == ["20260817", "20260818"]
    assert store.kwargs["target_trade_date"] == "20260818"
    assert store.kwargs["target_trade_dates"] == [
        "20260817",
        "20260818",
    ]
    assert set(store.frame["trade_date"]) == {"20260817", "20260818"}


def test_catch_up_window_over_five_blocks_before_provider(tmp_path):
    provider = _WindowProvider()
    args = _capture_args(tmp_path, provider)
    authority, authority_sha = _authority(
        tmp_path,
        open_trade_dates=[
            "20260812",
            "20260813",
            "20260814",
            "20260815",
            "20260816",
            "20260817",
            "20260818",
        ],
    )
    args["target_authority_path"] = authority
    args["expected_target_authority_sha256"] = authority_sha
    args["target_trade_dates"] = [
        "20260813",
        "20260814",
        "20260815",
        "20260816",
        "20260817",
        "20260818",
    ]
    args["parent_latest_complete_trade_date"] = "20260812"

    with pytest.raises(MarketDailyCaptureBlocked, match="catch_up_window_too_large"):
        capture_market_daily(**args)

    assert provider.calls == []


def test_catch_up_missing_middle_session_blocks_before_provider(tmp_path):
    provider = _WindowProvider()
    args = _capture_args(tmp_path, provider)
    authority, authority_sha = _authority(
        tmp_path,
        open_trade_dates=["20260815", "20260816", "20260817", "20260818"],
    )
    args["target_authority_path"] = authority
    args["expected_target_authority_sha256"] = authority_sha
    args["target_trade_dates"] = ["20260816", "20260818"]
    args["parent_latest_complete_trade_date"] = "20260815"

    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="catch_up_window_not_contiguous",
    ):
        capture_market_daily(**args)

    assert provider.calls == []


def test_same_target_rebind_refreshes_pit_coverage_without_advancing_date(
    tmp_path,
):
    from tests.unit.test_market_data_parquet_direct_maintenance import (
        _write_pit_generation,
        _write_seed_snapshot,
    )

    _write_seed_snapshot(tmp_path)
    generation = _write_pit_generation(tmp_path)
    provider = _WindowProvider()
    evidence_root = tmp_path / "same-target-evidence"
    args = _capture_args(evidence_root, provider)
    authority, authority_sha = _authority(
        evidence_root,
        target="20260315",
        open_trade_dates=["20260315"],
    )
    args.update(
        {
            "target_authority_path": authority,
            "expected_target_authority_sha256": authority_sha,
            "pit_generation_binding": {
                "generation_id": generation["generation_id"],
                "generation_manifest_path": generation["generation_manifest_path"],
                "generation_manifest_sha256": generation["generation_manifest_sha256"],
                "canonical_path": generation["canonical_path"],
                "canonical_sha256": generation["canonical_sha256"],
            },
            "target_trade_dates": ["20260315"],
            "parent_latest_complete_trade_date": "20260315",
            "same_target_rebind": True,
        }
    )
    pointer_path = tmp_path / "parquet" / "cn" / "_latest.json"
    previous_pointer = pointer_path.read_bytes()
    args["expected_market_pointer_sha256"] = hashlib.sha256(previous_pointer).hexdigest()

    capture = capture_market_daily(**args)
    result = publish_market_daily_capture(
        capture_manifest_path=capture["manifest_path"],
        expected_capture_manifest_sha256=capture["manifest_sha256"],
        data_root=tmp_path,
        scope_path=args["scope_path"],
        expected_scope_sha256=args["expected_scope_sha256"],
        pit_generation_binding=args["pit_generation_binding"],
        expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
    )

    assert result["same_target_rebind"] is True
    assert result["target_trade_dates"] == ["20260315"]
    assert result["parquet_commit"]["latest_complete_trade_date"] == "20260315"
    assert pointer_path.read_bytes() != previous_pointer
    latest = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert latest["latest_complete_trade_date"] == "20260315"
    assert latest["coverage"]["pit_generation_id"] == generation["generation_id"]
    bound_pit = MarketDataReader(market="CN", data_root=tmp_path).coverage_bound_pit(refresh=True)
    assert bound_pit["generation_id"] == generation["generation_id"]


@pytest.mark.parametrize(
    ("window", "explicit_rebind", "blocker"),
    [
        ([], True, "catch_up_target_window_invalid"),
        (["20260818"], False, "catch_up_parent_trade_date_invalid"),
    ],
)
def test_same_target_rebind_never_treats_empty_or_implicit_window_as_success(
    tmp_path, window, explicit_rebind, blocker
):
    provider = _WindowProvider()
    args = _capture_args(tmp_path, provider)
    authority, authority_sha = _authority(
        tmp_path,
        target="20260818",
        open_trade_dates=["20260818"],
    )
    args.update(
        {
            "target_authority_path": authority,
            "expected_target_authority_sha256": authority_sha,
            "target_trade_dates": window,
            "parent_latest_complete_trade_date": "20260818",
            "same_target_rebind": explicit_rebind,
        }
    )

    with pytest.raises(MarketDailyCaptureBlocked, match=blocker):
        capture_market_daily(**args)

    assert provider.calls == []


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (lambda provider: provider.basic_rows.pop(), "daily_basic_keyset_missing:1"),
        (
            lambda provider: provider.basic_rows.append(
                {
                    **provider.basic_rows[0],
                    "ts_code": "000003.SZ",
                }
            ),
            "daily_basic_keyset_extra:1",
        ),
        (
            lambda provider: provider.adj_rows.append(dict(provider.adj_rows[0])),
            "adj_factor_duplicate_keys:1",
        ),
        (
            lambda provider: provider.daily_rows[0].update(trade_date="20260817"),
            "daily_wrong_trade_date:0:20260817",
        ),
    ],
)
def test_capture_hard_blocks_keyset_duplicate_and_wrong_date(tmp_path, mutation, blocker):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    mutation(provider)

    with pytest.raises(MarketDailyCaptureBlocked) as exc_info:
        capture_market_daily(**args)

    assert blocker in exc_info.value.blockers
    assert exc_info.value.receipt["status"] == "BLOCKED"
    assert Path(exc_info.value.receipt["receipt_path"]).is_file()


def test_capture_hard_blocks_endpoint_error_before_publication(tmp_path):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)

    def _failed(**_kwargs):
        raise RuntimeError("provider late token=SUPER_SECRET_VALUE")

    provider.daily_basic = _failed

    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="daily_basic_endpoint_error:RuntimeError",
    ) as exc_info:
        capture_market_daily(**args)

    assert "SUPER_SECRET_VALUE" not in str(exc_info.value)
    blocked_path = Path(exc_info.value.receipt["receipt_path"])
    assert b"SUPER_SECRET_VALUE" not in blocked_path.read_bytes()


def test_capture_accepts_disjoint_reason_coded_absence(tmp_path):
    provider = _Provider()
    provider.daily_rows.pop()
    provider.basic_rows.pop()
    provider.adj_rows.pop()
    args = _capture_args(tmp_path, provider)
    args["reason_sets"] = {"suspended": ["000002.SZ"]}

    capture = capture_market_daily(**args)

    classification = capture["classification"]
    assert classification["status"] == "PASSED"
    assert classification["counts"]["observed"] == 1
    assert classification["counts"]["suspended"] == 1
    assert classification["counts"]["true_missing"] == 0


def test_capture_requires_path_backed_nontrading_evidence(tmp_path):
    provider = _Provider()
    provider.daily_rows.pop()
    provider.basic_rows.pop()
    provider.adj_rows.pop()
    args = _capture_args(tmp_path, provider)
    args["reason_sets"] = {"non_trading": ["000002.SZ"]}

    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="non_trading_classification_evidence_required",
    ):
        capture_market_daily(**args)


def test_replay_rejects_endpoint_tamper(tmp_path):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    capture = capture_market_daily(**args)
    daily_path = Path(capture["manifest_path"]).with_name("daily.json")
    daily_path.write_bytes(daily_path.read_bytes() + b" ")

    with pytest.raises(MarketDailyCaptureBlocked, match="capture_daily_sha256_mismatch"):
        replay_market_daily_capture(
            capture_manifest_path=capture["manifest_path"],
            expected_capture_manifest_sha256=capture["manifest_sha256"],
            scope_path=args["scope_path"],
            expected_scope_sha256=args["expected_scope_sha256"],
            pit_generation_binding=args["pit_generation_binding"],
            expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
        )


def test_shadow_rejects_store_injection(tmp_path):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    capture = capture_market_daily(**args)
    store = _Store()
    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="shadow_store_injection_not_permitted",
    ):
        shadow_market_daily_capture(
            shadow_data_root=tmp_path / "shadow",
            production_data_root=tmp_path / "production",
            capture_manifest_path=capture["manifest_path"],
            expected_capture_manifest_sha256=capture["manifest_sha256"],
            scope_path=args["scope_path"],
            expected_scope_sha256=args["expected_scope_sha256"],
            pit_generation_binding=args["pit_generation_binding"],
            expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
            store=store,
        )


def test_shadow_clones_full_history_publishes_candidate_and_preserves_canonical(
    monkeypatch, tmp_path
):
    from quant_investor.config import config
    from tests.unit.test_market_data_parquet_direct_maintenance import (
        _write_pit_generation,
        _write_seed_snapshot,
    )

    production = tmp_path / "production"
    _write_seed_snapshot(production)
    generation = _write_pit_generation(production)
    monkeypatch.setattr(
        config,
        "PIT_UNIVERSE_SOURCE_ROOT",
        production / "parquet" / "cn" / "reference",
        raising=False,
    )
    provider = _Provider()
    evidence_root = tmp_path / "evidence"
    args = _capture_args(evidence_root, provider)
    shadow = tmp_path / "private-shadow"
    private_market_root = shadow / "parquet" / "cn"
    private_market_root.mkdir(parents=True, mode=0o700)
    private_pit_store = PITUniverseStore(
        root_dir=private_market_root / "reference",
        raw_root=evidence_root / "private-pit-raw",
        compatibility_path=evidence_root / "private-pit-compatibility.json",
    )
    private_records = [
        PITUniverseRecord(
            symbol=symbol,
            name=symbol,
            list_date="20200101",
            source_list_status="L",
            observed_at="2026-08-18T08:20:00Z",
            source_run_id="shadow-pit-unit",
        )
        for symbol in ("000001.SZ", "000002.SZ")
    ]
    private_pit_store.write_snapshot(
        raw_records=private_records,
        latest_records=private_records,
        observed_at="2026-08-18T08:20:00Z",
        source_run_id="shadow-pit-unit",
        write_compatibility_export=False,
    )
    private_binding = private_pit_store.load_generation_binding()
    protected_private_pit_bytes = {
        Path(private_binding["generation_manifest_path"]): Path(
            private_binding["generation_manifest_path"]
        ).read_bytes(),
        Path(private_binding["canonical_path"]): Path(
            private_binding["canonical_path"]
        ).read_bytes(),
        private_pit_store.manifest_path: private_pit_store.manifest_path.read_bytes(),
    }
    private_market_root.chmod(0o500)
    args["pit_generation_binding"] = {
        "generation_id": private_binding["generation_id"],
        "generation_manifest_path": private_binding["generation_manifest_path"],
        "generation_manifest_sha256": private_binding["generation_manifest_sha256"],
        "canonical_path": private_binding["canonical_path"],
        "canonical_sha256": private_binding["canonical_sha256"],
    }
    production_pointer = production / "parquet" / "cn" / "_latest.json"
    protected_pointer_bytes = production_pointer.read_bytes()
    production_pointer_payload = json.loads(protected_pointer_bytes)
    protected_manifest = Path(production_pointer_payload["manifest_path"])
    protected_manifest_bytes = protected_manifest.read_bytes()
    args["expected_market_pointer_sha256"] = hashlib.sha256(protected_pointer_bytes).hexdigest()
    capture = capture_market_daily(**args)

    external_binding_candidate = build_private_market_candidate(
        production_data_root=production,
        candidate_data_root=tmp_path / "external-pit-candidate",
        expected_production_market_pointer_sha256=args["expected_market_pointer_sha256"],
        private_pit_generation_binding={
            "generation_id": generation["generation_id"],
            "generation_manifest_path": generation["generation_manifest_path"],
            "generation_manifest_sha256": generation["generation_manifest_sha256"],
            "canonical_path": generation["canonical_path"],
            "canonical_sha256": generation["canonical_sha256"],
        },
    )
    assert external_binding_candidate["status"] == "READY"
    assert external_binding_candidate["private_pit_generation"] == {}

    unexpected_root = tmp_path / "unexpected-candidate"
    unexpected_root.mkdir()
    (unexpected_root / "junk.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="candidate_root_contains_unexpected_preexisting_files",
    ):
        build_private_market_candidate(
            production_data_root=production,
            candidate_data_root=unexpected_root,
            expected_production_market_pointer_sha256=args["expected_market_pointer_sha256"],
            private_pit_generation_binding=args["pit_generation_binding"],
        )

    result = shadow_market_daily_capture(
        shadow_data_root=shadow,
        production_data_root=production,
        capture_manifest_path=capture["manifest_path"],
        expected_capture_manifest_sha256=capture["manifest_sha256"],
        scope_path=args["scope_path"],
        expected_scope_sha256=args["expected_scope_sha256"],
        pit_generation_binding=args["pit_generation_binding"],
        expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
    )

    assert result["status"] == "SHADOW_CANDIDATE"
    assert result["canonical_write_performed"] is False
    assert result["candidate_data_root"] == str(shadow)
    assert Path(result["candidate_pointer_path"]).is_file()
    assert result["resource_preflight"]["status"] == "PASSED"
    assert Path(result["resource_preflight"]["path"]).is_file()
    assert production_pointer.read_bytes() == protected_pointer_bytes
    assert protected_manifest.read_bytes() == protected_manifest_bytes
    assert all(path.read_bytes() == raw for path, raw in protected_private_pit_bytes.items())
    assert not (production / "parquet" / "cn" / "_health_ledger.jsonl").exists()

    candidate_store = MarketDataStore(market="CN", data_root=shadow)
    assert candidate_store.validate_latest()["status"] == "passed"
    bound_pit = MarketDataReader(market="CN", data_root=shadow).coverage_bound_pit(refresh=True)
    assert bound_pit["generation_id"] == private_binding["generation_id"]
    assert bound_pit["canonical_path"] == private_binding["canonical_path"]
    assert {reference["role"] for reference in result["protected_candidate_pit_refs"]} == {
        "candidate_pit_manifest",
        "candidate_pit_membership",
        "candidate_pit_discovery_pointer",
    }
    seed_manifest_path = shadow / "parquet" / "cn" / "_snapshots" / "seed.json"
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    for field in ("manifest_path", "table_root", "derived_serving_root"):
        Path(seed_manifest[field]).relative_to(shadow)
    assert {reference["role"] for reference in result["protected_production_refs"]} >= {
        "production_market_pointer",
        "production_market_manifest",
        "production_market_table",
        "production_market_serving",
    }
    candidate_pointer = json.loads(
        Path(result["candidate_pointer_path"]).read_text(encoding="utf-8")
    )
    table_root = Path(candidate_pointer["table_root"])
    table = pd.concat(
        [pd.read_parquet(path) for path in sorted(table_root.rglob("*.parquet"))],
        ignore_index=True,
    )
    assert set(table["trade_date"].astype(str)) == {
        "20260315",
        "20260818",
    }
    assert (
        result["publication"]["capture_expected_market_pointer_sha256"]
        == args["expected_market_pointer_sha256"]
    )
    assert (
        result["publication"]["publication_previous_market_pointer_sha256"]
        == result["candidate_parent_pointer_sha256"]
    )


def test_private_candidate_builder_rejects_production_pointer_drift(
    tmp_path,
):
    production = tmp_path / "production"
    candidate = tmp_path / "candidate"
    production.mkdir()

    with pytest.raises(
        MarketDailyCaptureBlocked,
        match="production_market_snapshot_not_healthy",
    ):
        build_private_market_candidate(
            production_data_root=production,
            candidate_data_root=candidate,
            expected_production_market_pointer_sha256="a" * 64,
        )


def test_publish_uses_real_store_cas_and_passes_exact_readback(tmp_path):
    from tests.unit.test_market_data_parquet_direct_maintenance import (
        _write_pit_generation,
        _write_seed_snapshot,
    )

    _write_seed_snapshot(tmp_path)
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    generation = _write_pit_generation(tmp_path)
    args["pit_generation_binding"] = {
        "generation_id": generation["generation_id"],
        "generation_manifest_path": generation["generation_manifest_path"],
        "generation_manifest_sha256": generation["generation_manifest_sha256"],
        "canonical_path": generation["canonical_path"],
        "canonical_sha256": generation["canonical_sha256"],
    }
    pointer = tmp_path / "parquet" / "cn" / "_latest.json"
    args["expected_market_pointer_sha256"] = _sha256(pointer)
    capture = capture_market_daily(**args)

    result = publish_market_daily_capture(
        capture_manifest_path=capture["manifest_path"],
        expected_capture_manifest_sha256=capture["manifest_sha256"],
        data_root=tmp_path,
        scope_path=args["scope_path"],
        expected_scope_sha256=args["expected_scope_sha256"],
        pit_generation_binding=args["pit_generation_binding"],
        expected_market_pointer_sha256=args["expected_market_pointer_sha256"],
    )

    assert result["status"] == "PUBLISHED"
    assert result["parquet_commit"]["latest_complete_trade_date"] == "20260818"
    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()
    assert validation["status"] == "passed"
