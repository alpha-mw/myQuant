from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pytest

from quant_investor.market.fundamental_historical_taint import (
    HISTORICAL_TAINT_STATUS,
    HistoricalTaintError,
    build_historical_taint_registry,
    validate_historical_taint_registry,
)
from quant_investor.market.fundamental_provider_contract import (
    canonical_json_sha256,
)
from quant_investor.market.fundamental_successor_source import (
    FundamentalSuccessorSourceError,
    acquire_successor_support,
    build_successor_support_plan,
)
from quant_investor.v17_v4_runtime.tushare_https import (
    replay_tushare_response_bytes,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Client:
    def __init__(
        self,
        *,
        conflict: bool = False,
        same_period_delta: bool = False,
    ) -> None:
        self.conflict = conflict
        self.same_period_delta = same_period_delta
        self.calls = 0

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ):
        self.calls += 1
        availability = str(
            params.get("trade_date")
            or params.get("ann_date")
            or params.get("start_date")
        )
        end_date = (
            "20220331"
            if self.conflict or self.same_period_delta
            else "20260630"
        )
        values: dict[str, Any] = {
            "ts_code": "600816.SH",
            "ann_date": availability,
            "f_ann_date": availability,
            "end_date": end_date,
            "trade_date": availability,
            "report_type": "1",
            "comp_type": "1",
            "update_flag": "1",
            "type": "预增",
            "summary": "fixture",
            "change_reason": "fixture",
        }
        for field in expected_fields:
            values.setdefault(field, 1.25)
        rows = [dict(values)]
        if self.conflict and api_name == "cashflow_vip":
            first = dict(values)
            first["comp_type"] = "2"
            first["update_flag"] = "0"
            first["free_cashflow"] = None
            second = dict(values)
            second["comp_type"] = "1"
            second["update_flag"] = "1"
            second["free_cashflow"] = -374840901.64
            rows = [first, second]
        physical = [
            [row[field] for field in expected_fields]
            for row in rows
        ]
        raw = _canonical_bytes(
            {
                "code": 0,
                "data": {
                    "count": len(physical),
                    "fields": list(expected_fields),
                    "has_more": False,
                    "items": physical,
                },
                "detail": "",
                "msg": "",
                "request_id": f"request-{self.calls}",
            }
        )
        return replay_tushare_response_bytes(
            raw,
            api_name=api_name,
            expected_fields=expected_fields,
            strict_decimal_decode=True,
        )


def _capture(
    root: Path,
    *,
    support_start: str,
    target: str,
    client: _Client,
) -> dict[str, Any]:
    plan = build_successor_support_plan(
        support_start=support_start,
        target_date=target,
        open_sessions=[target],
        symbols=["600816.SH"],
        canonical_subject_scope_authority_sha256="a" * 64,
    )
    return acquire_successor_support(
        plan=plan,
        client=client,
        fileset_root=root,
        captured_pointer_bytes={
            "predecessor": b'{"generation":"parent"}\n',
            "market": b'{"snapshot":"target"}\n',
            "pit": b'{"generation":"pit"}\n',
        },
        immutable_refs={"fixture": {"sha256": "b" * 64}},
        implementation_sha256="c" * 64,
        captured_at="2026-08-14T09:00:00Z",
        max_attempts=1,
        retry_backoff_seconds=(),
        physical_memory_bytes=8 * 1024 * 1024 * 1024,
        available_memory_bytes=4 * 1024 * 1024 * 1024,
        rlimit_headroom_bytes=4 * 1024 * 1024 * 1024,
        table_memory_limit_bytes=128 * 1024 * 1024,
        minimum_free_disk_bytes=64 * 1024 * 1024,
        maximum_record_bytes=16 * 1024 * 1024,
        sleeper=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )


def _predecessor(tmp_path: Path) -> dict[str, Any]:
    period_path = tmp_path / "fundamental_period.parquet"
    daily_path = tmp_path / "fundamental_daily.parquet"
    quarantine_path = tmp_path / "fundamental_quarantine.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": "600816.SH",
                "end_date": "20220331",
                "availability_date": "20220430",
                "source_version": "20220430",
                "source": "fixture",
                "fetched_at": "2026-08-06T09:00:00Z",
                "free_cashflow": -374840901.64,
            },
            {
                "ts_code": "600816.SH",
                "end_date": "20260331",
                "availability_date": "20260429",
                "source_version": "20260429",
                "source": "fixture",
                "fetched_at": "2026-08-06T09:00:00Z",
                "free_cashflow": -127514500.0,
            },
        ]
    ).to_parquet(period_path, index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "600816.SH",
                "trade_date": "20260806",
                "end_date": "20260331",
                "availability_date": "20260429",
            }
        ]
    ).to_parquet(daily_path, index=False)
    pd.DataFrame(columns=["ts_code", "reason"]).to_parquet(
        quarantine_path,
        index=False,
    )
    table_paths = {
        "fundamental_period": period_path,
        "fundamental_daily": daily_path,
        "fundamental_quarantine": quarantine_path,
    }
    reference: dict[str, Any] = {
        "generation_id": "parent",
        "cutoff": "20260806",
        "pointer_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "table_sha256": {
            name: _sha_file(path) for name, path in table_paths.items()
        },
        "provenance_schema_version": (
            "cn-fundamental-primary-provenance.v2"
        ),
        "immutable_refs": {
            str(path): {"path": str(path), "sha256": _sha_file(path)}
            for path in table_paths.values()
        },
    }
    reference["reference_sha256"] = canonical_json_sha256(reference)
    return reference


def test_historical_conflict_isolated_only_when_delta_does_not_touch_key(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "legacy_capture"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT",
    ):
        _capture(
            old_root,
            support_start="20220430",
            target="20220430",
            client=_Client(conflict=True),
        )
    delta_root = tmp_path / "delta_capture"
    _capture(
        delta_root,
        support_start="20260807",
        target="20260807",
        client=_Client(),
    )
    predecessor = _predecessor(tmp_path)
    registry, evidence = build_historical_taint_registry(
        failure_evidence=[
            {
                "failure_root": str(tmp_path / "legacy_capture-failures"),
                "ordinal": 1,
            }
        ],
        predecessor=predecessor,
        parent_cutoff="20260806",
        target_cutoff="20260807",
        delta_fileset_root=delta_root,
    )

    assert registry["status"] == HISTORICAL_TAINT_STATUS
    assert registry["historical_conflict_count"] == 1
    assert registry["poisoned_keyset"] == ["600816.SH|20220331"]
    assert registry["winner_selection_applied"] is False
    evidence_root = tmp_path / "sealed_evidence"
    evidence_root.mkdir(mode=0o700)
    for relative, source in evidence.items():
        destination = evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o600)
    registry_path = evidence_root / "historical_taint" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    registry_path.write_bytes(_canonical_bytes(registry))
    registry_path.chmod(0o600)

    replayed = validate_historical_taint_registry(
        registry_path,
        evidence_root=evidence_root,
        predecessor=predecessor,
        delta_fileset_root=delta_root,
    )
    assert replayed["registry_sha256"] == registry["registry_sha256"]


def test_historical_conflict_blocks_when_same_period_reappears_in_delta(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "legacy_capture"
    with pytest.raises(FundamentalSuccessorSourceError):
        _capture(
            old_root,
            support_start="20220430",
            target="20220430",
            client=_Client(conflict=True),
        )
    delta_root = tmp_path / "delta_capture"
    _capture(
        delta_root,
        support_start="20260807",
        target="20260807",
        client=_Client(same_period_delta=True),
    )

    with pytest.raises(HistoricalTaintError, match="SAME_PERIOD_DELTA"):
        build_historical_taint_registry(
            failure_evidence=[
                {
                    "failure_root": str(
                        tmp_path / "legacy_capture-failures"
                    ),
                    "ordinal": 1,
                }
            ],
            predecessor=_predecessor(tmp_path),
            parent_cutoff="20260806",
            target_cutoff="20260807",
            delta_fileset_root=delta_root,
        )
