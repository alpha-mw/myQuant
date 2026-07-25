from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_investor.market.cn_secondary_daily_evidence import (
    probe_eastmoney_daily_evidence,
    validate_secondary_daily_evidence,
)


def _pit_binding(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "pit.parquet"
    path.write_bytes(b"pit-membership")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_secondary_probe_persists_raw_sha_and_normalizes_amount(tmp_path: Path) -> None:
    pit_path, pit_sha = _pit_binding(tmp_path)
    raw = json.dumps(
        {
            "data": {
                "klines": [
                    "2026-07-22,40.80,40.80,40.80,40.80,5036,20546880.00,0.00,20.00,6.80,0.00"
                ]
            }
        }
    ).encode()

    payload, manifest_path = probe_eastmoney_daily_evidence(
        ["301234.SZ"],
        "20260722",
        output_root=tmp_path,
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
        query_run_id="secondary-test-1",
        fetch_raw=lambda _url: raw,
    )

    assert payload["observed_symbols"] == ["301234.SZ"]
    assert payload["normalized_rows"][0]["amount"] == 20546.88
    assert Path(payload["entries"][0]["raw_capture_path"]).read_bytes() == raw
    readback = validate_secondary_daily_evidence(
        manifest_path,
        target_trade_date="20260722",
        expected_pit_membership_sha256=pit_sha,
    )
    assert readback["status"] == "passed"
    assert readback["blockers"] == []


def test_secondary_empty_response_does_not_create_observed_bar(tmp_path: Path) -> None:
    pit_path, pit_sha = _pit_binding(tmp_path)
    raw = b'{"data":null,"rc":0}'

    payload, manifest_path = probe_eastmoney_daily_evidence(
        ["688237.SH", "688277.SH"],
        "20260721",
        output_root=tmp_path,
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
        query_run_id="secondary-test-empty",
        fetch_raw=lambda _url: raw,
    )

    assert payload["observed_symbols"] == []
    assert all(entry["status"] == "empty" for entry in payload["entries"])
    readback = validate_secondary_daily_evidence(
        manifest_path,
        target_trade_date="20260721",
        expected_pit_membership_sha256=pit_sha,
    )
    assert readback["status"] == "passed"


def test_secondary_raw_capture_tamper_is_blocking(tmp_path: Path) -> None:
    pit_path, pit_sha = _pit_binding(tmp_path)
    raw = json.dumps(
        {
            "data": {
                "klines": [
                    "2026-07-22,40.80,40.80,40.80,40.80,5036,20546880.00,0.00,20.00,6.80,0.00"
                ]
            }
        }
    ).encode()
    _payload, manifest_path = probe_eastmoney_daily_evidence(
        ["301234.SZ"],
        "20260722",
        output_root=tmp_path,
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
        query_run_id="secondary-test-tamper",
        fetch_raw=lambda _url: raw,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest["entries"][0]["raw_capture_path"])
    raw_path.write_bytes(raw.replace(b"40.80", b"41.80", 1))

    readback = validate_secondary_daily_evidence(
        manifest_path,
        target_trade_date="20260722",
        expected_pit_membership_sha256=pit_sha,
    )
    assert readback["status"] == "blocked"
    assert "secondary_raw_capture_sha256_mismatch" in readback["blockers"]


def test_secondary_malformed_raw_response_is_blocking(tmp_path: Path) -> None:
    pit_path, pit_sha = _pit_binding(tmp_path)
    payload, manifest_path = probe_eastmoney_daily_evidence(
        ["920685.BJ"],
        "20260721",
        output_root=tmp_path,
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
        query_run_id="secondary-test-malformed",
        fetch_raw=lambda _url: b"not-json",
    )
    assert payload["entries"][0]["query_succeeded"] is True
    readback = validate_secondary_daily_evidence(
        manifest_path,
        target_trade_date="20260721",
        expected_pit_membership_sha256=pit_sha,
    )
    assert readback["status"] == "blocked"
    assert any(
        item.startswith("secondary_raw_payload_recompute_failed:")
        for item in readback["blockers"]
    )
