from pathlib import Path

from quant_investor.market.cn_official_suspension_evidence import (
    build_official_web_suspension_evidence,
    official_suspension_evidence_path,
    read_official_web_suspension_evidence,
)
from quant_investor.market.cn_nontrading_evidence import file_sha256


def test_official_web_suspension_evidence_is_hash_bound(tmp_path: Path) -> None:
    pit_path = tmp_path / "pit.parquet"
    pit_path.write_bytes(b"pit")
    pit_sha = file_sha256(pit_path)
    raw_path = tmp_path / "notice.html"
    raw_path.write_text(
        "301234 自2026年7月8日开市起停牌",
        encoding="utf-8",
    )
    evidence_path = build_official_web_suspension_evidence(
        output_root=tmp_path / "data",
        trade_date="20260721",
        symbols=["301234.SZ"],
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
        query_run_id="web-notice-test-1",
        notices=[
            {
                "ts_code": "301234.SZ",
                "notice_title": "停牌公告",
                "issuer_host": "example.test",
                "source_url": "https://example.test/notice",
                "publication_date": "20260708",
                "suspension_start_date": "20260708",
                "suspension_end_date_exclusive": "20260722",
                "required_text_fragments": ["301234", "2026年7月8日开市起停牌"],
                "raw_source_path": raw_path,
            }
        ],
    )
    result = read_official_web_suspension_evidence(
        evidence_path,
        trade_date="20260721",
        expected_pit_membership_path=pit_path,
        expected_pit_membership_sha256=pit_sha,
        expected_symbols=["301234.SZ"],
    )
    assert result["status"] == "passed"
    assert result["verified_symbols"] == ["301234.SZ"]
    assert evidence_path == official_suspension_evidence_path(
        tmp_path / "data",
        trade_date="20260721",
        pit_membership_sha256=pit_sha,
    )

    durable_raw_path = next((evidence_path.parent / "raw").glob("*"))
    durable_raw_path.write_text("tampered", encoding="utf-8")
    tampered = read_official_web_suspension_evidence(
        evidence_path,
        trade_date="20260721",
        expected_pit_membership_path=pit_path,
        expected_pit_membership_sha256=pit_sha,
        expected_symbols=["301234.SZ"],
    )
    assert tampered["status"] == "blocked"
