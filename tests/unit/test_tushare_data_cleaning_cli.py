from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-11",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "vol": 1000,
                "amount": 10000,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "2026-03-12",
                "open": 10.0,
                "high": 9.0,
                "low": 9.8,
                "close": 10.2,
                "vol": -1,
                "amount": -100,
                "adj_factor": 1.0,
            },
        ]
    ).to_csv(path, index=False)


def test_clean_tushare_downloads_cli_writes_artifacts(tmp_path):
    root = tmp_path / "market"
    _write_fixture(root / "hs300" / "000001.SZ.csv")
    report_dir = tmp_path / "reports"
    raw_dir = tmp_path / "raw"
    quarantine_dir = tmp_path / "quarantine"
    readiness_dir = tmp_path / "readiness"
    parquet_dir = tmp_path / "parquet"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/clean_tushare_downloads.py",
            "--root-dir",
            str(root),
            "--report-dir",
            str(report_dir),
            "--raw-backup-dir",
            str(raw_dir),
            "--quarantine-dir",
            str(quarantine_dir),
            "--factor-readiness-dir",
            str(readiness_dir),
            "--parquet-dir",
            str(parquet_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "total files=1" in proc.stdout
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    result = payload["results"][0]
    assert Path(result["cleaning_report_path"]).exists()
    assert Path(result["raw_backup_path"]).exists()
    assert Path(result["quarantine_path"]).exists()
    assert Path(result["row_flags_path"]).exists()
    assert Path(result["cell_flags_path"]).exists()
    assert Path(result["factor_ready_masks_path"]).exists()
    assert Path(result["matrix_coverage_path"]).exists()
    assert Path(result["storage_audit_report_path"]).exists()


def test_clean_tushare_downloads_cli_parquet_flag_is_non_blocking(tmp_path):
    root = tmp_path / "market"
    _write_fixture(root / "hs300" / "000001.SZ.csv")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/clean_tushare_downloads.py",
            "--root-dir",
            str(root),
            "--report-dir",
            str(tmp_path / "reports"),
            "--raw-backup-dir",
            str(tmp_path / "raw"),
            "--quarantine-dir",
            str(tmp_path / "quarantine"),
            "--factor-readiness-dir",
            str(tmp_path / "readiness"),
            "--parquet-dir",
            str(tmp_path / "parquet"),
            "--parquet-shadow-write",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert payload["results"][0]["parquet_status"] in {
        "shadow_written",
        "unsupported",
        "failed",
    }
