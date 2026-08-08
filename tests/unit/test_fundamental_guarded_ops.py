from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / "scripts"


def _run_python(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_cn_history_backfill_dry_run_is_offline_and_path_exact(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    pointer = data_root / "parquet" / "cn" / "_latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}")
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    result = _run_python(
        "backfill_cn_history.py",
        "--target-data-root",
        str(data_root),
        "--pointer-path",
        str(pointer),
        "--download-data-dir",
        str(download_dir),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
    )
    assert result.returncode == 0, result.stderr
    assert "offline; no provider imported" in result.stdout
    assert not (tmp_path / "checkpoint").exists()

    relative = _run_python(
        "backfill_cn_history.py",
        "--target-data-root",
        "data",
        "--pointer-path",
        str(pointer),
        "--download-data-dir",
        str(download_dir),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
    )
    assert relative.returncode != 0
    assert "must be absolute" in (relative.stdout + relative.stderr)


def test_cn_history_execute_needs_separate_live_acknowledgement(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    pointer = data_root / "parquet" / "cn" / "_latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}")
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    result = _run_python(
        "backfill_cn_history.py",
        "--target-data-root",
        str(data_root),
        "--pointer-path",
        str(pointer),
        "--download-data-dir",
        str(download_dir),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
        "--backup-dir",
        str(tmp_path / "backup"),
        "--execute",
    )
    assert result.returncode != 0
    assert "--allow-live-provider" in (result.stdout + result.stderr)
    assert not (tmp_path / "backup").exists()


def test_fundamental_rebuild_shell_is_offline_by_default(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.json"
    membership = tmp_path / "membership.parquet"
    scope = tmp_path / "scope.json"
    pointer.write_text("{}")
    membership.write_bytes(b"fixture")
    scope.write_text("{}")
    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS / "backfill_fundamental_pit_2015.sh"),
            "--repo-root",
            str(REPOSITORY),
            "--market-pointer",
            str(pointer),
            "--membership",
            str(membership),
            "--scope",
            str(scope),
            "--staging-root",
            str(tmp_path / "staging"),
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--backup-dir",
            str(tmp_path / "backup"),
            "--completion-receipt",
            str(tmp_path / "complete.json"),
            "--as-of",
            "20260807",
            "--years",
            "12",
            "--run-id",
            "fixture-run",
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "offline; provider and filesystem mutation disabled" in result.stdout
    assert not (tmp_path / "staging").exists()
    assert not (tmp_path / "backup").exists()


def test_checkpoint_boundary_is_diagnostic_only(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    generation = root / "_generations" / "g1"
    generation.mkdir(parents=True)
    (generation / "request_outcomes.json").write_text(
        json.dumps(
            {
                "outcomes": [
                    {
                        "table": "daily_basic",
                        "symbol": "920077.BJ",
                        "status": "success",
                        "history_complete": False,
                    }
                ]
            }
        )
    )
    result = _run_python(
        "declare_daily_basic_coverage_boundaries.py",
        "--checkpoint-outcome",
        str(generation / "request_outcomes.json"),
    )
    assert result.returncode == 0, result.stderr
    assert '"lane": "DIAGNOSTIC_ONLY"' in result.stdout
    assert '"status": "UNCONFIRMED"' in result.stdout
    assert '"authority_created": false' in result.stdout

    refused = _run_python(
        "declare_daily_basic_coverage_boundaries.py",
        "--checkpoint-outcome",
        str(generation / "request_outcomes.json"),
        "--execute",
    )
    assert refused.returncode != 0
    assert "can never be executed" in (refused.stdout + refused.stderr)


def test_owner_boundary_publication_is_exact_once_with_backup(tmp_path: Path) -> None:
    authority = {
        "schema_version": "daily-basic-provider-coverage-boundary-authority.v1",
        "authority": "OWNER_SUPPLIED_EXACT_PROVIDER_METADATA",
        "status": "CONFIRMED",
        "reason_code": "PROVIDER_COVERAGE_BOUNDARY",
        "coverage_starts": {"920077.BJ": "20211115"},
        "source_ref": {
            "artifact_id": "provider-metadata-920077",
            "artifact_version": "provider-metadata.v1",
            "byte_sha256": "1" * 64,
            "semantic_sha256": "2" * 64,
            "cutoff": "2026-08-07T00:00:00Z",
        },
    }
    authority["record_sha256"] = hashlib.sha256(_canonical(authority)).hexdigest()
    source = tmp_path / "authority.json"
    source.write_bytes(_canonical(authority))
    out = tmp_path / "published.json"
    result = _run_python(
        "declare_daily_basic_coverage_boundaries.py",
        "--authority-receipt",
        str(source),
        "--out",
        str(out),
        "--backup-dir",
        str(tmp_path / "backup-1"),
        "--execute",
    )
    assert result.returncode == 0, result.stderr
    first = out.read_bytes()
    assert (tmp_path / "backup-1" / "backup_manifest.json").is_file()
    assert json.loads(first)["lane"] == "AUTHORITATIVE_OWNER_SUPPLIED"

    repeated = _run_python(
        "declare_daily_basic_coverage_boundaries.py",
        "--authority-receipt",
        str(source),
        "--out",
        str(out),
        "--backup-dir",
        str(tmp_path / "backup-2"),
        "--execute",
    )
    assert repeated.returncode == 0, repeated.stderr
    assert "ALREADY_PRESENT" in repeated.stdout
    assert out.read_bytes() == first


def test_quarantine_requires_exact_files_and_has_fixture_readback(tmp_path: Path) -> None:
    table = tmp_path / "table.parquet"
    serving = tmp_path / "serving.parquet"
    rows = pd.DataFrame(
        [
            {"ts_code": "601989.SH", "trade_date": "19700101", "close": 1.0},
            {"ts_code": "601989.SH", "trade_date": "20260807", "close": 2.0},
        ]
    )
    rows.to_parquet(table, index=False)
    rows.to_parquet(serving, index=False)
    args = (
        "--table-file",
        str(table),
        "--serving-file",
        str(serving),
        "--snapshot-id",
        "fixture-snapshot",
    )
    dry = _run_python("quarantine_implausible_bar_dates.py", *args)
    assert dry.returncode == 0, dry.stderr
    assert "DRY RUN" in dry.stdout
    assert len(pd.read_parquet(table)) == 2

    executed = _run_python(
        "quarantine_implausible_bar_dates.py",
        *args,
        "--backup-dir",
        str(tmp_path / "backup"),
        "--quarantine-file",
        str(tmp_path / "quarantine.parquet"),
        "--evidence-file",
        str(tmp_path / "evidence.json"),
        "--execute",
    )
    assert executed.returncode == 0, executed.stderr
    assert len(pd.read_parquet(table)) == 1
    assert len(pd.read_parquet(serving)) == 1
    assert len(pd.read_parquet(tmp_path / "quarantine.parquet")) == 2
    assert (tmp_path / "backup" / "backup_manifest.json").is_file()

    repeated = _run_python("quarantine_implausible_bar_dates.py", *args)
    assert repeated.returncode == 0, repeated.stderr
    assert "ALREADY_CLEAN" in repeated.stdout


def test_staging_verifier_rejects_generation_discovery_and_relative_paths() -> None:
    missing = _run_python("verify_fundamental_staging.py")
    assert missing.returncode != 0
    assert "--staged-generation" in missing.stderr
    relative = _run_python(
        "verify_fundamental_staging.py",
        "--staged-generation",
        "staging.parquet",
        "--live-generation",
        "live.parquet",
    )
    assert relative.returncode != 0
    assert "must be absolute" in (relative.stdout + relative.stderr)


def test_staging_verifier_reads_exact_files_without_mutation(tmp_path: Path) -> None:
    live = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2021-06-01",
                "availability_date": "2021-05-01",
                "end_date": "2021-03-31",
                "fin_roe": 0.1,
                "fin_roa": 0.05,
                "fin_debt_to_assets": 0.4,
                "fin_net_profit_yoy": 0.2,
            }
        ]
    )
    staged = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2015-06-01",
                        "availability_date": "2015-05-01",
                        "end_date": "2015-03-31",
                        "fin_roe": 0.1,
                        "fin_roa": 0.05,
                        "fin_debt_to_assets": 0.4,
                        "fin_net_profit_yoy": 0.2,
                    }
                ]
            ),
            live,
        ],
        ignore_index=True,
    )
    staged_path = tmp_path / "staged.parquet"
    live_path = tmp_path / "live.parquet"
    staged.to_parquet(staged_path, index=False)
    live.to_parquet(live_path, index=False)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (staged_path, live_path)
    }
    result = _run_python(
        "verify_fundamental_staging.py",
        "--staged-generation",
        str(staged_path),
        "--live-generation",
        str(live_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (staged_path, live_path)
    }
    assert after == before
