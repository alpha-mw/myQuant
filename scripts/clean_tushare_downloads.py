#!/usr/bin/env python3
"""Clean local Tushare download Parquet files and emit factor-readiness sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_investor.config import config
from quant_investor.market.tushare_data_cleaning import (
    TushareStorageOptimizationConfig,
    clean_tushare_download_directory,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline clean Tushare daily download Parquet files with factor-readiness and storage reports."
    )
    parser.add_argument("--root-dir", default=config.CN_MARKET_DATA_DIR)
    parser.add_argument("--table", default="daily")
    parser.add_argument("--include", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--raw-backup-dir", default=config.TUSHARE_RAW_BACKUP_DIR)
    parser.add_argument("--quarantine-dir", default=config.TUSHARE_QUARANTINE_DIR)
    parser.add_argument("--report-dir", default=config.TUSHARE_CLEANING_REPORT_DIR)
    parser.add_argument("--factor-readiness-dir", default=config.TUSHARE_FACTOR_READINESS_DIR)
    parser.add_argument("--parquet-dir", default=config.TUSHARE_PARQUET_DIR)
    parser.add_argument("--parquet-shadow-write", action="store_true")
    parser.add_argument("--parquet-canonical", action="store_true")
    parser.add_argument("--no-factor-readiness", action="store_true")
    parser.add_argument("--no-storage-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    storage_config = TushareStorageOptimizationConfig(
        parquet_shadow_write=bool(args.parquet_shadow_write),
        parquet_canonical=bool(args.parquet_canonical),
        delete_redundant_csv=bool(config.TUSHARE_DELETE_REDUNDANT_CSV),
        parquet_dir=args.parquet_dir,
        parquet_compression=config.TUSHARE_PARQUET_COMPRESSION,
        metadata={"source": "scripts/clean_tushare_downloads.py"},
    )
    summary = clean_tushare_download_directory(
        Path(args.root_dir),
        table_name=args.table,
        include=args.include,
        exclude=args.exclude,
        promote=not args.no_promote,
        raw_backup_dir=args.raw_backup_dir,
        quarantine_dir=args.quarantine_dir,
        report_dir=args.report_dir,
        factor_readiness_dir=args.factor_readiness_dir,
        enable_factor_readiness=not args.no_factor_readiness,
        enable_storage_audit=not args.no_storage_audit,
        storage_config=storage_config,
    )
    print(
        "total files={total_files} pass={pass_count} warn={warn_count} fail={fail_count} "
        "raw_rows={raw_rows} clean_rows={clean_rows} quarantine_rows={quarantine_rows}".format(**summary)
    )
    readiness = sorted(
        {
            str(item.get("factor_readiness_status"))
            for item in summary["results"]
            if item.get("factor_readiness_status")
        }
    )
    storage = sorted(
        {
            str(item.get("storage_status"))
            for item in summary["results"]
            if item.get("storage_status")
        }
    )
    parquet = sorted(
        {
            str(item.get("parquet_status"))
            for item in summary["results"]
            if item.get("parquet_status")
        }
    )
    print(f"readiness_status={','.join(readiness) if readiness else 'none'}")
    print(f"storage_status={','.join(storage) if storage else 'none'}")
    print(f"parquet_status={','.join(parquet) if parquet else 'none'}")
    for item in summary["results"]:
        print(
            "report={cleaning_report_path} raw={raw_backup_path} quarantine={quarantine_path} "
            "row_flags={row_flags_path} cell_flags={cell_flags_path} masks={factor_ready_masks_path} "
            "coverage={matrix_coverage_path} storage={storage_audit_report_path} parquet={parquet_migration_report_path}".format(
                **item
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
