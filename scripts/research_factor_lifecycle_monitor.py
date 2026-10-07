#!/usr/bin/env python3
"""Write one non-authorizing Factor lifecycle monitor report.

Reads only the registered production outcome revision heads (exact predecessor
chains re-validated by the outcome owner) and, for the size-exposure
diagnostic, the canonical table partitions of one frozen Market snapshot.
Writes a single write-once JSON file under ``reports/factor_lifecycle/``.
No pointer, weight, admission, observation or outcome is written.

    uv run python scripts/research_factor_lifecycle_monitor.py --workspace-root .
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.lifecycle_monitor import (  # noqa: E402
    MonitorPolicy,
    build_monitor_report,
    load_production_outcome_rows,
)


def _frozen_snapshot_ref(workspace: Path) -> dict[str, str]:
    pointer = json.loads((workspace / "data/parquet/cn/_latest.json").read_text("utf-8"))
    manifest = Path(pointer["manifest_path"]).resolve(strict=True)
    relative = manifest.relative_to((workspace / "data").resolve())
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def _size_by_origin(
    workspace: Path, origins: list[str], snapshot_ref: dict[str, str]
) -> dict[str, pd.Series]:
    from quant_investor.market.market_data_reader import MarketDataReader

    reader = MarketDataReader(
        market="CN", data_root=workspace / "data", frozen_snapshot_ref=snapshot_ref
    )
    result: dict[str, pd.Series] = {}
    if not origins:
        return result
    paths = reader.table_partition_paths(min(origins), max(origins))
    wanted = set(origins)
    for path in paths:
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "total_mv"])
        frame = frame[frame["trade_date"].astype(str).isin(wanted)]
        for date, group in frame.groupby("trade_date"):
            result[str(date)] = pd.Series(
                group["total_mv"].to_numpy(dtype=float), index=group["ts_code"].astype(str)
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", default="reports/factor_lifecycle")
    parser.add_argument("--half-life-sessions", type=float, default=120.0)
    parser.add_argument(
        "--priors",
        help="priors JSON written by research_factor_lifecycle_backtest.py",
    )
    parser.add_argument("--skip-size-exposure", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve(strict=True)
    policy = MonitorPolicy(half_life_sessions=args.half_life_sessions)
    rows, input_refs = load_production_outcome_rows(str(workspace))
    priors = None
    prior_source = "POLICY_DEFAULT_WEAK_PRIOR"
    if args.priors:
        prior_path = Path(args.priors).resolve(strict=True)
        raw = prior_path.read_bytes()
        priors = json.loads(raw)["priors"]
        prior_source = f"{prior_path.name}@{hashlib.sha256(raw).hexdigest()}"
    size_by_origin = None
    snapshot_ref = None
    if not args.skip_size_exposure:
        snapshot_ref = _frozen_snapshot_ref(workspace)
        origins = sorted({row.origin_session for row in rows})
        size_by_origin = _size_by_origin(workspace, origins, snapshot_ref)
    report = build_monitor_report(
        rows,
        policy=policy,
        input_refs=input_refs,
        size_by_origin=size_by_origin,
        market_snapshot_ref=snapshot_ref,
        priors=priors,
        prior_source=prior_source,
    )
    report["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    as_of = max((row.origin_session for row in rows), default="none")
    raw = (json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    output_dir = (workspace / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"monitor-{as_of}-{digest[:12]}.json"
    if target.exists() and target.read_bytes() != raw:
        raise SystemExit(f"refusing to replace a different report at {target}")
    target.write_bytes(raw)
    summary = {
        factor_id: {h: body["state"] for h, body in item["horizons"].items()}
        for factor_id, item in report["factors"].items()
    }
    print(json.dumps({"report": str(target), "sha256": digest, "states": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
