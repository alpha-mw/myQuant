#!/usr/bin/env python3
"""Render a concise summary and evidence appendix from one checker-approved v2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.strategy_records.performance import (  # noqa: E402
    assert_private_tmp,
    immutable_write,
)  # noqa: E402
from scripts.check_cn_weekly_review_evidence import check  # noqa: E402
from scripts.cn_weekly_review_v2 import SCHEMA_V2, render_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    bundle_path = Path(args.bundle)
    result = check(bundle_path)
    raw = bundle_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != result["byte_sha256"]:
        raise ValueError("checked bundle changed before rendering")
    bundle = json.loads(raw)
    if bundle["schema_id"] != SCHEMA_V2:
        parser.error("v1 is historical readback only; rendering requires v2")
    output = assert_private_tmp(Path(args.output_dir))
    summary = render_summary(bundle)
    summary += f"\n完整证据：[evidence-appendix.md]({output / 'evidence-appendix.md'})\n"
    appendix = "# Weekly evidence appendix\n\n"
    appendix += "This is a read-only projection of the exact checker-approved bundle.\n\n"
    appendix += (
        "```json\n"
        + json.dumps({"checker": result, "bundle": bundle}, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    refs = []
    for name, body in [("investment-summary.md", summary), ("evidence-appendix.md", appendix)]:
        path = output / name
        sha = immutable_write(path, body.encode(), max_bytes=16 * 1024 * 1024)
        refs.append({"path": str(path), "sha256": sha})
    print(
        json.dumps({"checker_ok": True, "bundle_sha256": result["byte_sha256"], "artifacts": refs})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
