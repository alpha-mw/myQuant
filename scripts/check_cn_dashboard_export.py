#!/usr/bin/env python3
"""Read back and verify a generated CN aggressive Dashboard bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cn_dashboard_common import validate_bundle_shape, verify_source_refs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "portfolio_dashboard"
    / "private"
    / "generated"
    / "cn_aggressive_dashboard.v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw = args.bundle.read_bytes()
        if raw != args.bundle.read_bytes():
            raise ValueError("bundle_unstable_double_read")
        bundle = json.loads(raw.decode("utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(
            json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False)
        )
        return 2
    errors = validate_bundle_shape(bundle) + verify_source_refs(
        bundle, args.project_root.resolve()
    )
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "status": bundle["status"],
                "latest_valid_record": bundle["latest_valid_record"],
                "content_sha256": bundle["content_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
