#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from quant_investor.themes.membership_migration import (
    DEFAULT_CANONICAL_PATH,
    DEFAULT_DRAFT_DIR,
    approve_membership_v2_draft,
    build_membership_v2_draft,
    validate_membership_v2_store,
)
from quant_investor.themes.taxonomy import DEFAULT_TAXONOMY_PATH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline fail-closed Theme membership v2 migration.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    draft = commands.add_parser("build-draft")
    draft.add_argument("source")
    draft.add_argument("--symbol-master", default="")
    draft.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    draft.add_argument("--draft-dir", default=str(DEFAULT_DRAFT_DIR))

    approve = commands.add_parser("approve")
    approve.add_argument("draft")
    approve.add_argument("--expected-draft-hash", required=True)
    approve.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))
    approve.add_argument("--expected-canonical-hash", default="")

    validate = commands.add_parser("validate")
    validate.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))
    validate.add_argument("--symbol-master", default="")
    validate.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    validate.add_argument("--as-of", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-draft":
        result = build_membership_v2_draft(
            args.source,
            symbol_master_path=args.symbol_master or None,
            taxonomy_path=args.taxonomy,
            draft_dir=args.draft_dir,
        )
    elif args.command == "approve":
        result = approve_membership_v2_draft(
            args.draft,
            expected_draft_hash=args.expected_draft_hash,
            canonical_path=args.canonical,
            expected_canonical_hash=args.expected_canonical_hash,
        )
    else:
        result = validate_membership_v2_store(
            args.canonical,
            symbol_master_path=args.symbol_master or None,
            taxonomy_path=args.taxonomy,
            as_of=args.as_of,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"ready_for_approval", "approved", "success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
