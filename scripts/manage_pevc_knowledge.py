#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_investor.themes.pevc import (
    DEFAULT_CANONICAL_PATH,
    DEFAULT_DRAFT_DIR,
    PeVcKnowledgeStore,
    import_pevc_draft,
    initialize_pevc_approval_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the local-only PE/VC thesis knowledge base.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-key")
    init_parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))

    import_parser = subparsers.add_parser("import-draft")
    import_parser.add_argument("source")
    import_parser.add_argument(
        "--source-type",
        choices=("auto", "json", "markdown", "word", "notion_export"),
        default="auto",
    )
    import_parser.add_argument("--draft-dir", default=str(DEFAULT_DRAFT_DIR))
    import_parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))

    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("draft")
    approve_parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))
    approve_parser.add_argument("--expected-draft-hash", required=True)
    approve_parser.add_argument("--approved-at", default="")
    approve_parser.add_argument("--migration-mode", action="store_true")
    approve_parser.add_argument("--migration-evidence-file", default="")
    approve_parser.add_argument("--expected-migration-evidence-hash", default="")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL_PATH))
    validate_parser.add_argument("--as-of", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-key":
        result = initialize_pevc_approval_key(args.canonical)
        result["network_called"] = False
    elif args.command == "import-draft":
        result = import_pevc_draft(
            args.source,
            source_type=args.source_type,
            draft_dir=args.draft_dir,
            canonical_path=args.canonical,
        )
    elif args.command == "approve":
        thesis = PeVcKnowledgeStore(args.canonical).approve_draft(
            args.draft,
            expected_draft_hash=args.expected_draft_hash,
            approved_at=args.approved_at or None,
            migration_mode=args.migration_mode,
            migration_evidence_file=args.migration_evidence_file or None,
            expected_migration_evidence_hash=(
                args.expected_migration_evidence_hash
            ),
        )
        result = {
            "status": "approved",
            "canonical_path": str(Path(args.canonical)),
            "thesis_id": thesis.thesis_id,
            "theme_id": thesis.theme_id,
            "version": thesis.version,
            "content_hash": thesis.content_hash or thesis.compute_content_hash(),
            "network_called": False,
        }
    else:
        theses = PeVcKnowledgeStore(args.canonical).load(as_of=args.as_of or None)
        result = {
            "status": "valid",
            "canonical_path": str(Path(args.canonical)),
            "record_count": len(theses),
            "records": [
                {
                    "thesis_id": thesis.thesis_id,
                    "theme_id": thesis.theme_id,
                    "version": thesis.version,
                    "content_hash": thesis.content_hash or thesis.compute_content_hash(),
                }
                for thesis in theses
            ],
            "network_called": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
