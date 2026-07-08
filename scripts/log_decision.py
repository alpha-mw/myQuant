#!/usr/bin/env python3
"""Append or inspect local advisory decision-log JSONL records.

The log is a governance artifact under ``results/decision_log/``. It is local,
offline, append-only by default, and intentionally separate from strategy
records. This script does not call providers, brokers, LLMs, or execution APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "decision_log" / "decision_log.jsonl"
EVENT_TYPES = {"pipeline_proposal", "advisory", "human_action"}
ADVISORY_SOURCES = {"codex_thread", "claude", "workbench", "other"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--metadata-json must decode to an object")
    return payload


def make_event(payload: dict[str, Any]) -> dict[str, Any]:
    event = dict(payload)
    event_type = str(event.get("event_type") or "").strip()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(EVENT_TYPES)}")
    event.setdefault("schema_version", "decision_log.v1")
    event.setdefault("recorded_at", _utc_now())
    event.setdefault("metadata", {})
    if event_type == "advisory":
        source = str(event.get("answer_source") or "").strip()
        if source not in ADVISORY_SOURCES:
            raise ValueError(f"advisory answer_source must be one of {sorted(ADVISORY_SOURCES)}")
        if not str(event.get("question") or "").strip():
            raise ValueError("advisory requires question")
        if not str(event.get("answer_summary") or "").strip():
            raise ValueError("advisory requires answer_summary")
    if event_type == "human_action" and not str(event.get("action") or "").strip():
        raise ValueError("human_action requires action")
    if event_type == "pipeline_proposal" and not str(event.get("proposal_summary") or "").strip():
        raise ValueError("pipeline_proposal requires proposal_summary")
    event["event_id"] = event.get("event_id") or _event_id(event)
    return event


def _event_id(event: dict[str, Any]) -> str:
    basis = json.dumps(
        {key: event.get(key) for key in sorted(event) if key != "event_id"},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "decision-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def append_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    resolved = make_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(resolved, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def build_event_from_args(args: argparse.Namespace) -> dict[str, Any]:
    event = {
        "event_type": args.event_type,
        "trade_date": args.trade_date,
        "channel": args.channel,
        "question": args.question,
        "answer_summary": args.answer_summary,
        "answer_source": args.answer_source,
        "symbol": args.symbol,
        "action": args.action,
        "proposal_summary": args.proposal_summary,
        "rejected_options": [item for item in (args.rejected_option or []) if item],
        "regime_state": args.regime_state,
        "machine_suggestion": args.machine_suggestion,
        "metadata": _load_metadata(args.metadata_json),
    }
    return {key: value for key, value in event.items() if value not in (None, "", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--list", type=int, default=0, help="Print the latest N events instead of appending.")
    parser.add_argument("--event-type", choices=sorted(EVENT_TYPES))
    parser.add_argument("--trade-date")
    parser.add_argument("--channel")
    parser.add_argument("--question")
    parser.add_argument("--answer-summary")
    parser.add_argument("--answer-source", choices=sorted(ADVISORY_SOURCES))
    parser.add_argument("--symbol")
    parser.add_argument("--action")
    parser.add_argument("--proposal-summary")
    parser.add_argument("--rejected-option", action="append")
    parser.add_argument("--regime-state")
    parser.add_argument("--machine-suggestion")
    parser.add_argument("--metadata-json")
    args = parser.parse_args()
    if args.list:
        events = read_events(args.log_path)
        print(json.dumps(events[-args.list :], ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.event_type:
        raise SystemExit("--event-type is required unless --list is used")
    event = append_event(args.log_path, build_event_from_args(args))
    print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
