#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from quant_investor.themes.pevc import DEFAULT_CANONICAL_PATH, PeVcKnowledgeStore
from quant_investor.themes.protocol_v2 import ThemeEvidenceEvent, evaluate_theme_protocol_v2
from quant_investor.themes.taxonomy import DEFAULT_TAXONOMY_PATH, ThemeTaxonomy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Theme Protocol v2 offline.")
    parser.add_argument("--theme-snapshot", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--evidence-events", default="")
    parser.add_argument("--pevc-canonical", default=str(DEFAULT_CANONICAL_PATH))
    parser.add_argument("--previous-states", default="")
    parser.add_argument("--downstream-gates", default="")
    parser.add_argument("--markov-regime", default="")
    parser.add_argument("--formal-enabled", action="store_true")
    parser.add_argument("--formal-kill-switch", action="store_true")
    parser.add_argument(
        "--trading-dates",
        default="",
        help="Local JSON trading-calendar artifact; required for formal evaluation.",
    )
    parser.add_argument(
        "--expected-trading-dates-hash",
        default="",
        help="Required SHA-256 of --trading-dates bytes in formal mode.",
    )
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = _load_object(args.theme_snapshot)
    theme_scores = snapshot.get("theme_scores")
    if not isinstance(theme_scores, Mapping):
        raise ValueError("theme snapshot requires theme_scores object")
    memberships = snapshot.get("symbol_theme_memberships")
    membership_theme_ids = sorted(
        {
            str(theme_id)
            for raw in (memberships.values() if isinstance(memberships, Mapping) else [])
            for theme_id in (raw if isinstance(raw, list) else [])
        }
    )
    raw_details = snapshot.get("symbol_theme_membership_details")
    membership_details = [
        dict(detail)
        for details in (raw_details.values() if isinstance(raw_details, Mapping) else [])
        for detail in (details if isinstance(details, list) else [])
        if isinstance(detail, Mapping)
    ]
    evidence = _load_events(args.evidence_events)
    pevc_path = Path(args.pevc_canonical)
    theses = (
        [
            item.to_dict()
            for item in PeVcKnowledgeStore(pevc_path).load(as_of=args.as_of)
        ]
        if pevc_path.exists()
        else []
    )
    previous_states = _load_object(args.previous_states) if args.previous_states else {}
    if isinstance(previous_states.get("states"), Mapping):
        previous_states = dict(previous_states["states"])
    downstream_gates = _load_object(args.downstream_gates) if args.downstream_gates else {}
    trading_dates, trading_dates_hash = _load_trading_dates(
        args.trading_dates,
        expected_hash=args.expected_trading_dates_hash,
        required=args.formal_enabled,
    )
    result = evaluate_theme_protocol_v2(
        theme_scores={str(key): dict(value or {}) for key, value in theme_scores.items()},
        taxonomy=ThemeTaxonomy.load(args.taxonomy),
        as_of=args.as_of,
        evidence_events=evidence,
        pevc_theses=theses,
        valid_membership_theme_ids=membership_theme_ids,
        theme_membership_details=membership_details,
        previous_states=previous_states,
        downstream_gates=downstream_gates,
        markov_regime=args.markov_regime,
        formal_enabled=args.formal_enabled,
        formal_kill_switch=args.formal_kill_switch,
        valid_trading_dates=trading_dates,
    )
    result["trading_dates_artifact_sha256"] = trading_dates_hash or None
    if not trading_dates:
        result.setdefault("diagnostic_notes", []).append(
            "trading_dates_artifact_missing_lifecycle_pending"
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _load_object(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return dict(payload)


def _load_events(path: str) -> list[ThemeEvidenceEvent]:
    if not path:
        return []
    events: list[ThemeEvidenceEvent] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"evidence line {line_number} must be an object")
        events.append(ThemeEvidenceEvent.from_mapping(payload))
    return events


def _load_trading_dates(
    path: str,
    *,
    expected_hash: str,
    required: bool,
) -> tuple[list[str], str]:
    if not path:
        if required:
            raise ValueError("formal mode requires --trading-dates")
        if expected_hash:
            raise ValueError("--expected-trading-dates-hash requires --trading-dates")
        return [], ""
    source = Path(path)
    raw = source.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if required and not expected_hash:
        raise ValueError("formal mode requires --expected-trading-dates-hash")
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("trading dates artifact SHA-256 mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = (
            payload.get("expected_open_dates")
            or payload.get("trading_dates")
            or []
        )
    else:
        raise ValueError("trading dates artifact must be an array or object")
    if not isinstance(values, list):
        raise ValueError("trading dates artifact dates must be an array")
    dates = sorted({str(value) for value in values if str(value).strip()})
    if required and not dates:
        raise ValueError("formal trading dates artifact is empty")
    return dates, actual_hash


if __name__ == "__main__":
    raise SystemExit(main())
