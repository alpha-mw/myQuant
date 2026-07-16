#!/usr/bin/env python3
"""Retired v2 rollback CLI; v3 has no authorized registry mutation yet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_protocol_v3 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
    PROTOCOL_VERSION,
    protocol_hash,
)
from quant_investor.factors.registry_store import (  # noqa: E402
    load_registry_snapshot_strict,
    rollback_factor_record_patch,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_hash(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _load_wal(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"inverse WAL is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("inverse WAL must be a JSON object")
    return dict(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--inverse-wal", required=True)
    parser.add_argument("--mutation-budget-ledger", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--expected-protocol-hash", required=True)
    parser.add_argument("--expected-current-registry-sha256", required=True)
    parser.add_argument("--expected-inverse-wal-sha256", required=True)
    parser.add_argument("--expected-transition-hash", required=True)
    parser.add_argument("--expected-mutation-plan-hash", required=True)
    parser.add_argument("--expected-evidence-hash", required=True)
    parser.add_argument(
        "--rollback-wal",
        default="",
        help="Required output WAL path only with --apply-rollback.",
    )
    parser.add_argument(
        "--apply-rollback",
        action="store_true",
        help="Apply the inverse patch. The default is a read-only dry run.",
    )
    args = parser.parse_args(argv)
    for field in (
        "expected_current_registry_sha256",
        "expected_inverse_wal_sha256",
        "expected_transition_hash",
        "expected_mutation_plan_hash",
        "expected_evidence_hash",
    ):
        try:
            _required_hash(getattr(args, field), f"--{field.replace('_', '-')}")
        except ValueError as exc:
            parser.error(str(exc))
    if args.apply_rollback and not str(args.rollback_wal or "").strip():
        parser.error("--apply-rollback requires --rollback-wal")
    return args


def run_rollback(args: argparse.Namespace) -> dict[str, Any]:
    del args
    raise ValueError(
        "FactorGovernanceProtocol v2 rollback is retired; "
        + FORWARD_PRODUCTION_APPLY_BLOCKER
    )

    # Historical implementation below is intentionally unreachable and kept
    # only for source-level audit of old WAL semantics.
    registry_path = Path(args.registry_path).expanduser()
    inverse_wal_path = Path(args.inverse_wal).expanduser()
    ledger_path = Path(args.mutation_budget_ledger).expanduser()
    snapshot = load_registry_snapshot_strict(registry_path)
    expected_current_sha = _required_hash(
        args.expected_current_registry_sha256,
        "expected current registry SHA",
    )
    if snapshot.registry_sha256 != expected_current_sha:
        raise ValueError("expected current registry SHA mismatch")

    expected_wal_sha = _required_hash(
        args.expected_inverse_wal_sha256,
        "expected inverse WAL SHA",
    )
    wal_sha = _file_sha256(inverse_wal_path)
    if wal_sha != expected_wal_sha:
        raise ValueError("inverse WAL SHA mismatch")
    mutation_manifest = _load_wal(inverse_wal_path)
    expected_hashes = {
        "protocol_hash": _required_hash(
            args.expected_protocol_hash,
            "expected protocol hash",
        ),
        "transition_hash": _required_hash(
            args.expected_transition_hash,
            "expected transition hash",
        ),
        "mutation_plan_hash": _required_hash(
            args.expected_mutation_plan_hash,
            "expected mutation plan hash",
        ),
        "evidence_hash": _required_hash(
            args.expected_evidence_hash,
            "expected evidence hash",
        ),
    }
    if mutation_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("inverse WAL protocol version mismatch")
    for key, expected in expected_hashes.items():
        if mutation_manifest.get(key) != expected:
            raise ValueError(f"inverse WAL {key} mismatch")

    manifest_ledger_path = Path(
        str(mutation_manifest.get("mutation_budget_ledger_path", ""))
    ).expanduser()
    if not str(mutation_manifest.get("mutation_budget_ledger_path", "")).strip():
        raise ValueError("inverse WAL mutation budget ledger path missing")
    if manifest_ledger_path.resolve() != ledger_path.resolve():
        raise ValueError("mutation budget ledger path mismatch")
    ledger_before_bytes = ledger_path.read_bytes()
    ledger_before_sha = hashlib.sha256(ledger_before_bytes).hexdigest()
    budget_rows = load_mutation_budget_ledger(ledger_path)
    reservation = mutation_manifest.get("mutation_budget_reservation")
    if not isinstance(reservation, Mapping):
        raise ValueError("inverse WAL budget reservation missing")
    reservation = dict(reservation)
    if reservation.get("evidence_hash") != expected_hashes["evidence_hash"]:
        raise ValueError("budget reservation evidence hash mismatch")
    if reservation.get("transition_hash") != expected_hashes["transition_hash"]:
        raise ValueError("budget reservation transition hash mismatch")
    if reservation.get("mutation_plan_hash") != expected_hashes["mutation_plan_hash"]:
        raise ValueError("budget reservation mutation plan hash mismatch")
    if not any(
        row.get("entry_hash") == reservation.get("entry_hash")
        for row in budget_rows
    ):
        raise ValueError("budget reservation is absent from append-only ledger")

    rollback_id = (
        "factor-governance-v2-rollback:"
        f"{expected_hashes['transition_hash'][:16]}:"
        f"{snapshot.registry_sha256[:12]}"
    )
    rollback_wal = (
        Path(args.rollback_wal).expanduser()
        if str(args.rollback_wal or "").strip()
        else None
    )
    result = rollback_factor_record_patch(
        registry_path,
        mutation_manifest,
        mutation_id=rollback_id,
        reason="FactorGovernanceProtocol v2 explicit inverse WAL rollback",
        manifest_metadata={
            "protocol_version": PROTOCOL_VERSION,
            **expected_hashes,
            "inverse_wal_sha256": wal_sha,
            "mutation_budget_ledger_path": str(ledger_path),
            "mutation_budget_ledger_sha256": ledger_before_sha,
            "monthly_budget_refunded": False,
        },
        journal_path=rollback_wal,
        write=bool(args.apply_rollback),
    )

    ledger_after_bytes = ledger_path.read_bytes()
    ledger_after_sha = hashlib.sha256(ledger_after_bytes).hexdigest()
    if ledger_after_bytes != ledger_before_bytes:
        raise RuntimeError("rollback mutated the monthly budget ledger")
    readback = load_registry_snapshot_strict(registry_path)
    if args.apply_rollback:
        if readback.registry_sha256 != result.get("after_registry_sha256"):
            raise RuntimeError("rollback registry SHA readback mismatch")
        assert rollback_wal is not None
        rollback_wal_payload = _load_wal(rollback_wal)
        if rollback_wal_payload != result:
            raise RuntimeError("rollback WAL readback mismatch")
        rollback_wal_sha = _file_sha256(rollback_wal)
    else:
        if readback.registry_sha256 != snapshot.registry_sha256:
            raise RuntimeError("dry-run rollback changed the registry")
        rollback_wal_sha = ""
    return {
        "status": "applied" if args.apply_rollback else "dry_run_ready",
        "apply_requested": bool(args.apply_rollback),
        "protocol_version": PROTOCOL_VERSION,
        **expected_hashes,
        "inverse_wal_path": str(inverse_wal_path),
        "inverse_wal_sha256": wal_sha,
        "rollback_wal_path": str(rollback_wal) if rollback_wal else "",
        "rollback_wal_sha256": rollback_wal_sha,
        "before_registry_sha256": snapshot.registry_sha256,
        "after_registry_sha256": readback.registry_sha256,
        "mutation_budget_ledger_path": str(ledger_path),
        "mutation_budget_ledger_before_sha256": ledger_before_sha,
        "mutation_budget_ledger_after_sha256": ledger_after_sha,
        "monthly_budget_refunded": False,
        "rollback_manifest": result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run_rollback(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"factor_governance_rollback_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
