#!/usr/bin/env python3
"""Verify v13.1 replay attestations without enabling production activation.

The command consumes prepared local evidence only. It does not run a provider,
LLM, broker, portfolio mutation, or factor-registry mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SEAL_ROOT = PROJECT_ROOT / "private" / "replay" / "threshold_seals"
CANONICAL_SEAL_LEDGER = CANONICAL_SEAL_ROOT / "seal_ledger.jsonl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.governance.replay_v13_1 import (  # noqa: E402
    FREEZE_EXCEPTION_CYCLE_ID,
    build_joint_replay_manifest,
    write_manifest_atomic,
)


def _read_json(path: str) -> Any:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_verified_json(path: str, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    expected = str(expected_sha256 or "").strip().lower()
    if not expected or actual_sha256 != expected:
        raise ValueError(
            f"artifact SHA mismatch for {resolved}: expected={expected or '<missing>'} "
            f"actual={actual_sha256}"
        )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain an object: {resolved}")
    payload["_artifact_readback_verified"] = True
    payload["_artifact_sha256"] = actual_sha256
    payload["_artifact_path"] = str(resolved)
    return payload


def _read_canonical_threshold_seal(
    *,
    dataset_sha256: str,
    expected_seal_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    dataset_hash = str(dataset_sha256 or "").strip().lower()
    target = CANONICAL_SEAL_ROOT / f"{dataset_hash}.json"
    if CANONICAL_SEAL_LEDGER.stat().st_mode & 0o777 != 0o600:
        raise ValueError("canonical threshold seal ledger permissions must be 0600")
    ledger_raw = CANONICAL_SEAL_LEDGER.read_bytes()
    ledger_sha256 = _sha256_bytes(ledger_raw)
    if ledger_sha256 != str(expected_ledger_sha256 or "").strip().lower():
        raise ValueError("canonical threshold seal ledger SHA mismatch")
    previous_hash = ""
    matching_entries: list[dict[str, Any]] = []
    ledger_entry_count = 0
    for line_number, raw_line in enumerate(
        ledger_raw.decode("utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"seal ledger line {line_number} is not an object")
        if payload.get("schema_version") != "myquant.holdout_threshold_seal_ledger.v2":
            raise ValueError(f"seal ledger schema mismatch at line {line_number}")
        if payload.get("freeze_exception_cycle_id") != FREEZE_EXCEPTION_CYCLE_ID:
            raise ValueError(f"seal ledger cycle mismatch at line {line_number}")
        entry_hash = str(payload.get("entry_hash") or "")
        unsigned = dict(payload)
        unsigned.pop("entry_hash", None)
        if str(payload.get("previous_entry_hash") or "") != previous_hash:
            raise ValueError(f"seal ledger chain mismatch at line {line_number}")
        if entry_hash != _canonical_sha256(unsigned):
            raise ValueError(f"seal ledger entry hash mismatch at line {line_number}")
        if str(payload.get("dataset_sha256") or "") == dataset_hash:
            matching_entries.append(payload)
        ledger_entry_count += 1
        previous_hash = entry_hash
    if len(matching_entries) != 1 or ledger_entry_count != 1:
        raise ValueError(
            "canonical threshold seal ledger must contain exactly one entry for "
            "the fixed freeze-exception cycle"
        )
    entry = matching_entries[0]
    if str(entry.get("seal_path") or "") != f"threshold_seals/{target.name}":
        raise ValueError("canonical threshold seal path mismatch")
    if target.stat().st_mode & 0o777 != 0o600:
        raise ValueError("canonical threshold seal permissions must be 0600")
    seal = _read_verified_json(str(target), expected_seal_sha256)
    if seal.get("freeze_exception_cycle_id") != FREEZE_EXCEPTION_CYCLE_ID:
        raise ValueError("canonical threshold seal cycle mismatch")
    if str(entry.get("seal_artifact_sha256") or "") != str(
        seal.get("_artifact_sha256") or ""
    ):
        raise ValueError("seal artifact SHA does not match canonical ledger")
    if str(entry.get("threshold_hash") or "") != str(
        seal.get("threshold_hash") or ""
    ):
        raise ValueError("threshold hash does not match canonical ledger")
    if str(entry.get("validation_end_date") or "") != str(
        seal.get("validation_end_date") or ""
    ):
        raise ValueError("validation end date does not match canonical ledger")
    seal["_canonical_seal_ledger_verified"] = True
    seal["_seal_ledger_sha256"] = ledger_sha256
    return seal


def _load_artifact_index(path: str, *, collection_key: str) -> Any:
    index_path = Path(path).expanduser().resolve()
    payload = _read_json(str(index_path))
    collection = payload.get(collection_key) if isinstance(payload, dict) else payload
    base = index_path.parent
    if isinstance(collection, dict):
        result: dict[str, Any] = {}
        for name, raw_ref in collection.items():
            ref = dict(raw_ref or {}) if isinstance(raw_ref, dict) else {}
            candidate = Path(str(ref.get("path") or ""))
            if not candidate.is_absolute():
                candidate = base / candidate
            result[str(name)] = _read_verified_json(
                str(candidate), str(ref.get("sha256") or "")
            )
        return result
    if isinstance(collection, list):
        result_list: list[dict[str, Any]] = []
        for raw_ref in collection:
            ref = dict(raw_ref or {}) if isinstance(raw_ref, dict) else {}
            candidate = Path(str(ref.get("path") or ""))
            if not candidate.is_absolute():
                candidate = base / candidate
            result_list.append(
                _read_verified_json(
                    str(candidate), str(ref.get("sha256") or "")
                )
            )
        return result_list
    raise ValueError(f"artifact index must contain {collection_key!r}")


def _list_payload(value: Any, key: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get(key), list):
        return list(value[key])
    raise ValueError(f"expected a JSON list or object containing {key!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trade-dates-json", required=True)
    parser.add_argument("--theme-shadow-dates-json", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--expected-threshold-seal-sha256", required=True)
    parser.add_argument("--expected-threshold-seal-ledger-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--protocol-hashes-json", required=True)
    parser.add_argument("--scenario-results-json", required=True)
    parser.add_argument("--acceptance-json", required=True)
    parser.add_argument("--expected-acceptance-sha256", required=True)
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--expected-threshold-hash", default="")
    parser.add_argument(
        "--output",
        required=True,
        help="Private 0600 manifest path; activation is never reported from stdout only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = _read_json(args.thresholds_json)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds JSON must be an object")
    seal = _read_canonical_threshold_seal(
        dataset_sha256=args.dataset_sha256,
        expected_seal_sha256=args.expected_threshold_seal_sha256,
        expected_ledger_sha256=args.expected_threshold_seal_ledger_sha256,
    )
    scenario_results = _load_artifact_index(
        args.scenario_results_json,
        collection_key="scenarios",
    )
    theme_shadow_evidence = _load_artifact_index(
        args.theme_shadow_dates_json,
        collection_key="observations",
    )
    acceptance = _read_verified_json(
        args.acceptance_json,
        args.expected_acceptance_sha256,
    )
    manifest = build_joint_replay_manifest(
        run_id=args.run_id,
        trade_dates=_list_payload(
            _read_json(args.trade_dates_json),
            "trade_dates",
        ),
        dataset_sha256=args.dataset_sha256,
        protocol_hashes=_read_json(args.protocol_hashes_json),
        scenario_results=scenario_results,
        theme_shadow_dates=theme_shadow_evidence,
        threshold_seal=seal,
        current_thresholds=thresholds,
        acceptance=acceptance,
        holdout_opened=args.open_holdout,
        expected_threshold_hash=args.expected_threshold_hash,
    )
    output_path = write_manifest_atomic(args.output, manifest)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if manifest["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
