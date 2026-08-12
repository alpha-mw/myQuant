#!/usr/bin/env python3
"""Build non-authoritative Factor/Fundamental Phase B1 research receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors.governance_v5._core import canonical_bytes  # noqa: E402
from quant_investor.factors.governance_v5.authority_v5_1 import (  # noqa: E402
    build_authority_matrix_v5_1,
)
from quant_investor.factors.governance_v5.contracts_v5_1 import (  # noqa: E402
    OWNER_POLICY_FIELDS,
    build_candidate_registration_v5_1,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (  # noqa: E402
    forensic_v5_1,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stable_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe input path: {path}")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise ValueError(f"unstable input bytes: {path}")
    return first


def _json(path: Path) -> dict[str, Any]:
    raw = _stable_bytes(path)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(value) is not dict:
        raise ValueError(f"JSON input root must be an object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(_stable_bytes(path)).hexdigest()


def _candidates(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = source.get("candidates")
    if type(rows) is not list:
        raise ValueError("candidate source has no candidate list")
    result: list[dict[str, Any]] = []
    for raw in rows:
        if type(raw) is not dict:
            raise ValueError("candidate source row is invalid")
        role = str(raw.get("role", ""))
        normalized_role = (
            "PRIMARY"
            if role == "primary"
            else (
                "ALTERNATE_FOR:" + role.split(":", 1)[1]
                if role.startswith("alternate_for:")
                else role
            )
        )
        result.append(
            {
                "candidate_id": raw.get("candidate_id"),
                "expression": raw.get("expression"),
                "family": raw.get("family"),
                "input_fields": raw.get("inputs"),
                "role": normalized_role,
            }
        )
    return result


def _bundle(args: argparse.Namespace) -> bytes:
    candidate_source = Path(args.candidate_source)
    implementation_source = Path(args.implementation_source)
    universe_source = Path(args.universe_source)
    calendar_source = Path(args.calendar_source)
    binding_source = Path(args.subject_binding_source)
    forensic_root = Path(args.forensic_root)
    registration = build_candidate_registration_v5_1(
        registered_at=args.created_at,
        candidates=_candidates(_json(candidate_source)),
        catalog_source_sha256=_sha(candidate_source),
        implementation_source_sha256=_sha(implementation_source),
        pit_universe_sha256=_sha(universe_source),
        exchange_calendar_sha256=_sha(calendar_source),
        missing_owner_policy_fields=OWNER_POLICY_FIELDS,
    )
    authority = build_authority_matrix_v5_1(
        registered_at=args.created_at, registration=registration
    )
    forensic = forensic_v5_1.build_fundamental_forensic_receipt_v5_1(
        produced_at=args.created_at,
        subject_id=args.subject_id,
        period=args.period,
        baseline_ann_date=args.baseline_ann_date,
        vip_ann_date=args.vip_ann_date,
        subject_binding_source_sha256=_sha(binding_source),
        expected_row_sha256=args.expected_row_sha256,
        expected_key_sha256=args.expected_key_sha256,
        summary=_json(forensic_root / "summary.json"),
        raw_row_diff=_json(forensic_root / "raw_row_diff.json"),
        raw_value_diff=_json(forensic_root / "raw_value_diff.json"),
        duplicate_diff=_json(forensic_root / "duplicate_diff.json"),
        table_evidence=_json(forensic_root / "table_evidence.json"),
    )
    epoch_plan = forensic_v5_1.build_inert_same_epoch_plan_v1(
        produced_at=args.created_at, forensic_receipt=forensic
    )
    body = {
        "authority": {
            "broker": False,
            "execution": False,
            "factor_governance_write": False,
            "mainline_authority": False,
            "order": False,
            "portfolio": False,
            "production": False,
            "provider": False,
            "research_only": True,
            "selector": False,
            "trade": False,
        },
        "candidate_registration": registration,
        "created_at": args.created_at,
        "factor_authority_matrix": authority,
        "fundamental_forensic_receipt": forensic,
        "fundamental_same_epoch_plan": epoch_plan,
        "version": "full-a-tech-closure-v1.phase-b1-receipts.v1",
    }
    body["bundle_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return canonical_bytes(body) + b"\n"


def _validate_private_target(target: Path, *, project_root: Path) -> None:
    if not target.is_absolute():
        raise ValueError("--out must be an explicit absolute path")
    if target.resolve(strict=False) != target:
        raise ValueError("--out must be a canonical path without symlink traversal")
    if (
        not project_root.is_absolute()
        or project_root.is_symlink()
        or not project_root.is_dir()
        or project_root.resolve() != project_root
    ):
        raise ValueError("--project-root must be a canonical absolute directory")
    private_root = project_root / "reports" / "full_a_tech_closure" / "private"
    try:
        target.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("--out must be inside the full-A closure private namespace") from exc
    if target == private_root:
        raise ValueError("--out must name a file inside the private namespace")


def _write_once(target: Path, payload: bytes, *, project_root: Path) -> None:
    _validate_private_target(target, project_root=project_root)
    if target.exists() or target.is_symlink():
        raise ValueError("refusing to overwrite receipt bundle")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.parent.resolve() != target.parent:
        raise ValueError("receipt parent path is unsafe")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    if target.is_symlink() or not target.is_file():
        raise ValueError("receipt output is unsafe")
    first = target.read_bytes()
    second = target.read_bytes()
    if first != payload or second != payload:
        raise ValueError("receipt stable readback mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--implementation-source", required=True)
    parser.add_argument("--universe-source", required=True)
    parser.add_argument("--calendar-source", required=True)
    parser.add_argument("--subject-binding-source", required=True)
    parser.add_argument("--forensic-root", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--baseline-ann-date", required=True)
    parser.add_argument("--vip-ann-date", required=True)
    parser.add_argument("--expected-row-sha256", action="append", required=True)
    parser.add_argument("--expected-key-sha256", action="append", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = _bundle(args)
    if not args.execute:
        print(payload.decode("utf-8"), end="")
        return 0
    _write_once(Path(args.out), payload, project_root=Path(args.project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
