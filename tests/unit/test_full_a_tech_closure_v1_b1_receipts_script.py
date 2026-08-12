from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from quant_investor.intelligence_v2._core import canonical_bytes

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("balancesheet", "cashflow", "daily_basic", "fina_indicator", "forecast", "income")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True), encoding="utf-8")


def _forensic_inputs(root: Path) -> None:
    root.mkdir()
    row_diff = {table: [] for table in TABLES}
    row_diff["fina_indicator"] = [
        {"baseline_count": 1, "row_sha256": "a" * 64, "vip_count": 0},
        {"baseline_count": 0, "row_sha256": "b" * 64, "vip_count": 1},
    ]
    value_diff = {table: [] for table in TABLES}
    value_diff["fina_indicator"] = [
        {
            "baseline_winner_sha256": "a" * 64,
            "key_sha256": "c" * 64,
            "vip_winner_sha256": None,
        },
        {
            "baseline_winner_sha256": None,
            "key_sha256": "d" * 64,
            "vip_winner_sha256": "b" * 64,
        },
    ]
    duplicate = {
        table: {"baseline_duplicate_row_count": 0, "vip_duplicate_row_count": 0} for table in TABLES
    }
    table_evidence = {table: {"evidence": table} for table in TABLES}
    files = {
        "duplicate_diff.json": duplicate,
        "raw_row_diff.json": row_diff,
        "raw_value_diff.json": value_diff,
        "table_evidence.json": table_evidence,
    }
    summary = {
        "checkpoint_execution_bundle_sha256": "e" * 64,
        "diff_counts": {
            table: {
                "row": 2 if table == "fina_indicator" else 0,
                "value": 2 if table == "fina_indicator" else 0,
            }
            for table in TABLES
        },
        "file_sha256": {
            name: hashlib.sha256(canonical_bytes(value)).hexdigest()
            for name, value in files.items()
        },
        "implementation_commit": "f" * 40,
        "package_sha256": "1" * 64,
        "passed": False,
        "physical_receipt_count": 11471,
        "status": "RECONCILIATION_BLOCKED",
        "transport_calls": 0,
        "version": "myquant.r7.fundamental-v5-reconciliation-forensic.v1",
    }
    for name, value in {"summary.json": summary, **files}.items():
        _write_json(root / name, value)


def test_script_writes_once_with_stable_non_authoritative_readback(tmp_path: Path) -> None:
    candidate_source = tmp_path / "candidate.json"
    _write_json(
        candidate_source,
        {
            "candidates": [
                {
                    "candidate_id": "quality_primary",
                    "expression": "cs_rank(fin_roe)",
                    "family": "quality",
                    "inputs": ["fin_roe"],
                    "role": "primary",
                }
            ]
        },
    )
    ordinary_inputs = []
    for name in ("implementation.py", "universe.json", "calendar.json", "binding.md"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        ordinary_inputs.append(path)
    forensic_root = tmp_path / "forensic"
    _forensic_inputs(forensic_root)
    output = tmp_path / "reports" / "full_a_tech_closure" / "private" / "receipts.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_full_a_tech_closure_v1_b1_receipts.py"),
        "--created-at",
        "2026-08-13T02:00:00Z",
        "--candidate-source",
        str(candidate_source),
        "--implementation-source",
        str(ordinary_inputs[0]),
        "--universe-source",
        str(ordinary_inputs[1]),
        "--calendar-source",
        str(ordinary_inputs[2]),
        "--subject-binding-source",
        str(ordinary_inputs[3]),
        "--forensic-root",
        str(forensic_root),
        "--subject-id",
        "920188.BJ",
        "--period",
        "20250630",
        "--baseline-ann-date",
        "20251205",
        "--vip-ann-date",
        "20260312",
        "--expected-row-sha256",
        "a" * 64,
        "--expected-row-sha256",
        "b" * 64,
        "--expected-key-sha256",
        "c" * 64,
        "--expected-key-sha256",
        "d" * 64,
        "--project-root",
        str(tmp_path),
        "--out",
        str(output),
        "--execute",
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    assert output.read_bytes() == first_bytes
    bundle = json.loads(first_bytes)
    assert bundle["candidate_registration"]["admission_eligible"] is False
    assert bundle["fundamental_forensic_receipt"]["transport_calls"] == 0
    assert bundle["fundamental_same_epoch_plan"]["execution_authorized"] is False
    assert bundle["authority"]["provider"] is False

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
    assert output.read_bytes() == first_bytes

    outside = tmp_path / "results" / "factor_governance" / "receipt.json"
    outside_command = command.copy()
    outside_command[outside_command.index(str(output))] = str(outside)
    rejected = subprocess.run(
        outside_command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert "must be inside the full-A closure private namespace" in rejected.stderr
    assert not outside.exists()

    traversal = output.parent / ".." / ".." / "outside" / "receipt.json"
    traversal_command = command.copy()
    traversal_command[traversal_command.index(str(output))] = str(traversal)
    traversal_rejected = subprocess.run(
        traversal_command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert traversal_rejected.returncode != 0
    assert "must be a canonical path" in traversal_rejected.stderr
    assert not traversal.resolve().exists()
