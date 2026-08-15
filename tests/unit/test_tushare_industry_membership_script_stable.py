from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "capture_tushare_industry_membership.py"
SPEC = importlib.util.spec_from_file_location(
    "capture_tushare_industry_membership_stable",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    taxonomy_plan = {"kind": "taxonomy-plan"}
    taxonomy_capture = {"kind": "taxonomy-capture"}
    membership_plan = {
        "endpoint_plan": {
            "ordered_expected_partition_keyset": [
                "l3_code=850111.SI|is_new=Y",
                "l3_code=850111.SI|is_new=N",
            ],
            "planned_terminal_request_count": 2,
        },
        "membership_plan_id": "a" * 64,
    }
    return taxonomy_plan, taxonomy_capture, membership_plan


def _args(output: Path, *, live: bool, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=live,
        membership_plan_path="unused",
        membership_plan_sha256="a" * 64,
        output_root=str(output),
        resume=resume,
        taxonomy_capture_path="unused",
        taxonomy_capture_sha256="b" * 64,
        taxonomy_plan_path="unused",
        taxonomy_plan_sha256="c" * 64,
    )


def test_dry_run_does_not_create_root_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SCRIPT, "_load_inputs", lambda _: _inputs())
    output = tmp_path / "dry-output"
    result = SCRIPT.run(_args(output, live=False))
    assert result["status"] == "DRY_RUN_VALIDATED"
    assert not output.exists()


def test_live_capture_and_resume_skip_valid_exact_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SCRIPT, "_load_inputs", lambda _: _inputs())
    calls: list[int] = []

    def capture(**kwargs: Any) -> dict[str, Any]:
        ordinal = kwargs["partition_ordinal"]
        calls.append(ordinal)
        return {
            "partition_capture_id": f"{ordinal + 1:064x}",
            "partition_key": _inputs()[2]["endpoint_plan"]["ordered_expected_partition_keyset"][
                ordinal
            ],
            "partition_ordinal": ordinal,
        }

    monkeypatch.setattr(SCRIPT, "capture_industry_membership_partition", capture)
    monkeypatch.setattr(
        SCRIPT,
        "validate_industry_membership_partition_capture",
        lambda document, **_: document,
    )
    monkeypatch.setattr(
        SCRIPT,
        "build_industry_membership_capture",
        lambda **_: {"capture_id": "c" * 64},
    )
    monkeypatch.setattr(
        SCRIPT,
        "validate_industry_membership_capture",
        lambda document, **_: document,
    )
    output = tmp_path / "live-output"
    first = SCRIPT.run(_args(output, live=True), client=object())
    assert first == {
        "capture_id": "c" * 64,
        "completed_partitions": 2,
        "membership_plan_id": "a" * 64,
        "network_attempts": 2,
        "planned_partitions": 2,
        "status": "COMPLETE",
    }
    assert calls == [0, 1]

    calls.clear()
    resumed = SCRIPT.run(_args(output, live=True, resume=True), client=object())
    assert resumed["network_attempts"] == 0
    assert calls == []
