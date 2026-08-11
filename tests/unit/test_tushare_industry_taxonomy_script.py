from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from typing import Any

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_industry_taxonomy_execution_plan,
    validate_industry_taxonomy_capture,
)
from quant_investor.intelligence_v2.sources.tushare.industry_taxonomy import (
    INDEX_CLASSIFY_FIELDS,
    OFFICIAL_PARTITIONS,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "capture_tushare_industry_taxonomy.py"
SPEC = importlib.util.spec_from_file_location("capture_tushare_industry_taxonomy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)

NOW = "2026-08-11T08:00:00Z"


def _plan_file(root: Path) -> tuple[Path, str, dict[str, Any]]:
    plan = build_industry_taxonomy_execution_plan(
        document_observed_at=NOW,
        created_at=NOW,
    )
    raw = canonical_bytes(plan)
    path = root / "plan.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), plan


def _args(plan_path: Path, plan_sha: str, output: Path, *, live: bool) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=live,
        captured_at=NOW,
        output_root=str(output),
        plan_path=str(plan_path),
        plan_sha256=plan_sha,
    )


def _rows(level: str, count: int) -> tuple[tuple[Any, ...], ...]:
    rows = []
    for index in range(count):
        parent = "0"
        if level == "L2":
            parent = f"L1I{index % 31:03d}"
        elif level == "L3":
            parent = f"L2I{index % 134:03d}"
        values = {
            "index_code": f"{level}{index:03d}.SI",
            "industry_name": f"{level}-{index}",
            "parent_code": parent,
            "level": level,
            "industry_code": f"{level}I{index:03d}",
            "is_pub": "1",
            "src": "SW2021",
        }
        rows.append(tuple(values[field] for field in INDEX_CLASSIFY_FIELDS))
    return tuple(rows)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> TushareResponse:
        self.calls.append(kwargs)
        level = kwargs["params"]["level"]
        rows = _rows(level, dict(OFFICIAL_PARTITIONS)[level])
        return TushareResponse(
            api_name="index_classify",
            request_id=f"request-{level}",
            reported_count=len(rows),
            has_more=False,
            fields=INDEX_CLASSIFY_FIELDS,
            rows=rows,
        )


def test_dry_run_has_no_client_or_writes(tmp_path: Path) -> None:
    plan_path, plan_sha, _ = _plan_file(tmp_path)
    output = tmp_path / "dry-output"
    result = SCRIPT.run(_args(plan_path, plan_sha, output, live=False))
    assert result["status"] == "DRY_RUN_VALIDATED"
    assert result["network_attempts"] == 0
    assert not output.exists()


def test_live_capture_writes_exact_private_replayable_files(tmp_path: Path) -> None:
    plan_path, plan_sha, plan = _plan_file(tmp_path)
    output = tmp_path / "live-output"
    client = FakeClient()
    result = SCRIPT.run(_args(plan_path, plan_sha, output, live=True), client=client)

    assert result["status"] == "LIVE_CAPTURE_RECORDED"
    assert result["network_attempts"] == 3
    assert len(client.calls) == 3
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {path.name for path in output.iterdir()} == {
        "capture.json",
        "plan.json",
        "summary.json",
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
        for path in output.iterdir()
    )
    capture = SCRIPT._load_plan(output / "plan.json", plan_sha)
    assert capture == plan
    import json

    capture_document = json.loads((output / "capture.json").read_bytes())
    assert validate_industry_taxonomy_capture(capture_document, plan=plan) == capture_document


def test_main_reports_only_controlled_contract_blocker(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(
        SCRIPT,
        "run",
        lambda _: (_ for _ in ()).throw(TushareContractError("taxonomy count mismatch")),
    )
    monkeypatch.setattr(SCRIPT, "parse_args", lambda: object())
    assert SCRIPT.main() == 2
    output = capsys.readouterr().out
    assert "taxonomy count mismatch" in output
    assert "Traceback" not in output
