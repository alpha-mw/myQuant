"""Resume the retained interrupted native initial run using its original installed code."""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import quant_investor


def inventory(root):
    return {
        str(p.relative_to(root)): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in root.rglob("*")
    }


def run(root: Path, dependency_path: str):
    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise AssertionError("original installed package required")
    repository = Path(receipt["repository"])
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            dependency_path,
        ]
    )
    from scripts.daily_materialization import execute_daily_recipe
    from quant_investor.market.daily_factor_loop import DailyFactorLoop

    original = json.loads((root / "initial-execute-boundary-proof.json").read_bytes())
    workspace = root / "factor-workspace"
    maintenance_root = workspace / "data/private/cn_daily_maintenance"
    before = inventory(maintenance_root)
    handoff = original["handoff_ref"]
    assert (
        hashlib.sha256((workspace / handoff["path"]).read_bytes()).hexdigest() == handoff["sha256"]
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("resume attempted maintenance/provider/loop creation")

    print("START native same-input RESUME with maintenance/providers forbidden", flush=True)
    with (
        patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden),
        patch.object(DailyFactorLoop, "__init__", forbidden),
        patch("socket.socket.connect", forbidden),
        patch("socket.create_connection", forbidden),
        patch("quant_investor.cli.unified.system_calendar_capture", forbidden),
    ):
        status = execute_daily_recipe(
            workspace=str(workspace), request_ref=original["request_ref"], synthetic=True
        )
    assert status["status"] in {"PARTIAL", "BLOCKED"}, status
    assert status.get("completion_ref") is None
    assert not calls
    assert inventory(maintenance_root) == before
    assert (
        hashlib.sha256((workspace / handoff["path"]).read_bytes()).hexdigest() == handoff["sha256"]
    )
    day_root = workspace / "results/operations/daily_production/CN/20260827"
    assert not (day_root / "completion.v1.json").exists()
    assert not (workspace / "results/prospective/CN/20260827/evidence-ledger.v1.json").exists()
    proof = {
        "synthetic": True,
        "producer_commit": receipt["commit"],
        "verification_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_resume_completed": True,
        "full_daily_closure": False,
        "maintenance_inventory_unchanged": True,
        "provider_calls": 0,
        "handoff_ref": handoff,
        "status": status,
    }
    (root / "initial-native-resume-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("PASS native resume with exact missing-source outcome", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
