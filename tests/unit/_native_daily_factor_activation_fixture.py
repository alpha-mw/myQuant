"""Exercise native Factor activation/observations/core handoff in isolated fixture."""

import hashlib
import json
from pathlib import Path
import sys

from quant_investor.cli.unified import (
    factor_production_activate,
    factor_production_observe,
    factor_production_rollover,
)
from quant_investor.factors.production_authority import verify_factor_production
from quant_investor.intelligence.storage import publish_theme_policy_v2
from quant_investor.operations.core_pool import publish_core_pool
from quant_investor.contracts import canonical_json_bytes


def run(root: Path, phase: str) -> dict:
    workspace = root / "factor-workspace"
    capture = json.loads((root / "calendar-2026-08-24/fixture-result.json").read_text())["capture"]
    if phase == "activate":
        result = factor_production_activate(
            workspace_root=str(workspace),
            market_data_root=str(workspace / "data"),
            calendar_capture_root=capture["capture_root"],
            expected_calendar_success_sha256=capture["capture_success_file_ref"]["byte_sha256"],
            expected_empty=True,
        )
    elif phase == "rollover":
        successor = json.loads((root / "calendar-2026-08-25/fixture-result.json").read_text())[
            "capture"
        ]
        maintenance = json.loads((root / "successor-maintenance-verification.json").read_text())[
            "native_core_checkpoint"
        ]
        original = json.loads((root / "factor-activate-result.json").read_text())
        result = factor_production_rollover(
            workspace_root=str(workspace),
            market_data_root=str(workspace / "data"),
            calendar_capture_root=successor["capture_root"],
            expected_calendar_success_sha256=successor["capture_success_file_ref"]["byte_sha256"],
            maintenance_receipt=maintenance["path"],
            expected_maintenance_receipt_sha256=maintenance["sha256"],
            expected_current_pointer_sha256=original["factor_pointer_byte_sha256"],
        )
    elif phase == "observe":
        result = factor_production_observe(workspace_root=str(workspace))
    elif phase == "core":
        state = verify_factor_production(str(workspace))
        publish_theme_policy_v2(workspace)
        release = json.loads((root / "release-input.json").read_text())["deployed_release"]
        path = workspace / "fixtures/release.json"
        path.parent.mkdir(exist_ok=True)
        raw = canonical_json_bytes(release)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("fixture release bytes changed")
        if not path.exists():
            path.write_bytes(raw)
            path.chmod(0o600)
        result = publish_core_pool(
            workspace=str(workspace),
            trade_date=state["as_of"],
            factor_pointer_sha256=state["factor_pointer_byte_sha256"],
            release_ref={
                "path": "fixtures/release.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
    else:
        raise ValueError("unknown fixture phase")
    (root / ("factor-" + phase + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), sys.argv[2]), indent=2))
