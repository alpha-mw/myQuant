"""Run native Factor prepare in the isolated release after synthetic Calendar capture."""

import json
from pathlib import Path
import sys

from quant_investor.factors.governance.factor_production_prepare import prepare_factor_production


def prepare(root: Path, *, create_market: bool) -> dict:
    # The native package is already imported from the isolated install.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs

    workspace = root / "factor-workspace"
    if create_market:
        fixture = NativeFactorInputs(workspace / "synthetic-inputs")
        arguments = fixture.day(0)
        strict_market_from_factor_inputs(workspace, arguments)
    capture = json.loads((root / "calendar-2026-08-24/fixture-result.json").read_text())["capture"]
    result = prepare_factor_production(
        workspace_root=workspace,
        market_data_root=workspace / "data",
        calendar_capture_root=capture["capture_root"],
        expected_calendar_success_sha256=capture["capture_success_file_ref"]["byte_sha256"],
    )
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            prepare(Path(sys.argv[1]), create_market="--create-market" in sys.argv[2:]), indent=2
        )
    )
