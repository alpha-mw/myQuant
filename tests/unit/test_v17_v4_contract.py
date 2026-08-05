from __future__ import annotations

import json
from pathlib import Path

from quant_investor.v17_v4_contract import (
    MAINLINE_AUTHORITY,
    PRODUCTION_AUTHORITY,
    RESEARCH_ONLY,
    verify_package,
)
from quant_investor.v17_v4_contract.resources import load_package_manifest


CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "quant_investor/v17_v4_contract"
def test_contract_has_only_research_authority() -> None:
    assert RESEARCH_ONLY is True
    assert MAINLINE_AUTHORITY is False
    assert PRODUCTION_AUTHORITY is False

    authority_schema = json.loads(
        (CONTRACT_ROOT / "schemas/authority.v1.schema.json").read_text()
    )["$defs"]["authority"]
    assert authority_schema["required"] == [
        "broker",
        "execution",
        "mainline_authority",
        "order",
        "production",
        "research_only",
        "trade",
    ]
    assert authority_schema["properties"]["research_only"]["const"] is True
    for field in ("broker", "execution", "mainline_authority", "order", "production", "trade"):
        assert authority_schema["properties"][field]["const"] is False


def test_packaged_contract_contains_no_retired_routing_assets() -> None:
    manifest = load_package_manifest()
    relative_paths = [row["relative_path"] for row in manifest["assets"]]
    assert relative_paths == sorted(relative_paths)
    assert set(relative_paths) == {
        path.relative_to(CONTRACT_ROOT).as_posix()
        for root in (CONTRACT_ROOT / "resources", CONTRACT_ROOT / "schemas")
        for path in root.glob("*.json")
        if path.name != "package_manifest.v1.json"
    }
    assert verify_package()
