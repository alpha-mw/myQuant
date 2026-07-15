from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.market import macro_mart
from tests.helpers.macro_fixture import bind_macro_generation


def _row() -> dict[str, object]:
    return {
        "trade_date": "2024-05-10",
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2024-05-10T08:00:00+00:00",
    }


def test_reader_rejects_fully_removed_formula_universe_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    catalog_path, _, manifest_path, _ = bind_macro_generation(
        root,
        generation_id="formula-required",
        row=_row(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("market_formula_universe")
    manifest.pop("market_formula_universe_sha256")
    provenance = manifest["primary_provenance"]
    provenance.pop("market_formula_universe_sha256")
    provenance.pop("envelope_sha256")
    provenance["envelope_sha256"] = macro_mart._canonical_json_sha256(
        provenance
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["tables"]["macro_daily"][
        "generation_manifest_sha256"
    ] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    catalog_path.write_text(
        json.dumps(catalog, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        macro_mart.MacroMartPromotionError,
        match="macro_primary_formula_universe_evidence_invalid",
    ):
        macro_mart.read_macro_mart(data_root=root)
