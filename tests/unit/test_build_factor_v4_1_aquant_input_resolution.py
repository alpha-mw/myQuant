from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_aquant_input_resolution_v4_1 as contract
from scripts import build_factor_v4_1_aquant_input_resolution as subject


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _inventory(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    return [
        {
            "relative_path": path.name,
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "dataset_member": True,
        }
    ]


def _mask() -> pd.DataFrame:
    return pd.DataFrame(
        True,
        index=pd.DatetimeIndex(pd.to_datetime(["2026-07-13", "2026-07-14"])),
        columns=["000001.SZ", "000002.SZ"],
        dtype=bool,
    )


def test_serving_loader_rejects_duplicate_symbol_session_keys(tmp_path: Path) -> None:
    serving = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260713", "20260713"],
            "turnover_rate": [1.0, 1.1],
        }
    ).to_parquet(serving, index=False)

    with pytest.raises(
        subject.FactorV4_1AquantInputResolutionRunnerError,
        match="duplicate serving keys",
    ):
        subject._load_serving_turnover(tmp_path, _inventory(serving), _mask())


def test_serving_loader_preserves_sparse_turnover_without_fill(tmp_path: Path) -> None:
    serving = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20260713", "20260714"],
            "turnover_rate": [1.0, 2.0],
        }
    ).to_parquet(serving, index=False)

    result = subject._load_serving_turnover(tmp_path, _inventory(serving), _mask())

    assert result.notna().sum().sum() == 2
    assert np.isnan(result.loc[pd.Timestamp("2026-07-14"), "000001.SZ"])
    assert np.isnan(result.loc[pd.Timestamp("2026-07-13"), "000002.SZ"])


def test_stable_read_rejects_hash_substitution(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        subject.FactorV4_1AquantInputResolutionRunnerError,
        match="SHA mismatch",
    ):
        subject._stable_bytes(path, expected_sha256=_digest("substituted"))


def test_binding_parsers_fail_closed_on_missing_or_duplicate_ids(tmp_path: Path) -> None:
    path = str((tmp_path / "artifact.json").resolve())
    sha = _digest("x")

    with pytest.raises(
        subject.FactorV4_1AquantInputResolutionRunnerError,
        match="exact formal/no-label/operator inventory",
    ):
        subject._parse_predecessor_bindings(
            [f"formal_catalog={path}={sha}={sha}"]
        )
    duplicate = [
        f"{contract.SOURCE_BINDING_IDS[0]}={path}={sha}",
        f"{contract.SOURCE_BINDING_IDS[0]}={path}={sha}",
    ]
    with pytest.raises(
        subject.FactorV4_1AquantInputResolutionRunnerError,
        match="duplicate source binding",
    ):
        subject._parse_file_bindings(
            duplicate,
            expected_ids=contract.SOURCE_BINDING_IDS,
            label="source binding",
        )


def test_run_publishes_exactly_once_then_performs_independent_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "reports" / "factor_governance" / "private" / "v4_1_aquant_input_resolution"
    table_root = tmp_path / "table"
    serving_root = tmp_path / "serving"
    table_root.mkdir()
    serving_root.mkdir()
    paths = {
        binding_id: tmp_path / f"{binding_id}.bin"
        for binding_id in contract.SOURCE_BINDING_IDS
        if binding_id not in {"table_inventory", "serving_inventory"}
    }
    paths["table_inventory"] = table_root
    paths["serving_inventory"] = serving_root
    predecessor = [
        f"{binding_id}={tmp_path / (binding_id + '.json')}={_digest(binding_id)}={_digest('s-' + binding_id)}"
        for binding_id in contract.PREDECESSOR_BINDING_IDS
    ]
    code = [
        f"{binding_id}={tmp_path / (binding_id + '.py')}={_digest(binding_id)}"
        for binding_id in contract.CODE_BINDING_IDS
    ]
    source = [
        f"{binding_id}={paths[binding_id]}={_digest(binding_id)}"
        for binding_id in contract.SOURCE_BINDING_IDS
    ]
    args = argparse.Namespace(
        private_root=str(private_root),
        run_id="fixture_run",
        cycle_id=contract.EXPECTED_CYCLE_ID,
        predecessor_binding=predecessor,
        code_binding=code,
        source_binding=source,
    )

    monkeypatch.setattr(
        subject,
        "_read_predecessors",
        lambda _bindings: (
            [
                {
                    "binding_id": item,
                    "path": f"/private/{item}",
                    "byte_sha256": _digest(item),
                    "semantic_sha256": _digest("s-" + item),
                }
                for item in contract.PREDECESSOR_BINDING_IDS
            ],
            {"bound_ideas": []},
        ),
    )
    monkeypatch.setattr(subject, "_read_code_bindings", lambda _bindings: [])
    json_values = {
        "latest_pointer": {
            "status": "OK",
            "snapshot_id": "snapshot",
            "manifest_path": str(paths["snapshot_manifest"]),
        },
        "snapshot_manifest": {
            "status": "OK",
            "snapshot_id": "snapshot",
            "table_root": str(table_root),
            "derived_serving_root": str(serving_root),
        },
        "pit_generation_manifest": {},
        "pit_components": {},
        "fundamental_pointer": {
            "generation_id": "generation",
        },
        "fundamental_generation_manifest": {},
    }
    monkeypatch.setattr(
        subject,
        "_read_json_source",
        lambda binding: (
            json_values[Path(binding["path"]).stem],
            _digest("semantic-" + Path(binding["path"]).stem),
        ),
    )
    inventory = [
        {
            "relative_path": "part.parquet",
            "byte_sha256": _digest("part"),
            "size_bytes": 1,
            "dataset_member": True,
        }
    ]
    monkeypatch.setattr(
        subject,
        "_inventory_root",
        lambda root, expected_sha256: (inventory, expected_sha256),
    )
    monkeypatch.setattr(subject, "_stable_bytes", lambda *args, **kwargs: b"x")
    monkeypatch.setattr(
        subject,
        "_table_sessions",
        lambda *_args: pd.DatetimeIndex(pd.to_datetime(["2026-07-13", "2026-07-14"])),
    )
    mask = _mask()
    monkeypatch.setattr(subject, "_build_eligibility_mask", lambda **_kwargs: mask)
    base = pd.DataFrame(1.0, index=mask.index, columns=mask.columns)
    monkeypatch.setattr(
        subject,
        "_load_market_matrices",
        lambda *_args: {"close": base, "vwap": base},
    )
    monkeypatch.setattr(subject, "_load_serving_turnover", lambda *_args: base)
    monkeypatch.setattr(
        subject,
        "_load_fundamentals",
        lambda **_kwargs: {field: base for field in subject.FUNDAMENTAL_FIELDS},
    )
    monkeypatch.setattr(subject, "_source_semantic", lambda *_args: _digest("file"))
    protected = [
        {
            "binding_id": binding_id,
            "path": str(paths[binding_id]),
            "before_sha256": _digest(binding_id),
            "after_sha256": _digest(binding_id),
        }
        for binding_id in contract.SOURCE_BINDING_IDS
    ]
    artifact = {
        "resolution_profile": "fully_resolved",
        "resolution_semantic_sha256": _digest("artifact-semantic"),
        "protected_stability": protected,
    }
    monkeypatch.setattr(
        contract,
        "build_input_resolution_artifact_v4_1",
        lambda **_kwargs: artifact,
    )
    bundle_contract = object()
    monkeypatch.setattr(
        contract,
        "build_private_bundle_contract_v4_1",
        lambda **_kwargs: bundle_contract,
    )
    calls = {"publish": 0, "readback": 0, "revalidated": 0}

    def fake_publish(**kwargs):
        calls["publish"] += 1
        kwargs["revalidate_inputs"]()
        calls["revalidated"] += 1
        return {"bundle_path": str(private_root / "fixture_run")}

    def fake_readback(bundle_path, *, contract):
        calls["readback"] += 1
        assert contract is bundle_contract
        return {
            "bundle_path": bundle_path,
            "artifact_descriptors": {
                subject.contract.ARTIFACT_FILENAME: {
                    "absolute_path": f"{bundle_path}/{subject.contract.ARTIFACT_FILENAME}",
                    "byte_sha256": _digest("artifact-byte"),
                    "mode": 0o600,
                },
                subject.contract.READBACK_FILENAME: {
                    "absolute_path": f"{bundle_path}/{subject.contract.READBACK_FILENAME}",
                    "byte_sha256": _digest("readback-byte"),
                    "mode": 0o600,
                },
            },
        }

    monkeypatch.setattr(subject.private_io, "publish_private_bundle", fake_publish)
    monkeypatch.setattr(subject.private_io, "readback_private_bundle", fake_readback)

    result = subject.run(args)

    assert calls == {"publish": 1, "readback": 1, "revalidated": 1}
    assert result["resolution_profile"] == "fully_resolved"
    assert result["artifact_mode"] == 0o600
    assert result["readback_mode"] == 0o600
    assert result["new_risk_authorized"] is False
