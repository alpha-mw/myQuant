from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_investor.macro.contracts import MacroObservation
from quant_investor.macro.observer import (
    build_macro_observer,
    load_macro_observation_generation,
)
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
)


def _row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": "2024-04-30",
        "release_at": "2024-05-01T06:00:00+00:00",
        "available_at": "2024-05-01T06:00:00+00:00",
        "vintage_id": "initial",
        "value": 50.2,
        "unit": "index",
        "frequency": "monthly",
        "source_system": "nbs_official",
        "source_record_id": "nbs:pmi:2024-04",
        "source_url": "https://www.stats.gov.cn/data/pmi",
        "fetched_at": "2024-05-01T06:05:00+00:00",
        "quality_status": "pass",
    }
    payload.update(overrides)
    return payload


def test_missing_store_read_is_side_effect_free_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-observations"

    with pytest.raises(
        MacroObservationStoreError,
        match="macro_observation_root_missing",
    ):
        load_observations(root)

    assert not root.exists()


def test_canonical_loader_requires_pointer_manifest_and_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    root.mkdir()

    with pytest.raises(
        MacroObservationStoreError,
        match="macro_observation_pointer_missing",
    ):
        load_observations(root)


def test_standalone_loader_requires_explicit_offline_flag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"observations": [_row()]}), encoding="utf-8")

    with pytest.raises(ValueError, match="macro_standalone_observations_disabled"):
        load_macro_observation_generation(path)

    rows, generation = load_macro_observation_generation(
        path,
        allow_standalone_offline=True,
    )
    assert len(rows) == 1
    assert generation == {}


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"source_system": "unknown"}, "source_system_unsupported"),
        ({"source_record_id": ""}, "source_record_id_missing"),
        ({"source_url": "http://www.stats.gov.cn/data"}, "source_url_https_required"),
        (
            {"source_url": "https://user:password@www.stats.gov.cn/data"},
            "source_url_userinfo_rejected",
        ),
        (
            {"source_url": "https://www.stats.gov.cn:8443/data"},
            "source_url_port_rejected",
        ),
        (
            {"source_url": "https://evil.example/data"},
            "source_url_issuer_mismatch",
        ),
        (
            {"source_url": "https://www.stats.gov.cn/data?token=secret"},
            "source_url_sensitive_query_rejected",
        ),
    ],
)
def test_observation_source_lineage_is_strict(
    overrides: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        MacroObservation.from_mapping(_row(**overrides))


def test_normalized_source_url_is_content_hash_bound() -> None:
    first = MacroObservation.from_mapping(_row())
    second = MacroObservation.from_mapping(
        _row(source_url="https://www.stats.gov.cn/data/pmi-revised")
    )

    assert first.source_url == "https://www.stats.gov.cn/data/pmi"
    assert first.content_hash != second.content_hash
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        MacroObservation.from_mapping(
            _row(
                source_url="https://www.stats.gov.cn/data/pmi-revised",
                content_hash=first.content_hash,
            )
        )


def test_local_strict_parquet_source_is_logical_and_hash_bound() -> None:
    row = _row(
        indicator_id="market.breadth",
        period_end="2024-04-30",
        release_at="2024-04-30T15:00:00+08:00",
        available_at="2024-04-30T15:00:00+08:00",
        value=52.0,
        unit="%",
        frequency="daily",
        source_system="local_strict_parquet",
        source_record_id="cn:snapshot-1:20240430:market-breadth",
        source_url="local://strict-parquet/cn/snapshots/snapshot-1/bars",
        fetched_at="2024-05-01T00:00:00+08:00",
    )

    observation = MacroObservation.from_mapping(row)

    assert observation.source_url == (
        "local://strict-parquet/cn/snapshots/snapshot-1/bars"
    )


@pytest.mark.parametrize(
    ("source_url", "error"),
    [
        ("https://localhost/cn/snapshot", "source_url_local_scheme_required"),
        ("local://other/cn/snapshot", "source_url_issuer_mismatch"),
        ("local://strict-parquet/us/snapshot", "source_url_local_path_invalid"),
        ("local://strict-parquet/cn/../secret", "source_url_local_path_invalid"),
        ("local://strict-parquet/cn/snapshot?token=x", "source_url_local_query_rejected"),
    ],
)
def test_local_strict_parquet_source_rejects_unsafe_lineage(
    source_url: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        MacroObservation.from_mapping(
            _row(
                indicator_id="market.breadth",
                period_end="2024-04-30",
                release_at="2024-04-30T15:00:00+08:00",
                available_at="2024-04-30T15:00:00+08:00",
                value=52.0,
                unit="%",
                frequency="daily",
                source_system="local_strict_parquet",
                source_record_id="cn:snapshot-1:20240430:market-breadth",
                source_url=source_url,
                fetched_at="2024-05-01T00:00:00+08:00",
            )
        )


def test_every_persisted_observer_artifact_is_non_production(
    tmp_path: Path,
) -> None:
    result = build_macro_observer(
        [_row()],
        as_of="2024-05-10",
        enabled=True,
        kill_switch=False,
        persist=True,
        output_root=tmp_path / "results",
    )

    assert result["production_eligible"] is False
    assert result["applied"] is False
    for name in ("snapshot", "readiness", "manifest"):
        payload = json.loads(Path(result["artifacts"][name]).read_text(encoding="utf-8"))
        assert payload["production_eligible"] is False
        assert payload["applied"] is False
    report = Path(result["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "Production eligible: `false`" in report
    assert "Production applied: `false`" in report
