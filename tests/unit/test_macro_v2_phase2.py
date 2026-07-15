from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.macro.contracts import MacroObservation
from quant_investor.macro.providers import (
    MacroFetchRequest,
    OfficialMacroProvider,
    ProviderFetchResult,
    TushareMacroProvider,
    fetch_official_first,
    maintain_macro_observations,
)
from quant_investor.macro.replay import run_macro_replay
from quant_investor.macro.registry import definition_for
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
    pointer_sha256,
    publish_observations,
)
from quant_investor.macro import store as observation_store


def _row(
    *,
    period: str = "2024-04-30",
    available: str = "2024-05-01T06:00:00+00:00",
    value: float = 10.0,
    source: str = "nbs_official",
    vintage: str = "initial",
) -> dict[str, object]:
    source_url = (
        "https://www.stats.gov.cn/fixture"
        if source == "nbs_official"
        else "https://tushare.pro/document/fixture"
    )
    return {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": period,
        "release_at": available,
        "available_at": available,
        "vintage_id": vintage,
        "value": value,
        "unit": "index",
        "frequency": "monthly",
        "source_system": source,
        "source_record_id": f"{source}:{period}:{vintage}",
        "source_url": source_url,
        "fetched_at": available,
        "quality_status": "pass",
    }


def test_semantic_content_hash_is_recomputed_and_mismatch_rejected():
    observation = MacroObservation.from_mapping(_row())
    assert len(observation.content_hash) == 64
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        MacroObservation.from_mapping({**_row(), "content_hash": "d" * 64})


@pytest.mark.parametrize(("first_status", "second_status"), [("fail", "pass"), ("pass", "fail")])
def test_quality_status_is_hash_bound_and_revisions_append(
    tmp_path: Path,
    first_status: str,
    second_status: str,
):
    root = tmp_path / "observations"
    first = MacroObservation.from_mapping({**_row(), "quality_status": first_status})
    second = MacroObservation.from_mapping(
        {**_row(vintage="quality-revision"), "quality_status": second_status}
    )
    assert first.content_hash != second.content_hash
    publish_observations([first], root=root, run_id="g1")
    result = publish_observations([second], root=root, run_id="g2")
    rows, _ = load_observations(root)
    assert result["promoted"] is True
    assert {row["quality_status"] for row in rows} == {first_status, second_status}


def test_forged_dataclass_cannot_bypass_store_validation(tmp_path: Path):
    valid = MacroObservation.from_mapping(_row())
    forged = MacroObservation(**{**valid.to_dict(), "unit": "wrong", "content_hash": "f" * 64})
    with pytest.raises(ValueError, match="unit_mismatch|content_hash_mismatch"):
        publish_observations([forged], root=tmp_path / "observations", run_id="forged")


def test_observation_store_append_idempotence_and_pointer_cas(tmp_path: Path):
    root = tmp_path / "observations"
    first = publish_observations([_row()], root=root, run_id="g1")
    first_pointer = pointer_sha256(root)
    assert first["promoted"] is True
    assert first["observer_only"] is True
    assert first["production_eligible"] is False
    assert first["applied"] is False
    assert int(oct((root / "_latest.json").stat().st_mode & 0o777), 8) == 0o600
    pointer_payload = json.loads(
        (root / "_latest.json").read_text(encoding="utf-8")
    )
    manifest_payload = json.loads(
        (root / pointer_payload["manifest_path"]).read_text(encoding="utf-8")
    )
    for payload in (pointer_payload, manifest_payload):
        assert payload["observer_only"] is True
        assert payload["production_eligible"] is False
        assert payload["applied"] is False

    duplicate = publish_observations([_row()], root=root, run_id="unused")
    assert duplicate["promoted"] is False
    assert pointer_sha256(root) == first_pointer

    with pytest.raises(MacroObservationStoreError, match="pointer_cas_mismatch"):
        publish_observations(
            [_row(period="2024-05-31", available="2024-06-01T06:00:00+00:00")],
            root=root,
            run_id="g2",
            expected_pointer_sha256="0" * 64,
        )
    assert pointer_sha256(root) == first_pointer

    second = publish_observations(
        [_row(period="2024-05-31", available="2024-06-01T06:00:00+00:00")],
        root=root,
        run_id="g2",
        expected_pointer_sha256=first_pointer,
    )
    rows, pinned = load_observations(root, generation_id="g1")
    assert second["row_count"] == 2
    assert len(rows) == 1
    assert pinned["generation_id"] == "g1"


def test_observation_store_conflict_and_corruption_leave_no_silent_fallback(tmp_path: Path):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    before = pointer_sha256(root)
    with pytest.raises(MacroObservationStoreError, match="conflicting_vintage"):
        publish_observations([_row(value=99.0)], root=root, run_id="g2")
    assert pointer_sha256(root) == before
    pointer = json.loads((root / "_latest.json").read_text(encoding="utf-8"))
    table = root / pointer["table_path"]
    table.write_bytes(table.read_bytes() + b"corrupt")
    with pytest.raises(MacroObservationStoreError, match="hash_mismatch"):
        load_observations(root)


def test_observation_store_rejects_generation_parent_symlink(tmp_path: Path):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    generation = root / "_generations" / "g1"
    moved = root / "_generations" / "moved"
    generation.rename(moved)
    generation.symlink_to(moved, target_is_directory=True)
    with pytest.raises(MacroObservationStoreError, match="symlink_rejected"):
        load_observations(root)


def test_observation_store_rejects_root_ancestor_symlink(tmp_path: Path):
    real_parent = tmp_path / "real"
    root = real_parent / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        MacroObservationStoreError,
        match="macro_observation_root_unsafe",
    ):
        load_observations(alias / "observations")
    with pytest.raises(
        MacroObservationStoreError,
        match="macro_observation_root_unsafe",
    ):
        pointer_sha256(alias / "observations")


def test_observation_store_parses_the_same_verified_table_bytes(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    pointer = json.loads((root / "_latest.json").read_text(encoding="utf-8"))
    table = root / pointer["table_path"]
    original_read = observation_store.pd.read_parquet
    mutated = False

    def mutate_path_then_parse(source, *args, **kwargs):
        nonlocal mutated
        if isinstance(source, io.BytesIO) and not mutated:
            table.write_bytes(b"corrupt-after-verified-read")
            mutated = True
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(
        observation_store.pd,
        "read_parquet",
        mutate_path_then_parse,
    )

    rows, _generation = load_observations(root)

    assert mutated is True
    assert rows[0]["value"] == 10.0
    with pytest.raises(
        MacroObservationStoreError,
        match="generation_hash_mismatch",
    ):
        load_observations(root)


@pytest.mark.parametrize("generation_id", [".", ".."])
def test_observation_pointer_rejects_dot_generation_ids_before_resolution(
    tmp_path: Path,
    generation_id: str,
):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    pointer_path = root / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_id"] = generation_id
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(MacroObservationStoreError, match="run_id_unsafe"):
        load_observations(root)


def test_observation_pointer_rejects_invalid_observer_flags(
    tmp_path: Path,
):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    pointer_path = root / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["observer_only"] = False
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        MacroObservationStoreError,
        match="pointer_observer_flags_invalid",
    ):
        load_observations(root)


def test_observation_manifest_rejects_invalid_observer_flags(
    tmp_path: Path,
):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    pointer_path = root / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = root / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["production_eligible"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        MacroObservationStoreError,
        match="manifest_observer_flags_invalid",
    ):
        load_observations(root)


def test_observation_store_concurrent_cas_allows_one_promotion(tmp_path: Path):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    expected = pointer_sha256(root)

    def promote(run_id: str, period: str, available: str) -> str:
        try:
            publish_observations(
                [_row(period=period, available=available)],
                root=root,
                run_id=run_id,
                expected_pointer_sha256=expected,
            )
            return "promoted"
        except MacroObservationStoreError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda args: promote(*args),
                [
                    ("g2", "2024-05-31", "2024-06-01T06:00:00+00:00"),
                    ("g3", "2024-06-30", "2024-07-01T06:00:00+00:00"),
                ],
            )
        )
    assert outcomes.count("promoted") == 1
    assert sum("pointer_cas_mismatch" in item for item in outcomes) == 1


@pytest.mark.parametrize("failure_point", ["staging", "generations"])
def test_observation_store_fsync_failure_preserves_pointer(
    tmp_path: Path,
    monkeypatch,
    failure_point: str,
):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    before = pointer_sha256(root)
    original = observation_store._fsync_directory

    def fail_selected(path: Path) -> None:
        if failure_point == "staging" and path.name.startswith(".g2."):
            raise OSError("injected_staging_fsync_failure")
        if failure_point == "generations" and path.name == "_generations":
            raise OSError("injected_generations_fsync_failure")
        original(path)

    monkeypatch.setattr(observation_store, "_fsync_directory", fail_selected)
    with pytest.raises(OSError, match="injected_"):
        publish_observations(
            [_row(period="2024-05-31", available="2024-06-01T06:00:00+00:00")],
            root=root,
            run_id="g2",
        )
    assert pointer_sha256(root) == before
    assert not (root / "_generations" / "g2").exists()


def test_provider_is_official_first_and_fallback_is_explicit():
    calls: list[tuple[str, str]] = []

    def official(indicator_id: str, request: MacroFetchRequest):
        calls.append(("official", indicator_id))
        return [_row()] if indicator_id == "cn.pmi_manufacturing" else []

    def tushare(indicator_id: str, request: MacroFetchRequest):
        calls.append(("tushare", indicator_id))
        definition = definition_for(indicator_id)
        return [
            {
                **_row(source="tushare_fallback"),
                "indicator_id": indicator_id,
                "unit": definition.unit if definition else "",
            }
        ]

    request = MacroFetchRequest("CN", "2024-05-10", ("cn.pmi_manufacturing",))
    result = fetch_official_first(
        request,
        official_provider=OfficialMacroProvider(official),
        tushare_provider=TushareMacroProvider(tushare),
        allow_tushare_fallback=True,
    )
    assert result.status == "OK"
    assert calls == [("official", "cn.pmi_manufacturing")]
    assert result.observations[0].source_system == "nbs_official"

    missing_request = MacroFetchRequest("CN", "2024-05-10", ("cn.cpi_yoy",))
    fallback = fetch_official_first(
        missing_request,
        official_provider=OfficialMacroProvider(official),
        tushare_provider=TushareMacroProvider(tushare),
        allow_tushare_fallback=True,
    )
    assert fallback.observations[0].source_system == "tushare_fallback"


def test_provider_rejects_spoofed_tushare_source():
    def spoofed(indicator_id: str, request: MacroFetchRequest):
        return [_row(source="notushare")]

    result = TushareMacroProvider(spoofed).fetch(
        MacroFetchRequest("CN", "2024-05-10", ("cn.pmi_manufacturing",))
    )
    assert result.status == "blocked"
    assert result.observations == ()
    assert any("provider_source_provenance_mismatch" in item for item in result.blockers)


def test_provider_composition_revalidates_custom_official_source():
    class SpoofedOfficialProvider:
        provider_id = "custom-official"

        def fetch(self, request: MacroFetchRequest):
            return ProviderFetchResult(
                observations=(
                    MacroObservation.from_mapping(
                        _row(source="tushare_fallback")
                    ),
                ),
                provider_manifest={"provider_id": self.provider_id},
            )

    result = fetch_official_first(
        MacroFetchRequest(
            "CN",
            "2024-05-10",
            ("cn.pmi_manufacturing",),
        ),
        official_provider=SpoofedOfficialProvider(),
    )

    assert result.status == "blocked"
    assert result.observations == ()
    assert any(
        "provider_source_provenance_mismatch:official"
        in item
        for item in result.blockers
    )


def test_live_without_injected_transport_is_blocked_and_writes_nothing(tmp_path: Path):
    root = tmp_path / "observations"
    result = maintain_macro_observations(
        market="CN",
        as_of="2024-05-10",
        indicator_ids=["cn.pmi_manufacturing"],
        root=str(root),
        run_id="blocked",
        allow_live=True,
        allow_tushare_fallback=True,
    )
    assert result["promoted"] is False
    assert result["status"] == "blocked"
    assert not (root / "_latest.json").exists()


def test_malformed_partial_provider_result_does_not_advance_pointer(tmp_path: Path):
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    before = pointer_sha256(root)

    def partial(indicator_id: str, request: MacroFetchRequest):
        if indicator_id == "cn.pmi_manufacturing":
            return [_row(period="2024-05-31", available="2024-06-01T06:00:00+00:00")]
        return [{**_row(), "indicator_id": indicator_id, "unit": "wrong"}]

    result = maintain_macro_observations(
        market="CN",
        as_of="2024-06-10",
        indicator_ids=["cn.pmi_manufacturing", "cn.cpi_yoy"],
        root=str(root),
        run_id="g2",
        allow_live=True,
        official_provider=OfficialMacroProvider(partial),
    )
    assert result["status"] == "blocked"
    assert result["promoted"] is False
    assert pointer_sha256(root) == before


def test_official_observation_beats_later_tushare_for_same_period():
    rows = [
        _row(period="2024-02-29", available="2024-03-01T06:00:00+00:00", value=8.0),
        _row(period="2024-03-31", available="2024-04-01T06:00:00+00:00", value=9.0),
        _row(period="2024-04-30", available="2024-05-01T06:00:00+00:00", value=10.0),
        _row(
            period="2024-04-30",
            available="2024-05-02T06:00:00+00:00",
            value=99.0,
            source="tushare_fallback",
        ),
    ]
    snapshot = build_macro_snapshot(rows, as_of="2024-05-10")
    assert snapshot.source_lineage["cn.pmi_manufacturing"]["source_system"] == "nbs_official"


def test_store_keeps_same_time_official_and_fallback_evidence(tmp_path: Path):
    official = _row(value=10.0, source="nbs_official")
    fallback = _row(value=99.0, source="tushare_fallback")
    root = tmp_path / "observations"
    result = publish_observations([official, fallback], root=root, run_id="g1")
    rows, _ = load_observations(root)
    snapshot = build_macro_snapshot(rows, as_of="2024-05-10")
    assert result["row_count"] == 2
    assert snapshot.source_lineage["cn.pmi_manufacturing"]["source_system"] == "nbs_official"


def test_replay_is_pit_observer_only_and_private(tmp_path: Path):
    observations_root = tmp_path / "observations"
    publish_observations(
        [
            _row(period="2024-02-29", available="2024-03-01T06:00:00+00:00", value=8.0),
            _row(period="2024-03-31", available="2024-04-01T06:00:00+00:00", value=9.0),
            _row(period="2024-04-30", available="2024-05-01T06:00:00+00:00", value=10.0),
            _row(
                period="2024-05-31",
                available="2024-06-01T06:00:00+00:00",
                value=99.0,
                vintage="future",
            ),
        ],
        root=observations_root,
        run_id="g1",
    )
    calendar = tmp_path / "trade_cal.parquet"
    pd.DataFrame(
        [
            {"cal_date": "20240509", "is_open": 1},
            {"cal_date": "20240510", "is_open": 1},
            {"cal_date": "20240511", "is_open": 0},
        ]
    ).to_parquet(calendar, index=False)
    result = run_macro_replay(
        start_date="2024-05-09",
        end_date="2024-05-11",
        observations_root=observations_root,
        calendar_path=calendar,
        output_root=tmp_path / "replay",
        run_id="r1",
    )
    table = Path(result["output_dir"]) / "daily_snapshots.parquet"
    frame = pd.read_parquet(table)
    assert len(frame) == 2
    assert frame["applied"].eq(False).all()  # noqa: E712
    assert frame["production_eligible"].eq(False).all()  # noqa: E712
    assert frame["snapshot_json"].map(lambda value: "future" not in value).all()
    for path in Path(result["output_dir"]).iterdir():
        assert os.stat(path).st_mode & 0o777 == 0o600

    second = run_macro_replay(
        start_date="2024-05-09",
        end_date="2024-05-11",
        observations_root=observations_root,
        calendar_path=calendar,
        output_root=tmp_path / "replay",
        run_id="r2",
    )
    assert second["daily_snapshots_sha256"] == result["daily_snapshots_sha256"]


def test_replay_rejects_conflicting_calendar_rows(tmp_path: Path):
    observations_root = tmp_path / "observations"
    publish_observations([_row()], root=observations_root, run_id="g1")
    calendar = tmp_path / "trade_cal.parquet"
    pd.DataFrame(
        [
            {"cal_date": "20240510", "is_open": 1},
            {"cal_date": "20240510", "is_open": 0},
        ]
    ).to_parquet(calendar, index=False)
    with pytest.raises(Exception, match="calendar_date_conflict"):
        run_macro_replay(
            start_date="2024-05-10",
            end_date="2024-05-10",
            observations_root=observations_root,
            calendar_path=calendar,
            output_root=tmp_path / "replay",
            run_id="conflict",
        )
    assert not (tmp_path / "replay" / "CN" / "conflict").exists()
