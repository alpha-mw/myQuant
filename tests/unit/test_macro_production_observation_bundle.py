from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

import quant_investor.macro.production_observation_bundle as production_bundle
from quant_investor.macro.contracts import MacroObservation, canonical_hash
from quant_investor.macro.local_market_observations import (
    LocalMarketObservationError,
)
from quant_investor.macro.official_web_compiler import (
    OfficialWebCompilerError,
    PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER,
    PBC_MONEY_STOCK_PARSER_V2,
    compile_official_web_bundle_file,
)
from quant_investor.macro.production_observation_bundle import (
    LOCAL_MARKET_OBSERVATION_ROLL_SCHEMA,
    OFFICIAL_OBSERVATION_REFRESH_SCHEMA,
    ProductionObservationBundleError,
    publish_local_market_breadth_roll,
    publish_local_market_breadth_update,
    publish_macro_official_observation_refresh,
    publish_macro_production_observation_bundle,
    validate_production_observation_chain,
)
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
    pointer_sha256,
    publish_observations,
)
from tests.unit.test_local_market_observations import (
    _Fixture,
    _write_fixture as _write_local_fixture,
)
from tests.unit.test_macro_official_web_compiler import (
    _bundle as _write_official_fixture,
    _fixture_pages,
    _page,
    _pbc_html,
    _requested_scope,
    _seal_inputs,
)


_BOOTSTRAP_DATES = ("20260710", "20260713", "20260714")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(fixture: _Fixture) -> dict[str, str]:
    return {
        "target_trade_date": fixture.target_trade_date,
        "snapshot_manifest_path": str(fixture.snapshot_manifest_path),
        "expected_snapshot_manifest_sha256": (
            fixture.snapshot_manifest_sha256
        ),
        "coverage_manifest_path": str(fixture.coverage_manifest_path),
        "expected_coverage_manifest_sha256": (
            fixture.coverage_manifest_sha256
        ),
        "scope_artifact_path": str(fixture.scope_artifact_path),
        "expected_scope_artifact_sha256": fixture.scope_artifact_sha256,
    }


def _write_bootstrap_plan(
    path: Path,
    fixtures: list[_Fixture],
) -> tuple[Path, str]:
    payload = {
        "schema_version": "cn-local-breadth-bootstrap-plan.v1",
        "market": "CN",
        "targets": [_target(fixture) for fixture in fixtures],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path, _sha256(path)


def _rewrite_plan(
    path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)
    return _sha256(path)


def _bootstrap_local_fixtures(tmp_path: Path) -> list[_Fixture]:
    return [
        _write_local_fixture(
            tmp_path / f"local-{trade_date}",
            target_trade_date=trade_date,
        )
        for trade_date in _BOOTSTRAP_DATES
    ]


def _inputs(tmp_path: Path) -> dict[str, Any]:
    plan, capture, raw_root, _plan, _capture = _write_official_fixture(
        tmp_path / "official-source"
    )
    official = compile_official_web_bundle_file(
        plan,
        capture_manifest_path=capture,
        raw_root=raw_root,
        output_root=tmp_path / "official-bundle",
        run_id="official-20260716",
    )
    local_plan, local_plan_sha = _write_bootstrap_plan(
        tmp_path / "local-bootstrap-plan.json",
        _bootstrap_local_fixtures(tmp_path),
    )
    return {
        "official_bundle_manifest_path": official["artifacts"]["manifest"],
        "expected_official_bundle_manifest_sha256": official[
            "normalization_manifest_sha256"
        ],
        "expected_official_plan_sha256": official["plan_file_sha256"],
        "local_bootstrap_plan_path": local_plan,
        "expected_local_bootstrap_plan_sha256": local_plan_sha,
    }


def _publish(
    tmp_path: Path,
    inputs: dict[str, Any],
    *,
    as_of: str = "20260715",
    expected_pointer: str = "",
    run_id: str = "macro-production-20260715",
) -> dict[str, Any]:
    return publish_macro_production_observation_bundle(
        **inputs,
        as_of=as_of,
        canonical_observations_root=tmp_path / "observations",
        run_id=run_id,
        expected_pointer_sha256=expected_pointer,
    )


def _daily_kwargs(fixture: _Fixture) -> dict[str, Any]:
    return {
        "snapshot_manifest_path": fixture.snapshot_manifest_path,
        "expected_snapshot_manifest_sha256": (
            fixture.snapshot_manifest_sha256
        ),
        "coverage_manifest_path": fixture.coverage_manifest_path,
        "expected_coverage_manifest_sha256": (
            fixture.coverage_manifest_sha256
        ),
        "target_trade_date": fixture.target_trade_date,
        "scope_artifact_path": fixture.scope_artifact_path,
        "expected_scope_artifact_sha256": fixture.scope_artifact_sha256,
    }


def _next_daily_fixture(tmp_path: Path) -> _Fixture:
    return _write_local_fixture(
        tmp_path,
        target_trade_date="20260715",
        data_latest_complete="20260715",
        data_snapshot_id="20260716T042027Z",
        coverage_snapshot_id="20260716T041500Z",
        rows_by_date={
            "20260713": 100,
            "20260714": 100,
            "20260715": 100,
        },
        part_mtime="2026-07-16T04:20:29.774000Z",
        snapshot_manifest_mtime="2026-07-16T04:21:14Z",
        coverage_manifest_mtime="2026-07-16T04:19:00Z",
        scope_mtime="2026-07-16T04:18:00Z",
    )


def _same_date_correction_fixture(tmp_path: Path) -> _Fixture:
    def correct_one_decliner(frame):
        target = frame.index[
            (frame["trade_date"] == "20260714")
            & frame["pct_chg"].lt(0.0)
        ][0]
        frame.loc[target, "pct_chg"] = 1.0
        return frame

    return _write_local_fixture(
        tmp_path,
        target_trade_date="20260714",
        data_latest_complete="20260714",
        data_snapshot_id="20260715T060000Z",
        coverage_snapshot_id="20260715T055900Z",
        frame_mutator=correct_one_decliner,
        part_mtime="2026-07-15T05:55:00Z",
        snapshot_manifest_mtime="2026-07-15T06:00:01Z",
        coverage_manifest_mtime="2026-07-15T05:59:01Z",
        scope_mtime="2026-07-15T05:54:00Z",
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _open_days(
    path: Path,
    dates: list[str] | None = None,
) -> tuple[Path, str, list[str]]:
    selected = dates or [
        "20260710",
        "20260713",
        "20260714",
        "20260715",
        "20260716",
        "20260717",
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-open-days.v1",
                "market": "CN",
                "open_dates": selected,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path, _sha256(path), selected


def _v2_official_inputs(tmp_path: Path) -> dict[str, Any]:
    pages = [
        item
        for item in _fixture_pages()
        if item[0]["page_id"] != "pbc-202602"
    ]
    for index, (page, relative, body) in enumerate(pages):
        if page["parser_id"] != PBC_MONEY_STOCK_PARSER:
            continue
        updated = dict(page)
        updated["parser_id"] = PBC_MONEY_STOCK_PARSER_V2
        updated["parser_contract_sha256"] = PARSER_CONTRACT_SHA256[
            PBC_MONEY_STOCK_PARSER_V2
        ]
        pages[index] = (updated, relative, body)
    june = _page(
        "pbc-202606",
        PBC_MONEY_STOCK_PARSER_V2,
        "pbc_official",
        (
            "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/"
            "2026071515025183948/index.html"
        ),
        "202606",
    )
    pages.append(
        (
            june,
            "pbc/pbc-202606.html",
            _pbc_html(
                "202606",
                "2026-07-15 15:00:09",
                "4.0",
                "8.0",
                "22.83",
                half_year_title=True,
            ),
        )
    )
    scope = [
        row
        for row in _requested_scope()
        if not (
            row["indicator_id"] in {"cn.m1_yoy", "cn.m2_yoy"}
            and row["period_end"] == "2026-03-31"
        )
    ]
    scope.extend(
        {"indicator_id": indicator_id, "period_end": "2026-06-30"}
        for indicator_id in ("cn.m1_yoy", "cn.m2_yoy")
    )
    plan, capture, raw_root, _plan, _capture = _seal_inputs(
        tmp_path / "official-refresh-source",
        {
            "schema_version": "macro-official-web-plan.v1",
            "market": "CN",
            "requested_scope": scope,
            "pages": [item[0] for item in pages],
        },
        pages,
    )
    result = compile_official_web_bundle_file(
        plan,
        capture_manifest_path=capture,
        raw_root=raw_root,
        output_root=tmp_path / "official-refresh-bundle",
        run_id="official-v2-20260716",
    )
    return {
        "official_bundle_manifest_path": result["artifacts"]["manifest"],
        "expected_official_bundle_manifest_sha256": result[
            "normalization_manifest_sha256"
        ],
        "expected_official_plan_sha256": result["plan_file_sha256"],
    }


def _roll_fixture(tmp_path: Path) -> _Fixture:
    return _write_local_fixture(
        tmp_path,
        target_trade_date="20260716",
        data_latest_complete="20260716",
        data_snapshot_id="20260716T072500Z",
        coverage_snapshot_id="20260716T072400Z",
        rows_by_date={
            "20260714": 100,
            "20260715": 100,
            "20260716": 100,
        },
        part_mtime="2026-07-16T07:20:00Z",
        snapshot_manifest_mtime="2026-07-16T07:25:01Z",
        coverage_manifest_mtime="2026-07-16T07:24:01Z",
        scope_mtime="2026-07-16T07:18:00Z",
    )


def _next_roll_fixture(tmp_path: Path) -> _Fixture:
    return _write_local_fixture(
        tmp_path,
        target_trade_date="20260717",
        data_latest_complete="20260717",
        data_snapshot_id="20260717T072500Z",
        coverage_snapshot_id="20260717T072400Z",
        rows_by_date={
            "20260715": 100,
            "20260716": 100,
            "20260717": 100,
        },
        part_mtime="2026-07-17T07:20:00Z",
        snapshot_manifest_mtime="2026-07-17T07:25:01Z",
        coverage_manifest_mtime="2026-07-17T07:24:01Z",
        scope_mtime="2026-07-17T07:18:00Z",
    )


def test_atomic_39_row_publication_and_one_date_local_append_round_trip(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    plan_path = Path(inputs["local_bootstrap_plan_path"])
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert set(plan_payload) == {"schema_version", "market", "targets"}
    assert plan_payload["schema_version"] == (
        "cn-local-breadth-bootstrap-plan.v1"
    )
    assert plan_payload["market"] == "CN"
    assert [item["target_trade_date"] for item in plan_payload["targets"]] == (
        list(_BOOTSTRAP_DATES)
    )
    assert len(plan_payload["targets"]) == 3
    assert os.stat(plan_path).st_mode & 0o777 == 0o600

    receipt = _publish(tmp_path, inputs)
    rows, pointer = load_observations(root)
    manifest = pointer["generation_manifest"]

    assert receipt["promoted"] is True
    assert receipt["observation_count"] == 39
    assert receipt["indicator_count"] == 13
    assert receipt["history_length_per_indicator"] == 3
    assert receipt["official_observation_count"] == 36
    assert receipt["local_observation_count"] == 3
    assert receipt["local_target_trade_dates"] == list(_BOOTSTRAP_DATES)
    assert receipt["as_of"] == "20260714"
    assert receipt["decision_cutoff_at"] == (
        "2026-07-15T07:00:00+00:00"
    )
    assert receipt["local_bootstrap_plan_sha256"] == _sha256(plan_path)
    assert receipt["official_evidence_file_count"] == 12
    assert receipt["incoming_evidence_file_count"] == (
        receipt["official_evidence_file_count"]
        + receipt["local_evidence_file_count"]
    )
    assert receipt["generation_evidence_file_count"] == receipt[
        "incoming_evidence_file_count"
    ]
    assert len(receipt["local_coverage_contract_sha256"]) == 64
    assert receipt["local_effective_available_at"] == (
        "2026-07-15T04:21:14+00:00"
    )
    assert receipt["snapshot_readiness_status"] == "pass"
    assert receipt["snapshot_national_coverage"] == 0.8125
    assert receipt["snapshot_blockers"] == []
    assert receipt["atomic_combined_publication"] is True
    assert receipt["strict_readback_validated"] is True
    assert len(rows) == 39
    assert manifest["schema_version"] == "macro-observation-generation.v2"
    assert manifest["evidence_file_count"] == receipt[
        "generation_evidence_file_count"
    ]
    assert len(manifest["observation_evidence"]) == 39
    assert {
        digest
        for digests in manifest["observation_evidence"].values()
        for digest in digests
    } == {item["sha256"] for item in manifest["evidence_files"]}
    generation = root / "_generations" / pointer["generation_id"]
    for item in manifest["evidence_files"]:
        evidence_path = generation / item["path"]
        assert os.stat(evidence_path).st_mode & 0o777 == 0o600
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == (
            item["sha256"]
        )
    local_evidence_items = {
        item["metadata"]["target_trade_date"]: item
        for item in manifest["evidence_files"]
        if item["metadata"].get("evidence_kind")
        == "strict_parquet_local_observation_evidence"
    }
    assert set(local_evidence_items) == set(_BOOTSTRAP_DATES)
    local_rows = [
        row for row in rows if row["source_system"] == "local_strict_parquet"
    ]
    local_rows.sort(key=lambda row: row["period_end"])
    assert [row["period_end"].replace("-", "") for row in local_rows] == list(
        _BOOTSTRAP_DATES
    )
    for row in local_rows:
        compact_period_end = row["period_end"].replace("-", "")
        evidence_item = local_evidence_items[compact_period_end]
        local_evidence = json.loads(
            (generation / evidence_item["path"]).read_bytes()
        )
        assert local_evidence["target_trade_date"] == compact_period_end
        assert evidence_item["sha256"] in manifest[
            "observation_evidence"
        ][row["content_hash"]]
    json.dumps(receipt, allow_nan=False)

    daily = _next_daily_fixture(tmp_path / "local-next")
    update = publish_local_market_breadth_update(
        **_daily_kwargs(daily),
        as_of="2026-07-16T04:30:00+00:00",
        canonical_observations_root=root,
        run_id="local-breadth-20260716",
        expected_pointer_sha256=receipt["pointer_sha256"],
    )
    updated_rows, updated_pointer = load_observations(root)

    assert update["promoted"] is True
    assert update["local_observation_count"] == 1
    assert update["local_target_trade_dates"] == ["20260715"]
    assert update["as_of"] == "20260715"
    assert update["decision_cutoff_at"] == (
        "2026-07-16T04:30:00+00:00"
    )
    assert update["parent_as_of"] == "20260714"
    assert update["parent_decision_cutoff_at"] == (
        "2026-07-15T07:00:00+00:00"
    )
    assert update["update_mode"] == "next_date_append"
    assert update["local_snapshot_manifest_sha256"] == (
        daily.snapshot_manifest_sha256
    )
    assert update["local_coverage_manifest_sha256"] == (
        daily.coverage_manifest_sha256
    )
    assert update["local_scope_artifact_sha256"] == (
        daily.scope_artifact_sha256
    )
    assert update["incoming_evidence_file_count"] == 5
    assert len(update["local_coverage_contract_sha256"]) == 64
    assert update["local_effective_available_at"] == (
        "2026-07-16T04:21:14+00:00"
    )
    assert update["snapshot_readiness_status"] == "pass"
    assert update["snapshot_national_coverage"] == 0.8125
    assert update["strict_readback_validated"] is True
    assert len(updated_rows) == 40
    assert sum(
        row["indicator_id"] == "market.breadth" for row in updated_rows
    ) == 4
    assert len(
        updated_pointer["generation_manifest"]["observation_evidence"]
    ) == 40
    update_metadata = updated_pointer["generation_manifest"]["metadata"]
    assert update_metadata["parent_as_of"] == "20260714"
    assert update_metadata["parent_decision_cutoff_at"] == (
        "2026-07-15T07:00:00+00:00"
    )
    assert update_metadata["update_mode"] == "next_date_append"
    assert pointer_sha256(root) == update["pointer_sha256"]
    json.dumps(update, allow_nan=False)


def test_legacy_append_v2_official_refresh_and_local_catchup_projection(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    bootstrap = _publish(tmp_path, inputs)
    daily = _next_daily_fixture(tmp_path / "local-next")
    appended = publish_local_market_breadth_update(
        **_daily_kwargs(daily),
        as_of="2026-07-16T04:30:00+00:00",
        canonical_observations_root=root,
        run_id="legacy-local-append-20260715",
        expected_pointer_sha256=bootstrap["pointer_sha256"],
    )
    open_path, open_sha, open_dates = _open_days(
        tmp_path / "market-open-days.json"
    )
    refreshed = publish_macro_official_observation_refresh(
        **_v2_official_inputs(tmp_path),
        target_as_of="20260716",
        decision_cutoff_at="2026-07-16T07:00:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=root,
        run_id="official-refresh-20260716",
        expected_pointer_sha256=appended["pointer_sha256"],
    )
    refreshed_rows, refreshed_pointer = load_observations(root)
    refreshed_manifest = refreshed_pointer["generation_manifest"]

    assert refreshed["schema_version"] == OFFICIAL_OBSERVATION_REFRESH_SCHEMA
    assert refreshed["promoted"] is True
    assert refreshed["observation_count"] == 39
    assert refreshed["latest_local_trade_date"] == "20260715"
    assert refreshed["local_open_session_lag"] == 1
    assert refreshed["retained_local_trade_dates"] == [
        "20260713",
        "20260714",
        "20260715",
    ]
    assert refreshed_manifest["metadata"]["schema_version"] == (
        OFFICIAL_OBSERVATION_REFRESH_SCHEMA
    )
    assert len(refreshed_manifest["added_content_hashes"]) == 36
    assert len(refreshed_manifest["removed_content_hashes"]) == 37
    assert len(refreshed_manifest["replaced_content_hashes"]) == 30
    assert len(refreshed_rows) == 39
    assert [
        row["period_end"].replace("-", "")
        for row in sorted(
            (
                row
                for row in refreshed_rows
                if row["indicator_id"] == "market.breadth"
            ),
            key=lambda row: row["period_end"],
        )
    ] == ["20260713", "20260714", "20260715"]
    pbc_parsers = {
        item["metadata"]["parser_id"]
        for item in refreshed_manifest["evidence_files"]
        if item["metadata"].get("source_system") == "pbc_official"
    }
    assert pbc_parsers == {PBC_MONEY_STOCK_PARSER_V2}
    validated = validate_production_observation_chain(
        refreshed_rows,
        generation_manifest=refreshed_manifest,
        pointer_metadata=refreshed_pointer["metadata"],
        canonical_root=root,
    )
    assert set(validated) == {
        row["content_hash"] for row in refreshed_rows
    }
    lag_tamper = deepcopy(refreshed_manifest)
    lag_tamper["metadata"]["local_open_session_lag"] = 2
    with pytest.raises(
        ProductionObservationBundleError,
        match="local_lag_binding_invalid",
    ):
        validate_production_observation_chain(
            refreshed_rows,
            generation_manifest=lag_tamper,
            pointer_metadata=lag_tamper["metadata"],
            canonical_root=root,
        )
    parser_tamper = deepcopy(refreshed_manifest)
    tampered_pbc = next(
        item
        for item in parser_tamper["evidence_files"]
        if item["metadata"].get("source_system") == "pbc_official"
        and item["metadata"].get("support_only") is False
    )
    tampered_pbc["metadata"]["parser_id"] = PBC_MONEY_STOCK_PARSER
    tampered_pbc["metadata"]["parser_contract_sha256"] = (
        PARSER_CONTRACT_SHA256[PBC_MONEY_STOCK_PARSER]
    )
    tampered_pbc["metadata_sha256"] = canonical_hash(
        tampered_pbc["metadata"]
    )
    parser_tamper["evidence_set_sha256"] = canonical_hash(
        {"evidence_files": parser_tamper["evidence_files"]}
    )
    with pytest.raises(
        ProductionObservationBundleError,
        match="pbc_parser_set_invalid",
    ):
        validate_production_observation_chain(
            refreshed_rows,
            generation_manifest=parser_tamper,
            pointer_metadata=parser_tamper["metadata"],
            canonical_root=root,
        )

    missing_parent_tamper = deepcopy(refreshed_manifest)
    missing_parent_tamper["parent_generation_id"] = "missing-parent"
    with pytest.raises(
        ProductionObservationBundleError,
        match="parent_readback_invalid",
    ):
        validate_production_observation_chain(
            refreshed_rows,
            generation_manifest=missing_parent_tamper,
            pointer_metadata=missing_parent_tamper["metadata"],
            canonical_root=root,
        )

    pointer_tamper = deepcopy(refreshed_manifest)
    pointer_tamper["parent_pointer_sha256"] = "0" * 64
    with pytest.raises(
        ProductionObservationBundleError,
        match="parent_pointer_mismatch",
    ):
        validate_production_observation_chain(
            refreshed_rows,
            generation_manifest=pointer_tamper,
            pointer_metadata=pointer_tamper["metadata"],
            canonical_root=root,
        )

    content_set_tamper = deepcopy(refreshed_manifest)
    content_set_tamper["metadata"]["parent_content_set_hash"] = "0" * 64
    with pytest.raises(
        ProductionObservationBundleError,
        match="parent_binding_mismatch",
    ):
        validate_production_observation_chain(
            refreshed_rows,
            generation_manifest=content_set_tamper,
            pointer_metadata=content_set_tamper["metadata"],
            canonical_root=root,
        )

    roll_fixture = _roll_fixture(tmp_path / "local-roll")
    rolled = publish_local_market_breadth_roll(
        **_daily_kwargs(roll_fixture),
        target_as_of="20260716",
        decision_cutoff_at="2026-07-16T07:30:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=root,
        run_id="local-catchup-20260716",
        expected_pointer_sha256=refreshed["pointer_sha256"],
    )
    rolled_rows, rolled_pointer = load_observations(root)
    rolled_manifest = rolled_pointer["generation_manifest"]

    assert rolled["schema_version"] == LOCAL_MARKET_OBSERVATION_ROLL_SCHEMA
    assert rolled["promoted"] is True
    assert rolled["update_mode"] == "local_catchup"
    assert rolled["latest_local_trade_date"] == "20260716"
    assert rolled["local_open_session_lag"] == 0
    assert rolled["local_target_trade_dates"] == [
        "20260714",
        "20260715",
        "20260716",
    ]
    assert len(rolled_rows) == 39
    assert len(rolled_manifest["added_content_hashes"]) == 1
    assert len(rolled_manifest["removed_content_hashes"]) == 1
    assert rolled_manifest["replaced_content_hashes"] == []
    validated_roll = validate_production_observation_chain(
        rolled_rows,
        generation_manifest=rolled_manifest,
        pointer_metadata=rolled_pointer["metadata"],
        canonical_root=root,
    )
    assert set(validated_roll) == {
        row["content_hash"] for row in rolled_rows
    }
    json.dumps(rolled, allow_nan=False)

    next_fixture = _next_roll_fixture(tmp_path / "local-next-roll")
    next_roll = publish_local_market_breadth_roll(
        **_daily_kwargs(next_fixture),
        target_as_of="20260717",
        decision_cutoff_at="2026-07-17T08:00:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=root,
        run_id="local-next-roll-20260717",
        expected_pointer_sha256=rolled["pointer_sha256"],
    )
    next_rows, next_pointer = load_observations(root)

    assert next_roll["update_mode"] == "next_date_roll"
    assert next_roll["local_target_trade_dates"] == [
        "20260715",
        "20260716",
        "20260717",
    ]
    assert len(next_rows) == 39
    validate_production_observation_chain(
        next_rows,
        generation_manifest=next_pointer["generation_manifest"],
        pointer_metadata=next_pointer["metadata"],
        canonical_root=root,
    )


def test_local_append_accepts_snapshot_and_coverage_same_manifest(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    bootstrap = _publish(tmp_path, inputs)
    daily = _next_daily_fixture(tmp_path / "local-next")
    daily_kwargs = _daily_kwargs(daily)
    snapshot = json.loads(
        daily.snapshot_manifest_path.read_text(encoding="utf-8")
    )
    coverage = json.loads(
        daily.coverage_manifest_path.read_text(encoding="utf-8")
    )
    coverage_summary = {
        **coverage["coverage"],
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "pit_generation_id": "pit-test-20260716",
        "pit_generation_manifest_path": "/tmp/pit-test/manifest.json",
        "pit_generation_manifest_sha256": "0" * 64,
        "pit_membership_path": "/tmp/pit-test/stock_basic_membership.parquet",
        "pit_membership_sha256": "0" * 64,
    }
    snapshot["coverage"] = coverage_summary
    snapshot["metadata"] = {
        **dict(snapshot.get("metadata") or {}),
        "coverage": coverage_summary,
    }
    snapshot["manifest_path"] = str(daily.snapshot_manifest_path)
    snapshot_mtime_ns = daily.snapshot_manifest_path.stat().st_mtime_ns
    daily.snapshot_manifest_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(daily.snapshot_manifest_path, 0o600)
    os.utime(
        daily.snapshot_manifest_path,
        ns=(snapshot_mtime_ns, snapshot_mtime_ns),
    )
    same_manifest_sha = _sha256(daily.snapshot_manifest_path)
    daily_kwargs.update(
        coverage_manifest_path=daily.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=same_manifest_sha,
        expected_coverage_manifest_sha256=same_manifest_sha,
    )

    update = publish_local_market_breadth_update(
        **daily_kwargs,
        as_of="2026-07-16T04:30:00+00:00",
        canonical_observations_root=root,
        run_id="local-breadth-same-manifest-20260716",
        expected_pointer_sha256=bootstrap["pointer_sha256"],
    )

    assert update["promoted"] is True
    assert update["incoming_evidence_file_count"] == 4
    assert update["strict_readback_validated"] is True


def test_official_refresh_rejects_three_open_session_local_lag(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    bootstrap = _publish(tmp_path, inputs)
    before = (root / "_latest.json").read_bytes()
    open_path, open_sha, open_dates = _open_days(
        tmp_path / "market-open-days.json"
    )

    with pytest.raises(
        ProductionObservationBundleError,
        match="local_history_lag_exceeds_two_sessions",
    ):
        publish_macro_official_observation_refresh(
            official_bundle_manifest_path=inputs[
                "official_bundle_manifest_path"
            ],
            expected_official_bundle_manifest_sha256=inputs[
                "expected_official_bundle_manifest_sha256"
            ],
            expected_official_plan_sha256=inputs[
                "expected_official_plan_sha256"
            ],
            target_as_of="20260717",
            decision_cutoff_at="2026-07-17T07:00:00+00:00",
            pinned_open_dates=open_dates,
            market_open_days_path=open_path,
            expected_market_open_days_sha256=open_sha,
            canonical_observations_root=root,
            run_id="lag-three-must-not-publish",
            expected_pointer_sha256=bootstrap["pointer_sha256"],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (
        root / "_generations" / "lag-three-must-not-publish"
    ).exists()


def test_official_refresh_accepts_two_open_session_local_lag(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    bootstrap = _publish(tmp_path, inputs)
    open_path, open_sha, open_dates = _open_days(
        tmp_path / "market-open-days.json"
    )

    refreshed = publish_macro_official_observation_refresh(
        official_bundle_manifest_path=inputs[
            "official_bundle_manifest_path"
        ],
        expected_official_bundle_manifest_sha256=inputs[
            "expected_official_bundle_manifest_sha256"
        ],
        expected_official_plan_sha256=inputs[
            "expected_official_plan_sha256"
        ],
        target_as_of="20260716",
        decision_cutoff_at="2026-07-16T07:00:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=root,
        run_id="lag-two-accepted",
        expected_pointer_sha256=bootstrap["pointer_sha256"],
    )

    assert refreshed["promoted"] is True
    assert refreshed["local_open_session_lag"] == 2
    assert refreshed["latest_local_trade_date"] == "20260714"
    rows, pointer = load_observations(root)
    validate_production_observation_chain(
        rows,
        generation_manifest=pointer["generation_manifest"],
        pointer_metadata=pointer["metadata"],
        canonical_root=root,
    )


def test_public_validator_rejects_allowed_schema_single_blob_spoof(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    _publish(tmp_path, inputs)
    rows, pointer = load_observations(tmp_path / "observations")
    manifest = pointer["generation_manifest"]

    validated = validate_production_observation_chain(
        rows,
        generation_manifest=manifest,
        pointer_metadata=pointer["metadata"],
    )
    assert set(validated) == {row["content_hash"] for row in rows}

    spoofed = deepcopy(manifest)
    single_file = next(
        item
        for item in manifest["evidence_files"]
        if item["metadata"].get("evidence_kind")
        == "official_web_response_entity"
        and item["metadata"].get("support_only") is False
    )
    digest = single_file["sha256"]
    spoofed["evidence_files"] = [single_file]
    spoofed["evidence_file_count"] = 1
    spoofed["evidence_set_sha256"] = canonical_hash(
        {"evidence_files": [single_file]}
    )
    spoofed["observation_evidence"] = {
        row["content_hash"]: [digest] for row in rows
    }

    assert spoofed["metadata"]["schema_version"] == (
        production_bundle.PRODUCTION_OBSERVATION_BUNDLE_SCHEMA
    )
    with pytest.raises(
        ProductionObservationBundleError,
        match="production_observation_chain_official_evidence_scope_invalid",
    ):
        validate_production_observation_chain(
            rows,
            generation_manifest=spoofed,
            pointer_metadata=pointer["metadata"],
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["targets"].pop(),
        lambda payload: payload["targets"].reverse(),
    ],
    ids=["not-exactly-three", "not-increasing"],
)
def test_bootstrap_plan_requires_exactly_three_increasing_dates(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    inputs = _inputs(tmp_path)
    plan_path = Path(inputs["local_bootstrap_plan_path"])
    inputs["expected_local_bootstrap_plan_sha256"] = _rewrite_plan(
        plan_path,
        mutator,
    )

    with pytest.raises(
        ProductionObservationBundleError,
        match=(
            "production_local_bootstrap_plan_(?:target_count_invalid|"
            "dates_not_strictly_increasing)"
        ),
    ):
        _publish(tmp_path, inputs, run_id="invalid-local-plan")

    assert pointer_sha256(tmp_path / "observations") == ""


def test_stale_pointer_cas_fails_before_write(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"

    with pytest.raises(
        ProductionObservationBundleError,
        match="production_observation_pointer_cas_mismatch",
    ):
        _publish(
            tmp_path,
            inputs,
            expected_pointer="0" * 64,
            run_id="stale-cas",
        )

    assert pointer_sha256(root) == ""
    assert not (root / "_latest.json").exists()


def test_official_reader_rejects_permission_tamper(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest_path = Path(inputs["official_bundle_manifest_path"])
    raw_path = next((manifest_path.parent / "raw").glob("*.html"))
    os.chmod(raw_path, 0o644)

    with pytest.raises(
        (ProductionObservationBundleError, OfficialWebCompilerError),
        match="permissions_unsafe",
    ):
        _publish(tmp_path, inputs, run_id="tampered-reader")

    assert pointer_sha256(tmp_path / "observations") == ""


def test_future_decision_cutoff_does_not_publish(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    with pytest.raises(
        ProductionObservationBundleError,
        match="production_observation_as_of_in_future",
    ):
        _publish(
            tmp_path,
            inputs,
            as_of="20260720",
            run_id="stale-readiness",
        )

    assert pointer_sha256(tmp_path / "observations") == ""


def test_local_update_rejects_target_date_rollback_before_source_compile(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    receipt = _publish(tmp_path, inputs)
    root = tmp_path / "observations"
    before = (root / "_latest.json").read_bytes()
    older = _write_local_fixture(
        tmp_path / "local-rollback",
        target_trade_date="20260713",
        data_latest_complete="20260713",
    )

    with pytest.raises(
        ProductionObservationBundleError,
        match="local_market_observation_target_trade_date_rollback",
    ):
        publish_local_market_breadth_update(
            **_daily_kwargs(older),
            as_of="2026-07-15T07:00:00+00:00",
            canonical_observations_root=root,
            run_id="target-rollback",
            expected_pointer_sha256=receipt["pointer_sha256"],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "target-rollback").exists()


def test_local_update_rejects_decision_cutoff_rollback_before_source_compile(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    receipt = _publish(tmp_path, inputs)
    root = tmp_path / "observations"
    before = (root / "_latest.json").read_bytes()
    next_date = _next_daily_fixture(tmp_path / "local-cutoff-rollback")

    with pytest.raises(
        ProductionObservationBundleError,
        match="local_market_observation_decision_cutoff_rollback",
    ):
        publish_local_market_breadth_update(
            **_daily_kwargs(next_date),
            as_of="2026-07-15T06:59:59+00:00",
            canonical_observations_root=root,
            run_id="cutoff-rollback",
            expected_pointer_sha256=receipt["pointer_sha256"],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "cutoff-rollback").exists()


def test_same_date_correction_then_identical_retry_is_immutable(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    bootstrap = _publish(tmp_path, inputs)
    root = tmp_path / "observations"
    correction = _same_date_correction_fixture(tmp_path / "same-date")

    with pytest.raises(
        ProductionObservationBundleError,
        match="local_market_observation_same_date_run_id_reuse",
    ):
        publish_local_market_breadth_update(
            **_daily_kwargs(correction),
            as_of="2026-07-15T07:30:00+00:00",
            canonical_observations_root=root,
            run_id=bootstrap["generation_id"],
            expected_pointer_sha256=bootstrap["pointer_sha256"],
        )

    corrected = publish_local_market_breadth_update(
        **_daily_kwargs(correction),
        as_of="2026-07-15T07:30:00+00:00",
        canonical_observations_root=root,
        run_id="same-date-correction",
        expected_pointer_sha256=bootstrap["pointer_sha256"],
    )
    corrected_rows, corrected_pointer = load_observations(root)
    corrected_pointer_bytes = (root / "_latest.json").read_bytes()

    assert corrected["promoted"] is True
    assert corrected["update_mode"] == "same_date_correction"
    assert corrected["parent_as_of"] == "20260714"
    assert corrected["parent_decision_cutoff_at"] == (
        "2026-07-15T07:00:00+00:00"
    )
    same_date_rows = [
        row
        for row in corrected_rows
        if row["indicator_id"] == "market.breadth"
        and row["period_end"] == "2026-07-14"
    ]
    assert len(same_date_rows) == 2
    assert corrected_pointer["generation_manifest"]["metadata"][
        "update_mode"
    ] == "same_date_correction"

    retried = publish_local_market_breadth_update(
        **_daily_kwargs(correction),
        as_of="2026-07-15T07:30:00+00:00",
        canonical_observations_root=root,
        run_id="same-date-idempotent-retry",
        expected_pointer_sha256=corrected["pointer_sha256"],
    )

    assert retried["promoted"] is False
    assert retried["update_mode"] == "same_date_idempotent_retry"
    assert retried["pointer_sha256"] == corrected["pointer_sha256"]
    assert (root / "_latest.json").read_bytes() == corrected_pointer_bytes
    assert not (
        root / "_generations" / "same-date-idempotent-retry"
    ).exists()


def test_partial_full_a_coverage_rejects_bootstrap(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    plan_path = Path(inputs["local_bootstrap_plan_path"])
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    target = plan_payload["targets"][-1]
    coverage_path = Path(target["coverage_manifest_path"])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["coverage"]["coverage_ratio"] = 0.5
    coverage["metadata"]["coverage"]["coverage_ratio"] = 0.5
    coverage_path.write_text(
        json.dumps(coverage, sort_keys=True),
        encoding="utf-8",
    )
    target["expected_coverage_manifest_sha256"] = _sha256(coverage_path)
    plan_path.write_text(
        json.dumps(plan_payload, sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(plan_path, 0o600)
    inputs["expected_local_bootstrap_plan_sha256"] = _sha256(plan_path)

    with pytest.raises(
        LocalMarketObservationError,
        match="local_breadth_coverage_contract_invalid",
    ):
        _publish(tmp_path, inputs, run_id="partial-coverage")

    assert pointer_sha256(tmp_path / "observations") == ""


def test_local_update_rejects_legacy_v1_generation(tmp_path: Path) -> None:
    local = _write_local_fixture(tmp_path / "local-source")
    root = tmp_path / "observations"
    legacy_row = MacroObservation.from_mapping(
        {
            "indicator_id": "cn.pmi_manufacturing",
            "dimension_type": "national",
            "period_end": "2026-06-30",
            "release_at": "2026-06-30T01:30:00+00:00",
            "available_at": "2026-06-30T01:30:00+00:00",
            "vintage_id": "legacy-v1",
            "value": 50.3,
            "unit": "index",
            "frequency": "monthly",
            "source_system": "nbs_official",
            "source_record_id": "legacy-v1-record",
            "source_url": "https://www.stats.gov.cn/legacy/pmi.html",
            "fetched_at": "2026-06-30T01:30:00+00:00",
            "quality_status": "pass",
        }
    )
    publish_observations([legacy_row], root=root, run_id="legacy-v1")
    before = pointer_sha256(root)

    with pytest.raises(
        ProductionObservationBundleError,
        match="local_market_observation_existing_generation_v2_required",
    ):
        publish_local_market_breadth_update(
            **_daily_kwargs(local),
            as_of="20260715",
            canonical_observations_root=root,
            run_id="must-not-publish",
            expected_pointer_sha256=before,
        )

    assert pointer_sha256(root) == before
    assert not (root / "_generations" / "must-not-publish").exists()


def test_strict_readback_rejects_one_missing_mapping_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"
    receipt = _publish(tmp_path, inputs)
    before_pointer = (root / "_latest.json").read_bytes()
    before_rows, before_loaded = load_observations(root)
    previous_generation = (
        root / "_generations" / before_loaded["generation_id"]
    )
    before_generation = _tree_hashes(previous_generation)
    real_validator = production_bundle._strict_publication_readback
    daily = _next_daily_fixture(tmp_path / "local-next-failure")

    def incomplete_mapping(**kwargs):
        tampered = deepcopy(kwargs["generation_manifest"])
        mapping = tampered["observation_evidence"]
        mapping.pop(next(iter(mapping)))
        return real_validator(**{**kwargs, "generation_manifest": tampered})

    monkeypatch.setattr(
        production_bundle,
        "_strict_publication_readback",
        incomplete_mapping,
    )

    with pytest.raises(
        ProductionObservationBundleError,
        match="production_observation_readback_evidence_mapping_mismatch",
    ):
        publish_local_market_breadth_update(
            **_daily_kwargs(daily),
            as_of="2026-07-16T04:30:00+00:00",
            canonical_observations_root=root,
            run_id="readback-mapping-missing",
            expected_pointer_sha256=receipt["pointer_sha256"],
        )

    after_rows, after_loaded = load_observations(root)
    assert (root / "_latest.json").read_bytes() == before_pointer
    assert after_rows == before_rows
    assert after_loaded["generation_id"] == before_loaded["generation_id"]
    assert _tree_hashes(previous_generation) == before_generation
    assert not (root / "_generations" / "readback-mapping-missing").exists()


def test_bootstrap_validator_failure_keeps_canonical_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "observations"

    def injected_failure(**_kwargs):
        raise RuntimeError("injected_strict_validation_failure")

    monkeypatch.setattr(
        production_bundle,
        "_strict_publication_readback",
        injected_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="injected_strict_validation_failure",
    ):
        _publish(tmp_path, inputs, run_id="bootstrap-validation-failure")

    assert not (root / "_latest.json").exists()
    assert pointer_sha256(root) == ""
    assert not (
        root / "_generations" / "bootstrap-validation-failure"
    ).exists()
    with pytest.raises(
        MacroObservationStoreError,
        match="macro_observation_pointer_missing",
    ):
        load_observations(root)
