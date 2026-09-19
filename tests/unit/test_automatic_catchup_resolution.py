"""Automatic selection uses real synthetic Calendar, with explicit full-EOD admission seams."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest
from _public_catchup_fixture import completion, put, routed_collection
from test_daily_evidence_catchup_routing import controlled_routed_native
from test_daily_evidence_public_catchup import snapshot
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.automatic_catchup_contract import (
    REQUEST_SCHEMA,
    document_ref,
    run_path,
    validate_automatic_request,
)
from quant_investor.operations.automatic_catchup_resolution import (
    resolve_automatic_request,
    read_automatic_resolution,
)
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    build_head,
    head_js,
)
from scripts import daily_catchup, daily_completion_replay

NOW = datetime(2026, 8, 28, 14, tzinfo=timezone.utc)


def fixture(root, monkeypatch):
    _, request = routed_collection(root)
    seal_base, calls, serving, records = controlled_routed_native(root, monkeypatch, request)
    # Source/admission seams are identical to the Part A fixture; no full-EOD claim.
    monkeypatch.setattr(
        daily_completion_replay, "replay_native_completion", daily_catchup.replay_native_completion
    )

    def seal(day, previous):
        ref = seal_base(day, previous, current=True)
        value = records[day]["recorded_completion"]
        value["native_validation_completed_at"] = "2026-08-28T13:01:00Z"
        return put(root, ref["path"], value)

    seed = seal("20260826", completion(root, "20260825", request))
    value = {
        "schema_version": REQUEST_SCHEMA,
        **{
            k: request[k]
            for k in (
                "market",
                "strategy_id",
                "graph_sha256",
                "release_install_ref",
                "calendar_ref",
                "raw_calendar_ref",
                "recipe_ref",
                "day_input_refs",
            )
        },
        "action": "PLAN",
        "seed_completion_ref": seed,
    }
    ref = put(root, "auto.json", value)
    return value, ref, seal, calls, serving


def resolve(root, ref, request, **kwargs):
    return resolve_automatic_request(
        workspace=str(root),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
        now=NOW,
        **kwargs,
    )


def head(root, ref):
    day = ref["path"].split("/")[-2]
    value = build_head(day=day, ref=ref, previous_sha=None, registered_at="2026-08-28T13:02:00Z")
    raw = canonical_json_bytes(value)
    put(root, f"{PREFIX}/{HEAD_JSON}", raw)
    put(root, f"{PREFIX}/{HEAD_JS}", head_js(raw))


@pytest.mark.parametrize("prefix", [0, 1, 2])
@pytest.mark.parametrize("use_head", [False, True])
def test_target_and_contiguous_prefix_are_native_calendar_selected(
    tmp_path, monkeypatch, prefix, use_head
):
    request, ref, seal, calls, serving = fixture(tmp_path, monkeypatch)
    seed = request["seed_completion_ref"]
    if use_head:
        head(tmp_path, seed)
        request["seed_completion_ref"] = None
        ref = put(tmp_path, ref["path"], request)
    previous, adopted = seed, []
    for day in ["20260827", "20260828"][:prefix]:
        previous = seal(day, previous)
        adopted.append(previous)
    before = snapshot(tmp_path)
    result = resolve(tmp_path, ref, request)
    value = result["resolution"]
    assert value["target_trade_date"] == "20260828"
    assert value["ordered_trade_dates"] == ["20260827", "20260828"]
    assert value["adopted_completion_refs"] == adopted and value["anchor_ref"] == previous
    assert list(value["derived_collection"]["recipes"]) == ["20260827", "20260828"][prefix:]
    assert value["day_scopes"]["20260827"]["dashboard_mode"] == "HISTORICAL_CAPTURE"
    assert value["day_scopes"]["20260828"]["dashboard_mode"] == "CURRENT_LATEST_EOD"
    assert result["missing_input_dates"] == []
    assert snapshot(tmp_path) == before and calls == serving == []


@pytest.mark.parametrize(
    "fault", ["missing_seed", "both", "mirror_only", "json_only", "wrong_mirror", "corrupt"]
)
def test_head_seed_faults_never_fall_back_or_write(tmp_path, monkeypatch, fault):
    request, ref, _, calls, serving = fixture(tmp_path, monkeypatch)
    if fault != "missing_seed":
        head(tmp_path, request["seed_completion_ref"])
    if fault != "both":
        request["seed_completion_ref"] = None
    if fault == "mirror_only":
        (tmp_path / PREFIX / HEAD_JSON).unlink()
    elif fault == "json_only":
        (tmp_path / PREFIX / HEAD_JS).unlink()
    elif fault == "wrong_mirror":
        put(tmp_path, f"{PREFIX}/{HEAD_JS}", b"wrong")
    elif fault == "corrupt":
        put(tmp_path, f"{PREFIX}/{HEAD_JSON}", b"{}")
    ref = put(tmp_path, ref["path"], request)
    before = snapshot(tmp_path)
    with pytest.raises(ContractError):
        resolve(tmp_path, ref, request)
    assert snapshot(tmp_path) == before and calls == serving == []


def test_gap_completion_stops_before_missing_date_producer(tmp_path, monkeypatch):
    request, ref, seal, calls, serving = fixture(tmp_path, monkeypatch)
    seal(
        "20260828",
        {
            "path": "results/operations/daily_production/CN/20260827/completion.v1.json",
            "sha256": "a" * 64,
        },
    )
    before = snapshot(tmp_path)
    with pytest.raises(ContractError, match="AUTO_COMPLETION_BEYOND_GAP"):
        resolve(tmp_path, ref, request)
    assert snapshot(tmp_path) == before and calls == serving == []


def test_resolution_readback_ignores_later_mutable_head_and_keeps_exact_range(
    tmp_path, monkeypatch
):
    request, ref, seal, _, _ = fixture(tmp_path, monkeypatch)
    selected = resolve(tmp_path, ref, request)["resolution"]
    saved = put(tmp_path, run_path(ref, "resolution.v1.json"), selected)
    seal("20260827", request["seed_completion_ref"])
    put(tmp_path, f"{PREFIX}/{HEAD_JSON}", b"later invalid presentation")
    before = snapshot(tmp_path)
    read = read_automatic_resolution(
        workspace=str(tmp_path),
        resolution_ref=saved,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )
    assert read["resolution"] == selected and read["resolution"]["adopted_completion_refs"] == []
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_trade_date", "20260827"),
        ("resolved_at", "2026-08-28T12:00:00Z"),
        ("ordered_trade_dates", ["20260828"]),
        ("schema_version", "unchecked"),
    ],
)
def test_rehashed_resolution_cannot_change_derivation(tmp_path, monkeypatch, field, value):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    original = resolve(tmp_path, ref, request)["resolution"]
    original[field] = value
    saved = put(tmp_path, run_path(ref, "resolution.v1.json"), original)
    with pytest.raises(ContractError):
        read_automatic_resolution(
            workspace=str(tmp_path),
            resolution_ref=saved,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
        )


def test_identity_binds_path_and_content_and_request_cannot_supply_target(tmp_path, monkeypatch):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    copied = put(tmp_path, "copied.json", request)
    assert copied["sha256"] == ref["sha256"]
    assert run_path(ref, "resolution.v1.json") != run_path(copied, "resolution.v1.json")
    invalid = {**deepcopy(request), "target_trade_date": "20260828"}
    with pytest.raises(ContractError, match="AUTO_REQUEST_FIELDS_INVALID"):
        validate_automatic_request(invalid, release_install_ref=request["release_install_ref"])
    candidate = resolve(tmp_path, copied, request)["resolution"]
    assert candidate["auto_request_ref"] == copied
    assert candidate["derived_request_ref"] == document_ref(
        run_path(copied, "request.json"), candidate["derived_request"]
    )


def test_initial_resolution_requires_current_observation_and_reports_missing_dates(
    tmp_path, monkeypatch
):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    collection_ref = request["recipe_ref"]
    collection = json.loads((tmp_path / collection_ref["path"]).read_bytes())
    del collection["recipes"]["20260828"]
    request["recipe_ref"] = put(tmp_path, collection_ref["path"], collection)
    ref = put(tmp_path, ref["path"], request)
    before = snapshot(tmp_path)
    assert resolve(tmp_path, ref, request)["missing_input_dates"] == ["20260828"]
    with pytest.raises(ContractError, match="AUTO_FRESH_CALENDAR_REQUIRED"):
        resolve_automatic_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
            now=datetime(2026, 8, 29, 14, tzinfo=timezone.utc),
        )
    assert snapshot(tmp_path) == before
