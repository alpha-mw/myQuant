"""Native Dashboard semantic replay bound to an exact recorded EOD completion."""

from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo
from typing import Any

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_journal import DailyJournal, request_identity
from quant_investor.operations.daily_contract import ContractError, validate_ref, utc_stamp
from quant_investor.operations.dashboard_replay_sources import (
    RetainedDashboardSources,
    retained_dashboard_sources,
    retained_store_inventory,
    RetainedDashboardMarket,
)
from scripts.daily_dashboard_capture import DailyDashboardCapture
from scripts.daily_completion_store import replay_completed_store
from scripts.daily_dashboard_adapter import dashboard_publication_bytes
from cn_dashboard_common import build_bundle
from cn_dashboard_v2 import build_v2_bundle
from export_cn_aggressive_dashboard_data import _render_json, _expected_output_paths


def replay_completed_dashboard(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    root = Path(workspace).resolve(strict=True)
    records = root / "results/strategy_records/CN/aggressive_tech_manufacturing"
    storage = SecureSystemStorage(workspace)
    seen = {}

    def read(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_DASHBOARD_SOURCE_SHA_MISMATCH")
        seen[ref["path"]] = raw.data
        return parse_canonical_json_bytes(raw.data)

    inputs = read(recorded["native_inputs_ref"])
    sealed_profile = inputs["schema_version"] in {
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }
    registered_profile = inputs["schema_version"] == "cn-daily-native-inputs.v7"
    terminal_ref = recorded["node_terminal_refs"]["dashboard"]
    terminal = read(terminal_ref)
    path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
    raw = storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
    request_ref = {"path": path, "sha256": raw.byte_sha256}
    request = read(request_ref)
    if request_identity(request)[1] != terminal["request_key"]:
        raise ContractError("EOD_DASHBOARD_REQUEST_INVALID")
    refs = {k: request["input_refs"][k] for k in ("store_plan", "market", "benchmark", "risk_free")}
    if refs != {
        "store_plan": inputs["store_plan_ref"],
        "market": inputs["market_snapshot_ref"],
        "benchmark": inputs["benchmark_ref"],
        "risk_free": inputs["risk_free_ref"],
    }:
        raise ContractError("EOD_DASHBOARD_SOURCE_BINDING_INVALID")
    intent = read(request["input_refs"]["render_intent"])
    historical = not inputs["publish_current_dashboard"]
    if intent != {
        "generated_at": intent["generated_at"],
        "historical_mode": historical,
        "trade_date": trade_date,
        "sources": refs,
        "release_ref": recorded["release_ref"],
    }:
        raise ContractError("EOD_DASHBOARD_INTENT_INVALID")
    now = utc_stamp(intent["generated_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    cap = DailyDashboardCapture(workspace, DailyJournal(workspace, trade_date))
    request_ref = {
        "path": str(
            cap.journal.root / "inputs" / ("dashboard-request-" + terminal["request_key"] + ".json")
        ),
        "sha256": terminal["request_key"],
    }
    if read(request_ref) != request:
        raise ContractError("EOD_DASHBOARD_CAPTURE_REQUEST_DIFFERS")
    captured = cap.read(request_ref=request_ref)
    if captured is None or read(terminal["output_refs"]["capture"]) != captured:
        raise ContractError("EOD_DASHBOARD_CAPTURE_BINDING_INVALID")
    if terminal["output_refs"]["capture"]["path"] != cap.receipt_path:
        raise ContractError("EOD_DASHBOARD_CAPTURE_PATH_INVALID")
    if (
        not utc_stamp(intent["generated_at"])
        <= utc_stamp(captured["captured_at"])
        <= utc_stamp(terminal["finished_at"])
    ):
        raise ContractError("EOD_DASHBOARD_CUSTODY_INVALID")
    store = replay_completed_store(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    pointer_ref = store["output_refs"]["pointer"]
    sources = {
        row["source_ref"]["path"]: cap._read(row["retained_ref"]) for row in captured["sources"]
    }
    inventory = retained_store_inventory(
        project_root=root, record_root=records, pointer_ref=pointer_ref
    )
    for logical, data in inventory.items():
        if logical in sources and sources[logical] != data:
            raise ContractError("EOD_DASHBOARD_RETAINED_SOURCE_CONFLICT")
        sources[logical] = data
    iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    history_args: dict[str, Any] = (
        {"historical_close_plan_ref": refs["store_plan"], "historical_valuation_date": iso}
        if historical
        else {}
    )
    with retained_dashboard_sources(RetainedDashboardSources(root, sources, pointer_ref)):
        v1 = build_bundle(
            project_root=root,
            record_root=records,
            benchmark_path=root / refs["benchmark"]["path"],
            risk_free_path=root / refs["risk_free"]["path"],
            generated_at=now.isoformat(timespec="seconds"),
            today=now.date(),
            **history_args,
        )
        market = (
            MarketDataReader(
                market="CN",
                data_root=root / "data",
                mode_policy="strict",
                frozen_snapshot_ref={
                    "path": str(Path(refs["market"]["path"]).relative_to("data")),
                    "sha256": refs["market"]["sha256"],
                },
            )
            if historical
            else RetainedDashboardMarket(project_root=root, manifest_ref=refs["market"])
        )
        v2 = build_v2_bundle(
            project_root=root,
            v1_bundle=v1,
            v1_json_path=_expected_output_paths(root)[0],
            record_root=records,
            generation_local_date=now.date(),
            generated_at=now.isoformat(timespec="seconds"),
            publication_attempt_id="dashboard-v2-dag-"
            + trade_date
            + "-"
            + request_ref["sha256"][:16],
            market_reader=market,
            v1_json_bytes_override=_render_json(v1),
            **history_args,
        )
    if _render_json(v1) != cap._read(captured["v1_ref"]) or _render_json(v2) != cap._read(
        captured["v2_ref"]
    ):
        raise ContractError("EOD_DASHBOARD_NATIVE_REBUILD_DIFFERS")
    expected = {
        "capture": terminal["output_refs"]["capture"],
        "v1": captured["v1_ref"],
        "v2": captured["v2_ref"],
    }
    daily_evidence = None
    registered_view = None
    if sealed_profile:
        from quant_investor.operations.dashboard_evidence import DashboardEvidenceSources, DOMAINS
        from quant_investor.operations.dashboard_serving_contract import POLICY

        recipe_ref = request["input_refs"]["daily_evidence_recipe"]
        recipe = read(recipe_ref)
        terminal_refs = {name: recorded["node_terminal_refs"][name] for name in DOMAINS}
        identity = {
            "schema_version": (
                "cn-daily-dashboard-evidence-recipe.v2"
                if registered_profile
                else "cn-daily-dashboard-evidence-recipe.v1"
            ),
            "trade_date": trade_date,
            "release_ref": recorded["release_ref"],
            "terminal_refs": terminal_refs,
            "publication_policy": POLICY,
        }
        if registered_profile:
            identity.update(
                corporate_terminal_ref=recorded["node_terminal_refs"]["corporate_action_recon"],
                store_plan_ref=inputs["store_plan_ref"],
                registered_event_declaration_ref=inputs["registered_event_declaration_ref"],
            )
            if (
                request["input_refs"].get("registered.corporate_terminal")
                != identity["corporate_terminal_ref"]
            ):
                raise ContractError("EOD_DASHBOARD_REGISTERED_SOURCE_MISMATCH")
        elif any(key.startswith("registered.") for key in request["input_refs"]):
            raise ContractError("EOD_DASHBOARD_UNEXPECTED_REGISTERED_PROFILE")
        if (
            set(recipe) != {*identity, "created_at"}
            or any(recipe[k] != v for k, v in identity.items())
            or any(
                request["input_refs"].get("authority." + k) != v for k, v in terminal_refs.items()
            )
            or inputs["dashboard_publication_policy"] != POLICY
        ):
            raise ContractError("EOD_DASHBOARD_DAILY_EVIDENCE_BINDING_INVALID")
        import hashlib
        from quant_investor.contracts import canonical_json_bytes

        key = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        if recipe_ref["path"] != str(
            cap.journal.root / "inputs" / f"dashboard-evidence-{key}.json"
        ):
            raise ContractError("EOD_DASHBOARD_EVIDENCE_RECIPE_PATH_INVALID")
        if utc_stamp(recipe["created_at"]) > utc_stamp(terminal["finished_at"]):
            raise ContractError("EOD_DASHBOARD_EVIDENCE_CUSTODY_INVALID")
        daily_evidence = DashboardEvidenceSources(
            workspace=root,
            trade_date=trade_date,
            release_ref=recorded["release_ref"],
            terminal_refs=terminal_refs,
        ).build(recipe["created_at"])
        sha = hashlib.sha256(canonical_json_bytes(daily_evidence)).hexdigest()
        expected["daily_evidence"] = {
            "path": str(cap.journal.root / "dashboard" / f"daily-evidence-{sha}.json"),
            "sha256": sha,
        }
        if read(expected["daily_evidence"]) != daily_evidence:
            raise ContractError("EOD_DASHBOARD_DAILY_EVIDENCE_DIFFERS")
        if registered_profile:
            from quant_investor.operations.registered_dashboard import RegisteredDashboardSources
            from quant_investor.operations.completion_corporate import (
                replay_completed_corporate_actions,
            )

            corporate = replay_completed_corporate_actions(
                workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
            )
            registered_view = RegisteredDashboardSources(
                workspace=root,
                trade_date=trade_date,
                release_ref=recorded["release_ref"],
                corporate_terminal_ref=identity["corporate_terminal_ref"],
                store_terminal_ref=terminal_refs["store"],
                store_plan_ref=inputs["store_plan_ref"],
                registered_event_declaration_ref=inputs["registered_event_declaration_ref"],
            ).result(recipe["created_at"])
            if (
                corporate["projection"].get("registered_transition_ref")
                != registered_view["registered_transition_ref"]
            ):
                raise ContractError("EOD_DASHBOARD_REGISTERED_CORPORATE_MISMATCH")
            expected["registered_transition"] = registered_view["registered_transition_ref"]
    if not historical and not sealed_profile:
        publication = read(terminal["output_refs"]["publication"])
        if (
            set(publication) != {"schema_version", "request_ref", "published_at", "files"}
            or publication["schema_version"] != "cn-daily-dashboard-publication.v1"
            or publication["request_ref"] != request_ref
        ):
            raise ContractError("EOD_DASHBOARD_PUBLICATION_INVALID")
        if (
            not utc_stamp(captured["captured_at"])
            <= utc_stamp(publication["published_at"])
            <= utc_stamp(terminal["finished_at"])
        ):
            raise ContractError("EOD_DASHBOARD_PUBLICATION_TIME_INVALID")
        rendered = dashboard_publication_bytes(
            workspace=root, v1=v1, v2=v2, raw_v1=_render_json(v1), raw_v2=_render_json(v2)
        )
        if len(publication["files"]) != len(rendered):
            raise ContractError("EOD_DASHBOARD_PUBLICATION_SET_INVALID")
        for row, (rendered_path, data) in zip(publication["files"], rendered.items()):
            if (
                set(row) != {"path", "retained_ref"}
                or row["path"] != str(rendered_path.relative_to(root))
                or cap._read(row["retained_ref"]) != data
            ):
                raise ContractError("EOD_DASHBOARD_PUBLICATION_BYTES_INVALID")
        expected["publication"] = terminal["output_refs"]["publication"]
    if terminal["output_refs"] != expected:
        raise ContractError("EOD_DASHBOARD_OUTPUT_SET_INVALID")
    if (
        retained_store_inventory(project_root=root, record_root=records, pointer_ref=pointer_ref)
        != inventory
    ):
        raise ContractError("EOD_DASHBOARD_STORE_CHANGED_DURING_REPLAY")
    for path, data in seen.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=16 * 1024 * 1024).data != data:
            raise ContractError("EOD_DASHBOARD_SOURCE_CHANGED_DURING_REPLAY")
    if (
        cap.read(request_ref=request_ref) != captured
        or inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_DASHBOARD_COMPLETION_CHANGED_DURING_REPLAY")
    result = {
        "completion_ref": dict(completion_ref),
        "v1": v1,
        "v2": v2,
        "validation_scope": "COMPLETED_DASHBOARD_NATIVE_REBUILD",
        "consumer_admission": False,
    }
    if daily_evidence is not None:
        result["daily_evidence"] = daily_evidence
    if registered_view is not None:
        result.update(registered_view)
    return result
