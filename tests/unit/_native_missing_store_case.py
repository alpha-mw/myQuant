"""Interrupt before native Store publication, then recover the prior day exactly."""

from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from unittest.mock import patch


def inventory(root):
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def validate_interrupted_boundary(status, projection):
    """An exception fails this invocation but leaves immutable write custody in doubt."""
    stored = status["nodes"]["store"]
    invoked = projection["nodes"]["store"]
    if (
        stored["state"] != "RUNNING"
        or stored.get("recovery_state") != "IN_DOUBT"
        or stored.get("terminal_ref") is not None
        or stored.get("output_refs") != {}
        or invoked["state"] != "FAILED"
        or invoked.get("blocking_reason") != "POST_WRITE_IN_DOUBT"
        or invoked.get("start_ref") != stored["start_ref"]
        or invoked.get("request_key") != stored["request_key"]
        or any(
            row["state"] != "SUCCEEDED"
            for name, row in status["nodes"].items()
            if name not in {"store", "dashboard"}
        )
    ):
        raise AssertionError("Store interruption did not leave the intended native boundary")


def dispatch_case(
    *, root, workspace, request_ref, config_ref, release_install_ref, resume_retained=False
):
    from _native_synthetic_clock import synthetic_clock
    from scripts.daily_production import dispatch_daily_request
    from scripts.daily_production_store_adapter import StoreCloseAdapter
    from scripts.daily_completion_replay import replay_native_completion
    from scripts.daily_dashboard_publication import observed_serving_status
    from scripts import cn_official_close_batch as native_close
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.operations.daily_contract import EOD_NODE_IDS
    from quant_investor.strategy_records import store, performance
    from quant_investor.intelligence.storage import DailyResearchPoolStore
    from quant_investor.factors.governance import factor_production_prepare
    from quant_investor.operations import theme_capture_stage

    day = "20260827"
    arguments = dict(
        workspace=str(workspace),
        request_ref=request_ref,
        release_install_ref=release_install_ref,
        synthetic=True,
    )
    interrupted = []

    def stop_before_store(adapter, request):
        if adapter.trade_date != day:
            raise AssertionError("unexpected Store target during explicit fault")
        interrupted.append(day)
        (root / "missing-store-injection.json").write_text(
            json.dumps({"trade_date": day, "store_request": request, "before_native_writer": True})
            + "\n"
        )
        raise RuntimeError("SYNTHETIC_STOP_BEFORE_NATIVE_STORE_PUBLICATION")

    first = None
    if resume_retained:
        previous = json.loads((root / "missing-store-driver-status.json").read_bytes())
        if previous["state"] != "FAIL" or previous["helper_drift"]:
            raise AssertionError("retained continuation requires the exact stopped test state")
    else:
        with patch.object(StoreCloseAdapter, "execute", stop_before_store):
            first = dispatch_daily_request(**arguments)
        (root / "missing-store-first-result.json").write_text(json.dumps(first, indent=2) + "\n")
        if not interrupted or first["business_state"] != "INCOMPLETE":
            raise AssertionError("planned Store interruption was not observed")
    status = read_daily_status(str(workspace), day)
    journal = workspace / "results/operations/daily_production/CN" / day
    projection_raw = (journal / "dag-status.v1.json").read_bytes()
    projection = json.loads(projection_raw)
    validate_interrupted_boundary(status, projection)
    (root / "missing-store-original-dag-status.json").write_bytes(projection_raw)
    if (journal / "completion.v1.json").exists():
        raise AssertionError("interrupted Store unexpectedly produced a full EOD")
    execution = journal / "executions" / request_ref["sha256"]
    retained_paths = [execution / "research-cutoff.v1.json", execution / "materialization.v5.json"]
    retained = {
        str(p.relative_to(workspace)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in retained_paths
    }

    def read_ref(ref):
        raw = (workspace / ref["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise AssertionError("retained native input changed before Store recovery")
        return json.loads(raw)

    materialization = json.loads(retained_paths[1].read_bytes())
    native_input = read_ref(materialization["native_inputs_ref"])
    plan_ref = native_input["store_plan_ref"]
    plan = read_ref(plan_ref)
    books = workspace / "results/strategy_records/CN/aggressive_tech_manufacturing"
    current = books / "_record_store/current.v1.json"
    if (
        hashlib.sha256(current.read_bytes()).hexdigest()
        != plan["preimages"]["store_pointer_sha256"]
    ):
        raise AssertionError("Store no longer matches the original pre-publication head")
    transaction = (workspace / plan_ref["path"]).parent
    native_close._read_close_source_pointer(books, plan)
    if (transaction / "source-pointer.v1.json").read_bytes() != current.read_bytes():
        raise AssertionError("retained native source pointer differs from the original head")
    if any(
        (transaction / name).exists()
        for name in ("committed-pointer.v1.json", "completion.v1.json")
    ):
        raise AssertionError("retained fault was not before native financial publication")
    pointer, catalog = store.load_registered_catalog(books)
    history = performance.load_performance_history(books, catalog["performance_history_ref"])
    if history["rows"][-1]["valuation_date"] != "2026-08-26":
        raise AssertionError("the missing prior Store day was not established")
    factor_before = inventory(workspace / "results/factors")
    pool_before = inventory(workspace / "results/intelligence/research_pool")
    upstream = {
        name: row["terminal_ref"]
        for name, row in status["nodes"].items()
        if name not in {"store", "dashboard"}
    }
    checkpoint = {
        "case": 4,
        "state": "ORIGINAL_CUTOFF_SEALED_STORE_NOT_PUBLISHED",
        "synthetic": True,
        "original_request_ref": request_ref,
        "config_ref": config_ref,
        "first_result": first,
        "original_public_result_preserved": first is not None,
        "continued_from_retained_test_failure": resume_retained,
        "original_dag_projection_sha256": hashlib.sha256(projection_raw).hexdigest(),
        "store_journal_state": "RUNNING",
        "store_recovery_state": "IN_DOUBT",
        "store_invocation_state": "FAILED",
        "original_store_preimage_still_selected": True,
        "retained_native_source_pointer_validated": True,
        "target_financial_publication_custody_absent": True,
        "store_last_valuation_date": "2026-08-26",
        "missing_store_date": day,
        "retained_cutoff_and_materialization": retained,
        "successful_upstream_terminal_refs": upstream,
    }
    (root / "missing-store-interruption.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    print("PASS original cutoff sealed; Aug27 Store absent before Aug28 recovery", flush=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("prior Store recovery attempted an upstream producer")

    with synthetic_clock(datetime.fromisoformat("2026-08-28T13:20:00+00:00")), ExitStack() as stack:
        stack.enter_context(
            patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", forbidden)
        )
        stack.enter_context(
            patch.object(factor_production_prepare, "prepare_factor_production", forbidden)
        )
        stack.enter_context(patch.object(DailyResearchPoolStore, "publish", forbidden))
        stack.enter_context(
            patch.object(theme_capture_stage.native, "OfficialTushareHttpsClient", forbidden)
        )
        cas = stack.enter_context(patch.object(store, "_cas_pointer", wraps=store._cas_pointer))
        resumed = dispatch_daily_request(**arguments, committed_recovery_only=True)
        (root / "missing-store-resumed-result.json").write_text(
            json.dumps(resumed, indent=2) + "\n"
        )
        completion_path = journal / "completion.v1.json"
        completion_raw = completion_path.read_bytes()
        completion_ref = {
            "path": str(completion_path.relative_to(workspace)),
            "sha256": hashlib.sha256(completion_raw).hexdigest(),
        }
        replay = replay_native_completion(
            workspace=str(workspace), trade_date=day, completion_ref=completion_ref
        )
        if (
            replay["native_replay_validated"] is not True
            or replay["validated_nodes"] != sorted(EOD_NODE_IDS)
            or replay["synthetic"] is not True
            or cas.call_count != 1
        ):
            raise AssertionError("prior-day recovery lacked complete native proof or exact one CAS")
        serving = observed_serving_status(str(workspace), day)
        if resumed["business_state"] != "COMPLETE" and (
            resumed["business_state"] != "INCOMPLETE"
            or serving["publication_state"]
            not in {"EVIDENCE_SEALED_PUBLICATION_EXPIRED", "RECORDED_EOD_PUBLICATION_EXPIRED"}
        ):
            raise AssertionError("recovery has an unexplained non-complete public outcome")
        if resumed["business_state"] != "COMPLETE":
            latest = json.loads((journal / "dag-status.v1.json").read_bytes())
            if (
                latest.get("completion_status") != "EVIDENCE_SEALED_PUBLICATION_EXPIRED"
                or latest.get("publication_error", {}).get("detail")
                not in {"EVIDENCE_SEALED_PUBLICATION_EXPIRED", "DASHBOARD_PUBLICATION_EXPIRED"}
                or latest.get("sealed_evidence_ref") != completion_ref
            ):
                raise AssertionError("recorded expiry must not conceal another publication error")
    after = read_daily_status(str(workspace), day)
    if any(after["nodes"][name]["terminal_ref"] != ref for name, ref in upstream.items()):
        raise AssertionError("prior-day Store recovery changed completed upstream terminals")
    if (
        inventory(workspace / "results/factors") != factor_before
        or inventory(workspace / "results/intelligence/research_pool") != pool_before
    ):
        raise AssertionError("prior-day Store recovery changed Factor or Top100 outputs")
    for path, expected in retained.items():
        p = workspace / path
        if (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) != expected:
            raise AssertionError("recovery replaced original cutoff or materialization")
    pointer, catalog = store.load_registered_catalog(books)
    history = performance.load_performance_history(books, catalog["performance_history_ref"])
    dates = [row["valuation_date"] for row in history["rows"]]
    if dates[-1] != "2026-08-27" or dates.count("2026-08-27") != 1:
        raise AssertionError("prior-day Store recovery skipped or duplicated the missing day")
    proof = {
        "case": 4,
        "synthetic": True,
        "original_request_ref": request_ref,
        "config_ref": config_ref,
        "completion_ref": completion_ref,
        "recovery_observed_at_simulated": "2026-08-28T13:20:00Z",
        "closed_valuation_date": "2026-08-27",
        "native_financial_cas_count": 1,
        "upstream_producers_forbidden": True,
        "original_upstream_cutoff_and_pool_unchanged": True,
        "continued_from_retained_test_failure": resume_retained,
        "original_public_result_preserved": first is not None,
        "public_result": resumed,
        "serving": serving,
        "replay": replay,
        "production_deployed": False,
    }
    (root / "configured-missing-store-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        "PASS full native missing-prior-Store recovery with exactly one financial publication",
        flush=True,
    )
    return proof
