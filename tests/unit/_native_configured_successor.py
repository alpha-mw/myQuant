"""Continue the retained installed scenario through its unchanged configured entry.

Only synthetic input generation, external transports and logical clocks are
controlled. No Factor/Core/EOD producer or validation gate is replaced. The
configuration and its pinned risk-free input are retained unchanged across days;
the native yield reader decides whether that existing source remains usable.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

SEQUENCE = ("20260827", "20260828", "20260831", "20260901", "20260902")
RECOVERY_SEQUENCE = (*SEQUENCE, "20260903")


def proof_path(root, day):
    suffix = "" if day == SEQUENCE[0] else "-" + day
    return root / ("configured-native-proof" + suffix + ".json")


@contextmanager
def observe_completion_replays(workspace, day):
    """Record real validator returns; never return a retained result to the runtime."""
    from scripts import daily_completion_replay, daily_production, daily_catchup

    original = daily_completion_replay.replay_native_completion
    completed = []
    calls = [0]

    def observed(*args, **kwargs):
        started = time.monotonic()
        calls[0] += 1
        call = calls[0]
        print("START native full replay", kwargs.get("trade_date"), "call", call, flush=True)
        value = original(*args, **kwargs)
        print(
            "PASS native full replay",
            kwargs.get("trade_date"),
            "call",
            call,
            "seconds",
            round(time.monotonic() - started, 2),
            flush=True,
        )
        if kwargs.get("workspace") == str(workspace) and kwargs.get("trade_date") == day:
            completed.append(deepcopy(value))
        return value

    with ExitStack() as stack:
        for module in (daily_completion_replay, daily_production, daily_catchup):
            if module.replay_native_completion is not original:
                raise ValueError("native replay observer found an already replaced callable")
            stack.enter_context(patch.object(module, "replay_native_completion", observed))
        yield completed


from _native_synthetic_clock import synthetic_clock


def run(root, dependencies, day, *, interrupt_top100=False, unresolved_corporate=False):
    from _native_daily_calendar_fixture import configured_future_calendar_scope

    with configured_future_calendar_scope(root=root, trade_date=day):
        return _run_inner(
            root,
            dependencies,
            day,
            interrupt_top100=interrupt_top100,
            unresolved_corporate=unresolved_corporate,
        )


def _run_inner(root, dependencies, day, *, interrupt_top100=False, unresolved_corporate=False):
    import quant_investor

    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("verified installed runtime required")
    sequence = RECOVERY_SEQUENCE if interrupt_top100 else SEQUENCE
    if day not in sequence[1:] or (interrupt_top100 and day != RECOVERY_SEQUENCE[-1]):
        raise ValueError("explicit consecutive synthetic successor required")
    if unresolved_corporate and not interrupt_top100:
        raise ValueError("case6 is confined to the separate sixth-day recovery scenario")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependencies),
        ]
    )
    workspace = root / "factor-workspace"
    marker = root / ("configured-successor-started-" + day + ".json")
    if marker.exists() or (workspace / "results/operations/daily_production/CN" / day).exists():
        raise ValueError("successor already attempted; inspect the retained original request")
    previous = sequence[sequence.index(day) - 1]
    prior = json.loads(proof_path(root, previous).read_bytes())
    initial = json.loads(proof_path(root, SEQUENCE[0]).read_bytes())

    def read(reference):
        raw = (workspace / reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("retained configured source changed: " + reference["path"])
        return json.loads(raw)

    config_ref = initial["config_ref"]
    config = read(config_ref)
    read(config["release_install_ref"])
    from scripts.daily_completion_replay import replay_native_completion

    checked = replay_native_completion(
        workspace=str(workspace), trade_date=previous, completion_ref=prior["completion_ref"]
    )
    if not checked["synthetic"] or not checked["native_replay_validated"]:
        raise ValueError("exact previous synthetic native completion required")
    from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
    from _native_daily_store_fixture import NativeStoreFixture
    from _native_shared_macro_fixture import build_shared_macro
    from quant_investor.strategy_records import store, event_store
    from quant_investor.market import cn_benchmark_store
    from quant_investor.market.cn_history_audit import run_cn_history_audit
    from quant_investor.market.daily_macro_layout import select_macro_layout
    from test_daily_evidence_requested_session import capture
    from scripts.daily_source_inputs import configured_source_inputs
    import pandas as pd

    iso = datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d")
    now = datetime.fromisoformat(iso + "T13:20:00+00:00")
    book = object.__new__(NativeStoreFixture)
    book.project, book.preserve_market = workspace, True
    book.root = workspace / "results/strategy_records/CN/aggressive_tech_manufacturing"
    if store.load_registered_catalog(book.root) is None:
        raise ValueError("native previous Store missing")
    book.policy_path = "fixtures/official-close-policy.json"
    book.policy_sha = hashlib.sha256((workspace / book.policy_path).read_bytes()).hexdigest()
    events = event_store.load_generation(book.root / "_event_store")
    book.closures, book.event_pointer = events["closures"], events["pointer_sha256"]
    book.days = [row["trade_date"] for row in book.closures]
    if not book.days or book.days[-1].replace("-", "") != previous:
        raise ValueError("native Event continuity does not end at previous day")
    book.benchmark_pointer = cn_benchmark_store.load_generation(
        workspace / "data/parquet/cn/benchmarks"
    )["pointer_sha256"]
    marker.write_text(
        json.dumps({"day": day, "previous_completion_ref": prior["completion_ref"]}) + "\n"
    )

    def save(name, value):
        (root / (name + "-" + day + ".json")).write_text(
            json.dumps(value, indent=2, default=str) + "\n"
        )
        print("PASS", day, name, flush=True)

    with synthetic_clock(now):
        fixture = NativeFactorInputs(
            workspace / "synthetic-inputs", extra_future_sessions=4 if interrupt_top100 else 3
        )
        book.stocks = tuple(fixture.symbols[:7])
        corporate_event_ref = None
        if unresolved_corporate:
            from _native_unresolved_corporate_case import prepare

            config_ref, config, corporate_event_ref = prepare(
                workspace, config_ref, day=day, symbol=book.stocks[0]
            )
        inputs = fixture.day(3 + sequence.index(day), extra_history=9)
        strict_market_from_factor_inputs(
            workspace,
            inputs,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
        dates = pd.read_parquet(inputs["exchange_calendar_path"])["open_session"].tolist()
        audit, audit_path = run_cn_history_audit(
            data_root=workspace / "data",
            output_root=workspace / "data/private/synthetic-history",
            days=100,
            end_date=day,
            allow_online=False,
            trade_dates=[d.strftime("%Y%m%d") for d in dates[-100:]],
        )
        if audit["history_audit_status"] != "passed":
            raise ValueError("native successor market audit failed")
        (root / ("successor-market-audit-" + iso + ".json")).write_text(
            json.dumps({"history": audit, "audit_path": str(audit_path)}, indent=2) + "\n"
        )
        risk_path = workspace / config["risk_free_ref"]["path"]
        risk_before = (risk_path.read_bytes(), risk_path.stat().st_mtime_ns)
        book.advance(iso, update_risk_free=False)
        assert (risk_path.read_bytes(), risk_path.stat().st_mtime_ns) == risk_before
        assert hashlib.sha256(risk_before[0]).hexdigest() == config["risk_free_ref"]["sha256"]
        layout = select_macro_layout(
            SimpleNamespace(
                workspace_root=workspace,
                run_root=workspace / "data/private/cn_daily_maintenance",
                target_date=day,
            )
        )
        save(
            "configured-native-macro-source",
            build_shared_macro(
                workspace,
                iso,
                transaction_identity=layout.transaction_id,
            ),
        )
        with patch(
            "quant_investor.operations.source_slot_inputs.acquire_close_session_authority",
            lambda **kw: capture(kw["now"].isoformat()),
        ):
            prepared = configured_source_inputs(
                workspace=str(workspace),
                config_ref=config_ref,
                release_install_ref=config["release_install_ref"],
                synthetic=True,
                mode="provision",
            )
        save("configured-source-result", prepared)
        return _execute_prepared(
            root=root,
            workspace=workspace,
            day=day,
            prior=prior,
            config_ref=config_ref,
            config=config,
            prepared=prepared,
            companies=fixture.symbols,
            now=now,
            interrupt_top100=interrupt_top100,
            corporate_event_ref=corporate_event_ref,
        )


def require_unstarted_day(workspace, day):
    from quant_investor.operations.daily_status import read_daily_status

    journal = workspace / "results/operations/daily_production/CN" / day
    if any(
        (journal / name).exists()
        for name in ("nodes", "executions", "core-handoff.v1.json", "completion.v1.json")
    ):
        raise ValueError("target day already contains business execution evidence")
    status = read_daily_status(str(workspace), day)
    if any(row["state"] != "NOT_STARTED" for row in status["nodes"].values()):
        raise ValueError("target business nodes are not all unstarted")


def run_prepared(root, dependencies, day):
    """Resume retained inputs inside the same byte-pinned offline transport scope."""
    from _native_daily_calendar_fixture import configured_future_calendar_scope

    with configured_future_calendar_scope(root=root, trade_date=day):
        return _run_prepared_inner(root, dependencies, day)


def _run_prepared_inner(root, dependencies, day):
    """Continue the retained pre-dispatch test stop without restaging any sources."""
    import quant_investor
    import pandas as pd

    if day != SEQUENCE[1]:
        raise ValueError("only the recorded first-successor preparation stop is supported")
    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("verified installed runtime required")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependencies),
        ]
    )
    stopped = json.loads((root / "configured-successors-status.json").read_bytes())
    if (
        stopped["state"] != "FAIL"
        or stopped["helper_drift"]
        or stopped["steps"][-1]["stage"] != day
        or stopped["steps"][-1]["exit_code"] != 1
    ):
        raise ValueError("exact retained stopped preparation required")
    if (
        proof_path(root, day).exists()
        or (root / ("configured-native-eod-result-" + day + ".json")).exists()
    ):
        raise ValueError("prepared target already has an execution result")
    workspace = root / "factor-workspace"
    require_unstarted_day(workspace, day)
    initial = json.loads(proof_path(root, SEQUENCE[0]).read_bytes())
    marker = json.loads((root / ("configured-successor-started-" + day + ".json")).read_bytes())
    if marker != {"day": day, "previous_completion_ref": initial["completion_ref"]}:
        raise ValueError("prepared successor has a different original predecessor")
    prepared = json.loads((root / ("configured-source-result-" + day + ".json")).read_bytes())

    def read(reference):
        raw = (workspace / reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("prepared source bytes changed")
        return json.loads(raw)

    config_ref = initial["config_ref"]
    config = read(config_ref)
    if (
        prepared["mode"] != "REQUEST_AVAILABLE"
        or prepared["trade_date"] != day
        or prepared["config_ref"] != config_ref
        or prepared["release_install_ref"] != config["release_install_ref"]
    ):
        raise ValueError("retained preparation differs from the original configuration")
    for key in ("request_ref", "calendar_capture_ref", "preparation_commitment_ref"):
        read(prepared[key])
    risk = workspace / config["risk_free_ref"]["path"]
    if hashlib.sha256(risk.read_bytes()).hexdigest() != config["risk_free_ref"]["sha256"]:
        raise ValueError("original configured risk-free input changed")
    companies = sorted(
        pd.read_parquet(workspace / "synthetic-inputs" / day / "pit.parquet")["symbol"].tolist()
    )
    if not companies or len(set(companies)) != len(companies):
        raise ValueError("retained staged synthetic PIT is invalid")
    checkpoint = {
        "state": "CONTINUING_EXISTING_PREPARED_REQUEST_BEFORE_ANY_BUSINESS_NODE",
        "trade_date": day,
        "request_ref": prepared["request_ref"],
        "config_ref": config_ref,
        "previous_completion_ref": initial["completion_ref"],
        "harness_interruption_disclosed": True,
        "business_nodes_started_before_resume": False,
        "source_staging_repeated": False,
        "previous_native_checks_completed_in_original_attempt": True,
        "independent_final_five_day_replay_required": True,
    }
    (root / ("prepared-successor-resume-" + day + ".json")).write_text(
        json.dumps(checkpoint, indent=2) + "\n"
    )
    iso = datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d")
    now = datetime.fromisoformat(iso + "T13:20:00+00:00")
    with synthetic_clock(now):
        return _execute_prepared(
            root=root,
            workspace=workspace,
            day=day,
            prior=initial,
            config_ref=config_ref,
            config=config,
            prepared=prepared,
            companies=companies,
            now=now,
            resumed_preparation=True,
        )


def _execute_prepared(
    *,
    root,
    workspace,
    day,
    prior,
    config_ref,
    config,
    prepared,
    companies,
    now,
    interrupt_top100=False,
    corporate_event_ref=None,
    resumed_preparation=False,
):
    from _native_daily_maintenance_fixture import maintenance
    from quant_investor.market.daily_components import build_default_components
    from quant_investor.operations import theme_capture_stage
    from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
    from test_tushare_theme_capture_stable import FakeClient
    from scripts.daily_production import dispatch_daily_request
    from scripts.daily_completion_replay import replay_native_completion

    repository = root / "repository"
    iso = datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d")

    def read(reference):
        raw = (workspace / reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("retained configured source changed: " + reference["path"])
        return json.loads(raw)

    def save(name, value):
        (root / (name + "-" + day + ".json")).write_text(
            json.dumps(value, indent=2, default=str) + "\n"
        )
        print("PASS", day, name, flush=True)

    request_ref = prepared["request_ref"]
    request = read(request_ref)
    if (
        request["schema_version"] != "cn-daily-automatic-request.v2"
        or request["action"] != "CATCH_UP"
    ):
        raise ValueError("successor must select the native automatic configured entry")
    require_unstarted_day(workspace, day)
    components = build_default_components(workspace_root=workspace)

    def maintained(**kwargs):
        return maintenance(
            root,
            iso,
            core_completed=kwargs["core_completed"],
            expected_target_trade_date=kwargs.get("_expected_target_trade_date"),
            core_replay_completed=kwargs.get("_core_replay_completed"),
            macro_callback=components.macro_release,
            now=now,
        )["maintenance_result"]

    theme = next(t.split(":", 1)[1] for t in TECHNOLOGY_THEME_IDS if t.startswith("TUSHARE_DC:"))
    rows = {("dc_index", "ALL"): [(theme, day, "synthetic", "概念板块", "1")]}
    rows.update({("dc_member", c): [(day, theme, c, "synthetic")] for c in companies})
    client = FakeClient(rows)
    with (
        patch("quant_investor.market.daily_maintenance.run_cn_daily_maintenance", maintained),
        patch.object(theme_capture_stage.native, "OfficialTushareHttpsClient", lambda **kw: client),
        observe_completion_replays(workspace, day) as observed_replays,
    ):
        recovery = None
        if interrupt_top100:
            from _native_top100_recovery_case import dispatch_with_interruption

            result, recovery = dispatch_with_interruption(
                workspace=str(workspace),
                request_ref=request_ref,
                release_install_ref=config["release_install_ref"],
                day=day,
            )
            save("configured-top100-recovery-boundary", recovery)
        else:
            result = dispatch_daily_request(
                workspace=str(workspace),
                request_ref=request_ref,
                release_install_ref=config["release_install_ref"],
                synthetic=True,
            )
    save("configured-native-eod-result", result)
    execution = result["result"]
    if execution["business_state"] != "COMPLETE" or result["ordered_trade_dates"] != [day]:
        raise ValueError("native configured successor incomplete or skipped a day")
    completion_ref = execution["days"][0]["completion_ref"]
    matched = [value for value in observed_replays if value.get("completion_ref") == completion_ref]
    if matched:
        replay = matched[-1]
        replay_origin = "ACTUAL_NATIVE_FULL_REPLAY_RETURN_INSIDE_SUCCESSFUL_DISPATCH"
    else:
        replay = replay_native_completion(
            workspace=str(workspace), trade_date=day, completion_ref=completion_ref
        )
        replay_origin = "SEPARATE_NATIVE_FULL_REPLAY_AFTER_DISPATCH"
    from quant_investor.operations.daily_contract import EOD_NODE_IDS

    if (
        replay["synthetic"] is not True
        or replay["native_replay_validated"] is not True
        or replay["trade_date"] != day
        or replay["completion_ref"] != completion_ref
        or replay["validated_nodes"] != sorted(EOD_NODE_IDS)
    ):
        raise ValueError("native successor completion replay failed")
    proof = {
        "synthetic": True,
        "production_deployed": False,
        "real_provider_calls": False,
        "config_ref": config_ref,
        "request_ref": request_ref,
        "completion_ref": completion_ref,
        "previous_completion_ref": prior["completion_ref"],
        "replay": replay,
        "replay_evidence_origin": replay_origin,
        "observed_current_day_native_replay_returns": len(matched),
        "native_replay_calls_forwarded_unchanged": True,
        "independent_five_day_historical_replay_required": True,
        "unchanged_configuration": corporate_event_ref is None,
        "harness_resumed_after_preparation": resumed_preparation,
        "pre_core_configuration": True,
        "factor_core_and_release_verifier_mocked": False,
        "risk_free_input_retained": True,
        "theme_transport_calls": len(client.calls),
    }
    save("configured-native-proof", proof)
    if recovery is not None:
        save(
            "configured-top100-recovery-proof",
            {
                **recovery,
                "full_eod_verified_by_caller": True,
                "completion_ref": completion_ref,
                "native_replay": replay,
            },
        )
    if corporate_event_ref is not None:
        from _native_unresolved_corporate_case import verify

        save(
            "configured-unresolved-corporate-proof",
            verify(
                workspace,
                completion_ref=completion_ref,
                replay=replay,
                event_ref=corporate_event_ref,
            ),
        )
    return proof


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]), indent=2))
