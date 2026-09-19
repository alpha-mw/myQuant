"""Prepare the existing native Store plan from recipe preimages and core handoff."""

from pathlib import Path
from typing import Mapping

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_production_store_adapter import RECORD_ROOT, prepare_store_plan, native


def verify_initial_store_controls(*, workspace: str, recipe: dict) -> dict:
    """Replay existing Store inputs before maintenance, without planning or locks."""
    root = Path(workspace).resolve(strict=True)
    record_root = root / RECORD_ROOT
    expected_paths = {
        "store_pointer_ref": RECORD_ROOT + "/_record_store/current.v1.json",
        "event_pointer_ref": RECORD_ROOT + "/_event_store/current.v1.json",
        "benchmark_pointer_ref": "data/parquet/cn/benchmarks/_latest.json",
    }
    refs = recipe["store_preimages"]
    if type(refs) is not dict or set(refs) != set(expected_paths):
        raise ContractError("EXECUTION_STORE_PREIMAGES_INVALID")

    def recheck():
        for name, path in expected_paths.items():
            ref = validate_ref(refs[name])
            if ref["path"] != path:
                raise ContractError("EXECUTION_STORE_PREIMAGE_PATH_INVALID")
            if native._pointer_sha(root / path) != ref["sha256"]:
                raise ContractError("EXECUTION_STORE_PREIMAGE_CHANGED")

    recheck()
    loaded = native.load_registered_catalog(record_root)
    if loaded is None or loaded[1].get("schema_id") != native.CATALOG_SCHEMA_V3:
        raise ContractError("EXECUTION_STORE_V3_REQUIRED")
    policy_ref = validate_ref(recipe["policy_refs"]["store"])
    policy, _ = native._policy(root, policy_ref["path"], policy_ref["sha256"])
    retrospective = recipe["retrospective_ref"]
    if retrospective is not None:
        validate_ref(retrospective)
    retrospective_values = native._retrospective(
        root,
        retrospective["path"] if retrospective else None,
        retrospective["sha256"] if retrospective else None,
    )
    benchmark = native.load_benchmarks(root / "data/parquet/cn/benchmarks")
    event = native.load_events(record_root / "_event_store")
    if (
        benchmark["pointer_sha256"] != refs["benchmark_pointer_ref"]["sha256"]
        or event["pointer_sha256"] != refs["event_pointer_ref"]["sha256"]
    ):
        raise ContractError("EXECUTION_STORE_NATIVE_PREIMAGE_CHANGED")
    performance = native.load_performance_history(record_root, loaded[1]["performance_history_ref"])
    official_date = str(performance["rows"][-1]["valuation_date"])
    if recipe["schema_version"] == "cn-daily-execute-recipe.v6":
        from scripts.registered_daily_event_sources import read_recipe_source

        proof = read_recipe_source(workspace=root, recipe=recipe)
        official_date = proof["baseline"]["record"]["data_date"]
    recheck()
    if native._policy(root, policy_ref["path"], policy_ref["sha256"])[0] != policy:
        raise ContractError("EXECUTION_STORE_POLICY_CHANGED")
    if (
        native._retrospective(
            root,
            retrospective["path"] if retrospective else None,
            retrospective["sha256"] if retrospective else None,
        )
        != retrospective_values
    ):
        raise ContractError("EXECUTION_STORE_RETROSPECTIVE_CHANGED")
    return {
        "official_date": official_date,
        "store_preimages": {name: dict(ref) for name, ref in refs.items()},
        "execution_authorized": False,
    }


def prepare_materialized_store_plan(*, journal: DailyJournal, recovered: dict) -> dict:
    """Caller owns the day lock; native planner owns the nested Store operation lock."""
    journal._require_lock()
    workspace = journal.storage._io.workspace_root
    sources = SecureSystemStorage(workspace)
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    if handoff["trade_date"] != journal.trade_date:
        raise ContractError("STORE_MATERIALIZATION_DATE_MISMATCH")
    observed = {}

    def read(ref: Mapping[str, str]):
        ref = validate_ref(ref)
        raw = sources.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("STORE_MATERIALIZATION_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = raw.data
        return raw.data

    if parse_canonical_json_bytes(read(recovered["handoff_ref"])) != handoff:
        raise ContractError("STORE_MATERIALIZATION_HANDOFF_CHANGED")
    if parse_canonical_json_bytes(read(handoff["recipe_ref"])) != recipe:
        raise ContractError("STORE_MATERIALIZATION_RECIPE_CHANGED")
    preimages = recipe["store_preimages"]
    for ref in preimages.values():
        read(ref)
    market = handoff["market_pointer_ref"]
    read(market)
    calendar = handoff["calendar_ref"]
    read(calendar)
    policy = recipe["policy_refs"]["store"]
    read(policy)
    retrospective = recipe["retrospective_ref"]
    if retrospective is not None:
        read(retrospective)
    arguments = store_materialization_arguments(workspace, handoff, recipe)
    for path, raw in observed.items():
        if sources.read_workspace_file_bytes(path, maximum_bytes=16 * 1024 * 1024).data != raw:
            raise ContractError("STORE_MATERIALIZATION_INPUT_CHANGED")
    plan = prepare_store_plan(arguments)
    if plan.get("status") not in {"PLAN_PREPARED", "PLAN_ADOPTED"}:
        raise ContractError("STORE_MATERIALIZATION_PLAN_NOT_PREPARED")
    selected = Path(plan["plan_path"])
    if selected.is_absolute():
        selected = selected.relative_to(workspace)
    ref = validate_ref({"path": selected.as_posix(), "sha256": plan["plan_sha256"]})
    read(ref)
    # The native Store format intentionally includes its terminal newline. Use
    # its owning reader rather than imposing the DAG JSON byte convention.
    value = native._load_json(
        workspace / ref["path"], expected_sha=ref["sha256"], label="materialized native Store plan"
    )
    from scripts.daily_store_adoption import verify_store_plan_binding

    binding = verify_store_plan_binding(arguments=arguments, plan_ref=ref, plan=value)
    if binding["adopted"]:
        arguments = {
            **arguments,
            "expected_store_pointer_sha": value["preimages"]["store_pointer_sha256"],
        }
    return {
        "store_plan_ref": ref,
        "native_plan": value,
        "store_arguments": arguments,
        "retained_source_pointer_ref": binding["source_pointer_ref"],
        "execution_authorized": False,
    }


def materialized_adjustment_refs(
    *, journal: DailyJournal, recovered: dict, prepared: dict, portfolio_state=None
) -> dict:
    """Capture held-security refs only from the exact frozen Market snapshot."""
    journal._require_lock()
    from quant_investor.market.market_data_reader import MarketDataReader
    from quant_investor.strategy_records.store import regular_file_sha256

    workspace = journal.storage._io.workspace_root
    arguments = prepared["store_arguments"]
    root = arguments["record_root"]
    expected_store = arguments["expected_store_pointer_sha"]
    if portfolio_state is None:
        if native._pointer_sha(root / "_record_store/current.v1.json") != expected_store:
            raise ContractError("MATERIALIZATION_HOLDINGS_POINTER_CHANGED")
        loaded = native.load_registered_catalog(root)
        if loaded is None:
            raise ContractError("MATERIALIZATION_HOLDINGS_UNREGISTERED")
        pointer, catalog = loaded
        plan = prepared["native_plan"]
        if pointer["active_record_id"] != plan["source_active_record_id"]:
            raise ContractError("MATERIALIZATION_HOLDINGS_RECORD_MISMATCH")
        rows = [
            row for row in catalog["records"] if row["record_id"] == plan["source_active_record_id"]
        ]
        if len(rows) != 1:
            raise ContractError("MATERIALIZATION_HOLDINGS_RECORD_MISSING")
        ledger, _ = native._holdings_identity(root, rows[0])
        symbols = sorted(ledger.loc[ledger["shares"] > 0, "symbol"].astype(str).tolist())
    else:
        from decimal import Decimal
        from quant_investor.intelligence.portfolio_state import validate_portfolio_state

        body = validate_portfolio_state(portfolio_state)["payload"]
        if body["store_plan_ref"] != prepared["store_plan_ref"]:
            raise ContractError("MATERIALIZATION_PORTFOLIO_PLAN_MISMATCH")
        symbols = sorted(row["symbol"] for row in body["positions"] if Decimal(row["shares"]) > 0)
        if native._plan_version(prepared["native_plan"]) == 2:
            proof = native._registered_plan_proof(root, prepared["native_plan"], retained=True)
            symbols = sorted(
                set(symbols) | {row["symbol"] for row in proof["writer"]["record"]["positions"]}
            )
    snapshot_ref = recovered["handoff"]["market_snapshot_ref"]
    relative = Path(snapshot_ref["path"]).relative_to("data").as_posix()
    market = MarketDataReader(
        market="CN",
        data_root=workspace / "data",
        mode_policy="strict",
        frozen_snapshot_ref={"path": relative, "sha256": snapshot_ref["sha256"]},
    )
    snapshot = market.snapshot()
    if (
        snapshot.get("healthy") is not True
        or snapshot.get("latest_complete_trade_date") != journal.trade_date
    ):
        raise ContractError("MATERIALIZATION_FROZEN_MARKET_INVALID")
    refs = {}
    if len(symbols) != len(set(symbols)):
        raise ContractError("MATERIALIZATION_HOLDINGS_DUPLICATED")
    for symbol in symbols:
        path = market.resolve_symbol_path(symbol)
        if path is None:
            raise ContractError("MATERIALIZATION_HELD_MARKET_SOURCE_MISSING")
        market._assert_path_has_no_symlink(
            path, boundary=market.data_root, label="held Market source", require_exists=True
        )
        digest, _ = regular_file_sha256(path, label="held Market source")
        market._read_strict_catalog_parquet(
            path, table_meta={"sha256": digest}, logical_table="corporate adjustment"
        )
        refs[symbol] = {"path": str(path.relative_to(workspace)), "sha256": digest}
    if (
        portfolio_state is None
        and native._pointer_sha(root / "_record_store/current.v1.json") != expected_store
    ):
        raise ContractError("MATERIALIZATION_HOLDINGS_POINTER_CHANGED")
    return refs


def verify_materialized_adjustment_refs(*, workspace, snapshot_ref: dict, refs: dict) -> None:
    """Replay held prices through the same frozen native reader used at capture."""
    from quant_investor.market.market_data_reader import MarketDataReader

    workspace = Path(workspace)
    validate_ref(snapshot_ref)
    market = MarketDataReader(
        market="CN",
        data_root=workspace / "data",
        mode_policy="strict",
        frozen_snapshot_ref={
            "path": Path(snapshot_ref["path"]).relative_to("data").as_posix(),
            "sha256": snapshot_ref["sha256"],
        },
    )
    for symbol, ref in refs.items():
        validate_ref(ref)
        selected = market.resolve_symbol_path(symbol)
        if selected is None or selected != workspace / ref["path"]:
            raise ContractError("MATERIALIZATION_HELD_MARKET_PATH_MISMATCH")
        market._assert_path_has_no_symlink(
            selected, boundary=market.data_root, label="held Market source", require_exists=True
        )
        market._read_strict_catalog_parquet(
            selected, table_meta={"sha256": ref["sha256"]}, logical_table="corporate adjustment"
        )


def store_materialization_arguments(workspace, handoff, recipe):
    workspace = Path(workspace)
    preimages = recipe["store_preimages"]
    market = handoff["market_pointer_ref"]
    calendar = handoff["calendar_ref"]
    policy = recipe["policy_refs"]["store"]
    retrospective = recipe["retrospective_ref"]
    return {
        "project_root": workspace,
        **(
            {"registered_event_declaration_ref": recipe["registered_event_declaration_ref"]}
            if recipe["schema_version"] == "cn-daily-execute-recipe.v6"
            else {}
        ),
        "record_root": workspace / RECORD_ROOT,
        "expected_store_pointer_sha": preimages["store_pointer_ref"]["sha256"],
        "expected_market_pointer_sha": market["sha256"],
        "expected_benchmark_pointer_sha": preimages["benchmark_pointer_ref"]["sha256"],
        "expected_event_pointer_sha": preimages["event_pointer_ref"]["sha256"],
        "calendar_receipt_path": workspace / calendar["path"],
        "calendar_receipt_sha": calendar["sha256"],
        "policy_path": policy["path"],
        "policy_sha": policy["sha256"],
        "retrospective_path": retrospective["path"] if retrospective is not None else None,
        "retrospective_sha": retrospective["sha256"] if retrospective is not None else None,
    }
