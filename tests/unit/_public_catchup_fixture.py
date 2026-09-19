"""Synthetic byte fixtures for derivation tests; native EOD/install replay is separate."""

from copy import deepcopy
from pathlib import PurePosixPath
import hashlib

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, DailyJournal
from quant_investor.operations.catchup_binding import (
    COLLECTION_SCHEMA,
    derive_catchup_binding,
    persist_catchup_binding,
    STORE_CURRENT,
)
from test_daily_evidence_production_request import recipe, request
from test_daily_evidence_catchup import fixture as calendar_fixture


def put(root, path, value):
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.parent
    while q != root:
        q.chmod(0o700)
        q = q.parent
    p.write_bytes(raw)
    p.chmod(0o600)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def completion(root, day, scope):
    pointer = put(root, f"store/{day}/committed-pointer.v1.json", {"synthetic_store_day": day})
    key = "a" * 64
    terminal = put(
        root,
        f"results/operations/daily_production/CN/{day}/nodes/store/"
        f"{key}/attempt-0001/terminal.json",
        {
            "schema_version": "cn-daily-node-terminal.v1",
            "state": "SUCCEEDED",
            "request_key": key,
            "output_refs": {"pointer": pointer},
            "authority": FALSE_AUTHORITY,
        },
    )
    return put(
        root,
        f"results/operations/daily_production/CN/{day}/completion.v1.json",
        {
            "schema_version": "cn-daily-eod-completion.v2",
            "status": "SUCCEEDED",
            "trade_date": day,
            **{k: scope[k] for k in ("market", "strategy_id", "graph_sha256")},
            "authority": FALSE_AUTHORITY,
            "node_terminal_refs": {"store": terminal},
        },
    )


def template(value, day):
    value = deepcopy(value)
    value["target_trade_date"] = day
    value["research_sources"]["as_of"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T13:30:00Z"
    value["previous_completion_ref"] = value["bootstrap_ref"] = None
    value["store_preimages"]["store_pointer_ref"] = None
    return value


def collection(root, *, target="20260828", days=("20260827", "20260828")):
    calendar = calendar_fixture(root)
    leaf = put(root, "input.json", {"synthetic_fixture": True})
    value = recipe()

    def replace_refs(obj):
        if type(obj) is dict:
            if set(obj) == {"path", "sha256"}:
                return dict(leaf)
            return {k: replace_refs(v) for k, v in obj.items()}
        return obj

    value = replace_refs(value)
    parent = completion(root, "20260826", value)
    declaration = put(
        root,
        "collection.json",
        {
            "schema_version": COLLECTION_SCHEMA,
            "recipes": {day: template(value, day) for day in days},
        },
    )
    root_request = request("CATCH_UP")
    root_request.update(
        target_trade_date=target,
        release_install_ref=leaf,
        recipe_ref=declaration,
        calendar_ref=calendar["calendar_ref"],
        raw_calendar_ref=calendar["raw_calendar_ref"],
        previous_completion_ref=parent,
        day_input_refs={},
    )
    return put(root, "catchup.json", root_request), root_request


def routed_collection(root, *, policy="CURRENT_OBSERVED_CLOSE_ONLY", now=None):
    """V2 routing fixture: real synthetic Calendar, controlled financial refs."""
    from datetime import datetime, timezone
    from quant_investor.operations.catchup_binding import COLLECTION_SCHEMA_V2
    from quant_investor.operations.dashboard_serving_contract import POLICY
    import json

    _, request_value = collection(root)
    calendar = calendar_fixture(root, now=now or datetime(2026, 8, 28, 13, tzinfo=timezone.utc))
    declaration = json.loads((root / request_value["recipe_ref"]["path"]).read_bytes())
    declaration.update(schema_version=COLLECTION_SCHEMA_V2, publication_policy=policy)
    for value in declaration["recipes"].values():
        value["research_sources"]["theme_source_ref"] = dict(request_value["release_install_ref"])
        value.update(
            schema_version="cn-daily-execute-recipe.v4",
            theme_acquisition_ref=None,
            corporate_action_context_ref=dict(request_value["release_install_ref"]),
            dashboard_publication_policy=POLICY,
            publish_current_dashboard=True,
        )
    request_value.update(
        recipe_ref=put(root, "collection.json", declaration),
        calendar_ref=calendar["calendar_ref"],
        raw_calendar_ref=calendar["raw_calendar_ref"],
    )
    return put(root, "catchup.json", request_value), request_value


def bind_existing_recipe(root, *, recipe_value, calendar_ref, raw_calendar_ref, routed=False):
    day = recipe_value["target_trade_date"]
    previous = PurePosixPath(recipe_value["previous_completion_ref"]["path"]).parent.name
    parent = completion(root, previous, recipe_value)
    declaration = {
        "schema_version": COLLECTION_SCHEMA,
        "recipes": {day: template(recipe_value, day)},
    }
    if routed:
        from quant_investor.operations.catchup_binding import COLLECTION_SCHEMA_V2

        declaration.update(
            schema_version=COLLECTION_SCHEMA_V2, publication_policy="HISTORICAL_ONLY"
        )
    collection_ref = put(root, "catchup-recipes.json", declaration)
    root_request = request("CATCH_UP")
    root_request.update(
        target_trade_date=day,
        release_install_ref=recipe_value["release_install_ref"],
        recipe_ref=collection_ref,
        calendar_ref=calendar_ref,
        raw_calendar_ref=raw_calendar_ref,
        previous_completion_ref=parent,
        day_input_refs={},
    )
    root_ref = put(root, "catchup-root.json", root_request)
    derived = derive_catchup_binding(
        workspace=str(root), request_ref=root_ref, day=day, previous_completion_ref=parent
    )
    journal = DailyJournal(str(root), day)
    with journal.locked():
        binding = persist_catchup_binding(journal=journal, derived=derived)
    put(
        root,
        STORE_CURRENT,
        (root / derived["binding"]["previous_store_pointer_ref"]["path"]).read_bytes(),
    )
    return derived["binding"]["execution_request_ref"], binding, derived
