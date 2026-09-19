"""Date-to-request construction only; owning execution gates remain authoritative."""

from datetime import datetime, timezone
from copy import deepcopy
import re
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes
from .automatic_catchup_contract import completion_day, document_ref, validate_automatic_request
from .bootstrap import validate_bootstrap_declaration
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref
from .daily_journal import FALSE_AUTHORITY, _validate_day
from .dashboard_serving_contract import POLICY, head_bytes, head_js, digest
from .execution_recipe import validate_execution_recipe, validate_catchup_template
from .prediction_policy import validate_prediction_policy
from .production_request import validate_production_request
from .research_timing import CURRENT, acquisition_deadline

CONFIG_SCHEMA = "cn-daily-preparation-config.v1"
COMMITMENT_SCHEMA = "cn-daily-preparation-commitment.v1"
COMMITMENT_SCHEMA_V2 = "cn-daily-preparation-commitment.v2"
CONSTRUCTION_SCHEMA_V2 = "cn-daily-preparation-construction.v2"
REGISTERED_CONSTRUCTION_FIELDS = {
    "schema_version",
    "registered_event_declaration_ref",
    "decision_baseline_pointer_ref",
    "writer_pointer_ref",
}
RESULT_SCHEMA = "cn-daily-preparation-result.v1"
ADMISSION = "DEFERRED_TO_NATIVE_STORE"
SCOPE = {
    "market": "CN",
    "strategy_id": "aggressive_tech_manufacturing",
    "graph_sha256": GRAPH_SHA256,
}
REF_FIELDS = {
    "release_ref",
    "release_install_ref",
    "factor_loop_context_ref",
    "research_policy_ref",
    "store_policy_ref",
    "timing_policy_ref",
    "theme_acquisition_ref",
    "corporate_action_template_ref",
    "industry_source_ref",
    "risk_free_ref",
    "exposure_rows_ref",
    "seed_completion_ref",
}
NULLABLE = {"exposure_rows_ref", "seed_completion_ref"}
FIELDS = {
    "schema_version",
    *SCOPE,
    *REF_FIELDS,
    "prediction_deadline_local_time",
    "publish_current_dashboard",
}
RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"
PREIMAGES = {
    "store_pointer_ref": RECORD_ROOT + "/_record_store/current.v1.json",
    "event_pointer_ref": RECORD_ROOT + "/_event_store/current.v1.json",
    "benchmark_pointer_ref": "data/parquet/cn/benchmarks/_latest.json",
}
BENCHMARK_CSV = "portfolio_dashboard/inputs/cn_index_benchmark.csv"


class PreparationError(ContractError):
    """Bounded input availability, never an execution outcome."""

    def __init__(self, code, **fields):
        super().__init__(code)
        self.code = code
        self.fields = fields


def validate_config(value):
    if type(value) is not dict or set(value) != FIELDS:
        raise PreparationError("PREPARATION_CONFIG_FIELDS_INVALID")
    if value["schema_version"] != CONFIG_SCHEMA or any(value[k] != v for k, v in SCOPE.items()):
        raise PreparationError("PREPARATION_CONFIG_SCOPE_INVALID")
    for key in REF_FIELDS:
        if key not in NULLABLE or value[key] is not None:
            validate_ref(value[key])
    if value["seed_completion_ref"] is not None:
        completion_day(value["seed_completion_ref"])
    deadline = value["prediction_deadline_local_time"]
    if deadline is not None and (
        type(deadline) is not str
        or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]", deadline) is None
    ):
        raise PreparationError("PREPARATION_DEADLINE_INVALID")
    if type(value["publish_current_dashboard"]) is not bool:
        raise PreparationError("PREPARATION_PUBLICATION_INVALID")
    return deepcopy(value)


def preparation_root(config_ref, day):
    validate_ref(config_ref)
    _validate_day(day)
    return f"results/operations/daily_production/CN/{day}/preparation/{config_ref['sha256']}"


def validate_locator(value, *, config, calendar):
    fields = {"mode", "completion_ref", "observed_head", "configured_seed_ref", "selected_seed_ref"}
    if type(value) is not dict or set(value) != fields:
        raise PreparationError("PREPARATION_LOCATOR_INVALID")
    if value["configured_seed_ref"] != config["seed_completion_ref"]:
        raise PreparationError("PREPARATION_SEED_BINDING_INVALID")
    head, seed = value["observed_head"], value["selected_seed_ref"]
    if head is not None:
        if type(head) is not dict or set(head) != {"document", "json_sha256", "mirror_sha256"}:
            raise PreparationError("PREPARATION_HEAD_INVALID")
        raw = canonical_json_bytes(head["document"])
        document = head_bytes(raw)
        if head["json_sha256"] != digest(raw) or head["mirror_sha256"] != digest(head_js(raw)):
            raise PreparationError("PREPARATION_HEAD_INVALID")
        expected = ("HEAD", document["completion_ref"], None)
    elif config["seed_completion_ref"] is not None:
        expected = ("SEED", config["seed_completion_ref"], config["seed_completion_ref"])
    else:
        expected = ("NONE", None, None)
    if (value["mode"], value["completion_ref"], seed) != expected:
        raise PreparationError("PREPARATION_LOCATOR_SELECTION_INVALID")
    if value["completion_ref"] is not None:
        day = completion_day(value["completion_ref"])
        if day not in calendar["ordered_open_dates"] or day > calendar["target_trade_date"]:
            raise PreparationError("PREPARATION_LOCATOR_DATE_INVALID")
    return value


def _construction(value, *, bootstrap):
    fields = {
        "store_preimages",
        "benchmark_ref",
        "previous_trade_date",
        "factor_parent_pointer_sha256",
    }
    registered = type(value) is dict and value.get("schema_version") == CONSTRUCTION_SCHEMA_V2
    if type(value) is not dict or set(value) != fields | (
        REGISTERED_CONSTRUCTION_FIELDS if registered else set()
    ):
        raise PreparationError("PREPARATION_CONSTRUCTION_INVALID")
    if registered:
        if bootstrap:
            raise PreparationError("REGISTERED_PREVIOUS_EOD_REQUIRED")
        for key in REGISTERED_CONSTRUCTION_FIELDS - {"schema_version"}:
            validate_ref(value[key])
        if (
            value["writer_pointer_ref"]["sha256"]
            != value["store_preimages"]["store_pointer_ref"]["sha256"]
        ):
            raise PreparationError("PREPARATION_REGISTERED_WRITER_MISMATCH")
    refs = value["store_preimages"]
    if type(refs) is not dict or set(refs) != set(PREIMAGES):
        raise PreparationError("PREPARATION_PREIMAGES_INVALID")
    for key, path in PREIMAGES.items():
        if validate_ref(refs[key])["path"] != path:
            raise PreparationError("PREPARATION_PREIMAGE_PATH_INVALID")
    if validate_ref(value["benchmark_ref"])["path"] != BENCHMARK_CSV:
        raise PreparationError("PREPARATION_BENCHMARK_PATH_INVALID")
    if bootstrap:
        _validate_day(value["previous_trade_date"])
        validate_ref({"path": "parent.json", "sha256": value["factor_parent_pointer_sha256"]})
    elif (
        value["previous_trade_date"] is not None
        or value["factor_parent_pointer_sha256"] is not None
    ):
        raise PreparationError("PREPARATION_AUTOMATIC_PARENT_FORBIDDEN")


def _prediction(config, day):
    local = datetime.strptime(day + config["prediction_deadline_local_time"], "%Y%m%d%H:%M:%S")
    value = {
        "schema_version": "cn-daily-prediction-policy.v1",
        "trade_date": day,
        "graph_sha256": GRAPH_SHA256,
        "session_rule": "CN_SSE_SZSE_OPEN",
        "prediction_deadline": local.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return validate_prediction_policy(value, trade_date=day)


def assemble_objects(
    *, config, config_ref, calendar, calendar_ref, raw_calendar_ref, locator, construction
):
    """Pure deterministic reconstruction; no reads, time sampling or publication."""
    config = validate_config(config)
    validate_locator(locator, config=config, calendar=calendar)
    day = calendar["target_trade_date"]
    root = preparation_root(config_ref, day)
    objects = []

    def add(leaf, document):
        path = root + ("/request.json" if leaf == "request.json" else "/objects/" + leaf)
        ref = document_ref(path, document)
        objects.append({"ref": ref, "document": deepcopy(document)})
        return ref

    bootstrap = locator["mode"] == "NONE"
    empty = not bootstrap and completion_day(locator["completion_ref"]) == day
    recipe = None
    if empty:
        if construction is not None:
            raise PreparationError("PREPARATION_EMPTY_CONSTRUCTION_FORBIDDEN")
    else:
        _construction(construction, bootstrap=bootstrap)
        prediction_ref = None
        if config["prediction_deadline_local_time"] is not None:
            prediction_ref = add("prediction-policy.json", _prediction(config, day))
        recipe = {
            "schema_version": "cn-daily-execute-recipe.v5",
            **SCOPE,
            "target_trade_date": day,
            **{
                k: config[k]
                for k in (
                    "release_ref",
                    "release_install_ref",
                    "factor_loop_context_ref",
                    "theme_acquisition_ref",
                    "corporate_action_template_ref",
                    "publish_current_dashboard",
                )
            },
            "previous_completion_ref": None,
            "bootstrap_ref": None,
            "policy_refs": {
                "research": config["research_policy_ref"],
                "store": config["store_policy_ref"],
                "prospective": prediction_ref,
            },
            "store_preimages": deepcopy(construction["store_preimages"]),
            "retrospective_ref": None,
            "dashboard_sources": {
                "benchmark_ref": construction["benchmark_ref"],
                "risk_free_ref": config["risk_free_ref"],
            },
            "research_sources": {
                "as_of": None,
                "industry_source_ref": config["industry_source_ref"],
                "exposure_rows_ref": config["exposure_rows_ref"],
                "theme_source_ref": None,
                "fundamental": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
                "macro": {"mode": "MAINTENANCE_STAGE", "source_ref": None},
            },
            "dashboard_publication_policy": POLICY,
            "research_timing": {
                "mode": CURRENT,
                "policy_ref": config["timing_policy_ref"],
                "acquisition_deadline": acquisition_deadline(day),
            },
        }
        if construction.get("schema_version") == CONSTRUCTION_SCHEMA_V2:
            recipe.update(
                schema_version="cn-daily-execute-recipe.v6",
                registered_event_declaration_ref=construction["registered_event_declaration_ref"],
            )
        if bootstrap:
            previous = construction["previous_trade_date"]
            days = calendar["ordered_open_dates"]
            if day not in days or days.index(day) == 0 or days[days.index(day) - 1] != previous:
                raise PreparationError("PREPARATION_BOOTSTRAP_PREDECESSOR_INVALID")
            declaration = {
                "schema_version": "cn-daily-bootstrap.v1",
                **SCOPE,
                "first_trade_date": day,
                "previous_trade_date": previous,
                "factor_parent_pointer_sha256": construction["factor_parent_pointer_sha256"],
                "store_preimages": construction["store_preimages"],
                "authority": dict(FALSE_AUTHORITY),
            }
            validate_bootstrap_declaration(declaration, recipe=recipe)
            recipe["bootstrap_ref"] = add("bootstrap.json", declaration)
        else:
            recipe["store_preimages"]["store_pointer_ref"] = None
        recipe_ref = add("recipe.json", recipe)
    if bootstrap:
        request = {
            "schema_version": "cn-daily-production-request.v1",
            **SCOPE,
            "action": "EXECUTE",
            "target_trade_date": day,
            "release_install_ref": config["release_install_ref"],
            "recipe_ref": recipe_ref,
            "maintenance_handoff_ref": None,
            "calendar_ref": None,
            "raw_calendar_ref": None,
            "previous_completion_ref": None,
            "day_input_refs": {},
        }
        validate_production_request(request, release_install_ref=config["release_install_ref"])
        validate_execution_recipe(recipe, request=request)
    else:
        collection_ref = add(
            "collection.json",
            {
                "schema_version": "cn-daily-catchup-recipes.v3",
                "publication_policy": "CURRENT_OBSERVED_CLOSE_ONLY",
                "recipes": {} if empty else {day: recipe},
            },
        )
        request = {
            "schema_version": "cn-daily-automatic-request.v2",
            **SCOPE,
            "action": "CATCH_UP",
            "release_install_ref": config["release_install_ref"],
            "calendar_ref": calendar_ref,
            "raw_calendar_ref": raw_calendar_ref,
            "seed_completion_ref": locator["selected_seed_ref"],
            "recipe_ref": collection_ref,
            "day_input_refs": {},
        }
        validate_automatic_request(request, release_install_ref=config["release_install_ref"])
        if recipe is not None:
            validate_catchup_template(
                recipe, request={**request, "action": "EXECUTE", "target_trade_date": day}
            )
    add("request.json", request)
    return objects
