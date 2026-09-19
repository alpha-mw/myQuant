"""Exact automatic daily-close contracts; none grants financial execution authority."""

from pathlib import PurePosixPath
import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref
from .daily_journal import _false_authority, _validate_day
from .production_result import validate_production_result, production_result_exit_code

REQUEST_SCHEMA = "cn-daily-automatic-request.v1"
REQUEST_SCHEMA_V2 = "cn-daily-automatic-request.v2"
RESOLUTION_SCHEMA = "cn-daily-catchup-resolution.v1"
RESULT_SCHEMA = "cn-daily-automatic-result.v1"
PREFIX = "results/operations/daily_production/CN/automatic/aggressive_tech_manufacturing"
REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "market",
        "strategy_id",
        "graph_sha256",
        "action",
        "release_install_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "seed_completion_ref",
        "recipe_ref",
        "day_input_refs",
    }
)
RESOLUTION_FIELDS = frozenset(
    {
        "schema_version",
        "auto_request_ref",
        "market",
        "strategy_id",
        "graph_sha256",
        "release_install_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "observed_head",
        "locator_ref",
        "anchor_ref",
        "adopted_completion_refs",
        "target_trade_date",
        "ordered_trade_dates",
        "day_scopes",
        "derived_collection",
        "derived_collection_ref",
        "derived_request",
        "derived_request_ref",
        "resolved_at",
        "authority",
    }
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "auto_request_ref",
        "target_trade_date",
        "anchor_ref",
        "adopted_completion_refs",
        "ordered_trade_dates",
        "day_scopes",
        "missing_input_dates",
        "resolution_ref",
        "status",
        "result",
        "authority",
    }
)


class AutomaticCatchupError(ContractError):
    """Expected automatic-run availability with bounded public recovery details."""

    def __init__(
        self, code, *, pending_request_ref=None, missing_input_dates=None, completed_eod_refs=None
    ):
        super().__init__(code)
        self.code = code
        self.fields = {}
        if pending_request_ref is not None:
            self.fields["pending_request_ref"] = validate_ref(pending_request_ref)
        if missing_input_dates is not None:
            for day in missing_input_dates:
                _validate_day(day)
            self.fields["missing_input_dates"] = list(missing_input_dates)
        if completed_eod_refs is not None:
            if code != "AUTO_PUBLICATION_EXPIRED" or type(completed_eod_refs) is not list:
                raise ContractError("AUTO_COMPLETED_EOD_REFS_INVALID")
            days = []
            for row in completed_eod_refs:
                if type(row) is not dict or set(row) != {"trade_date", "completion_ref"}:
                    raise ContractError("AUTO_COMPLETED_EOD_REFS_INVALID")
                _validate_day(row["trade_date"])
                if completion_day(row["completion_ref"]) != row["trade_date"]:
                    raise ContractError("AUTO_COMPLETED_EOD_REFS_INVALID")
                days.append(row["trade_date"])
            if days != sorted(set(days)):
                raise ContractError("AUTO_COMPLETED_EOD_REFS_INVALID")
            self.fields["completed_eod_refs"] = [
                dict(trade_date=row["trade_date"], completion_ref=dict(row["completion_ref"]))
                for row in completed_eod_refs
            ]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_path(request_ref, leaf):
    validate_ref(request_ref)
    if leaf not in {"resolution.v1.json", "collection.json", "request.json", "closure.v1.json"}:
        raise ContractError("AUTO_RUN_PATH_INVALID")
    return f"{PREFIX}/runs/{digest(canonical_json_bytes(request_ref))}/{leaf}"


def document_ref(path, document):
    return {"path": path, "sha256": digest(canonical_json_bytes(document))}


def validate_automatic_request(value, *, release_install_ref):
    if type(value) is not dict or set(value) != REQUEST_FIELDS:
        raise ContractError("AUTO_REQUEST_FIELDS_INVALID")
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"] not in {REQUEST_SCHEMA, REQUEST_SCHEMA_V2}
        or value["market"] != "CN"
        or value["strategy_id"] != "aggressive_tech_manufacturing"
        or value["graph_sha256"] != GRAPH_SHA256
        or type(value["action"]) is not str
        or value["action"] not in {"PLAN", "CATCH_UP"}
    ):
        raise ContractError("AUTO_REQUEST_SCOPE_INVALID")
    for field in ("release_install_ref", "calendar_ref", "raw_calendar_ref", "recipe_ref"):
        validate_ref(value[field])
    if value["release_install_ref"] != validate_ref(release_install_ref):
        raise ContractError("AUTO_REQUEST_INSTALL_MISMATCH")
    seed = value["seed_completion_ref"]
    if seed is not None:
        completion_day(seed)
    if type(value["day_input_refs"]) is not dict:
        raise ContractError("AUTO_REQUEST_INPUTS_INVALID")
    for day, ref in value["day_input_refs"].items():
        _validate_day(day)
        validate_ref(ref)
    return parse_canonical_json_bytes(canonical_json_bytes(value))


def completion_day(ref):
    validate_ref(ref)
    day = PurePosixPath(ref["path"]).parent.name
    _validate_day(day)
    if ref["path"] != f"results/operations/daily_production/CN/{day}/completion.v1.json":
        raise ContractError("AUTO_COMPLETION_PATH_INVALID")
    return day


def validate_automatic_result(value, *, request_ref, request, resolution):
    if (
        type(value) is not dict
        or set(value) != RESULT_FIELDS
        or value["schema_version"] != RESULT_SCHEMA
        or value["auto_request_ref"] != request_ref
        or value["action"] != request["action"]
        or not _false_authority(value["authority"])
    ):
        raise ContractError("AUTO_RESULT_FIELDS_INVALID")
    for field in (
        "target_trade_date",
        "anchor_ref",
        "adopted_completion_refs",
        "ordered_trade_dates",
        "day_scopes",
    ):
        if value[field] != resolution[field]:
            raise ContractError("AUTO_RESULT_RESOLUTION_MISMATCH")
    if request["action"] == "PLAN":
        anchor = completion_day(resolution["anchor_ref"])
        expected_missing = [
            day
            for day in resolution["ordered_trade_dates"]
            if day > anchor
            and day not in resolution["derived_collection"]["recipes"]
            and day not in resolution["derived_request"]["day_input_refs"]
        ]
        if (
            value["result"] is not None
            or value["resolution_ref"] is not None
            or type(value["missing_input_dates"]) is not list
            or value["missing_input_dates"] != expected_missing
            or value["status"] != ("BLOCKED" if value["missing_input_dates"] else "PLANNED")
        ):
            raise ContractError("AUTO_PLAN_RESULT_INVALID")
    else:
        if value["resolution_ref"] != document_ref(
            run_path(request_ref, "resolution.v1.json"), resolution
        ):
            raise ContractError("AUTO_RESULT_RESOLUTION_REF_INVALID")
        target = resolution["target_trade_date"]
        anchor = completion_day(resolution["anchor_ref"])
        dates = [day for day in resolution["ordered_trade_dates"] if day > anchor] or [target]
        result = validate_production_result(
            value["result"], action="CATCH_UP", target_trade_date=target, expected_dates=dates
        )
        if value["missing_input_dates"] != [] or value["status"] != result["execution_state"]:
            raise ContractError("AUTO_RESULT_STATE_INVALID")
    return value


def automatic_result_exit_code(value):
    """Known validated automatic results only; legacy callers retain their own path."""
    if value.get("schema_version") != RESULT_SCHEMA:
        raise ContractError("AUTO_RESULT_SCHEMA_INVALID")
    if value["action"] == "PLAN":
        if value["status"] not in {"PLANNED", "BLOCKED"}:
            raise ContractError("AUTO_RESULT_STATE_INVALID")
        return 2 if value["status"] == "BLOCKED" else 0
    if value["action"] != "CATCH_UP" or value["status"] != value["result"]["execution_state"]:
        raise ContractError("AUTO_RESULT_STATE_INVALID")
    return production_result_exit_code(value["result"])
