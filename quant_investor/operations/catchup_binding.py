"""Deterministic historical request derivation, with no native execution authority.

Readers prove immutable derivation only. The batch coordinator separately replays
each actual predecessor through the full native EOD reader before maintenance.
No mutable Store head is consulted when reconstructing an existing binding.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.close_session_authority import replay_close_session_authority
from quant_investor.market.historical_session import prepare_historical_maintenance_input
from .catchup import plan_catchup
from ._source_bytes import SourceBytesChanged, read_source_bytes
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import FALSE_AUTHORITY, _false_authority, _validate_day
from .execution_recipe import validate_catchup_template, validate_execution_recipe
from .production_request import (
    SCHEMA as ROOT_SCHEMA,
    HISTORICAL_SCHEMA as REQUEST_SCHEMA,
    validate_production_request,
)

SCHEMA = "cn-daily-catchup-binding.v1"
COLLECTION_SCHEMA = "cn-daily-catchup-recipes.v1"
SCHEMA_V2 = "cn-daily-catchup-binding.v2"
COLLECTION_SCHEMA_V2 = "cn-daily-catchup-recipes.v2"
SCHEMA_V3 = "cn-daily-catchup-binding.v3"
COLLECTION_SCHEMA_V3 = "cn-daily-catchup-recipes.v3"
PUBLICATION_POLICIES = {"HISTORICAL_ONLY", "CURRENT_OBSERVED_CLOSE_ONLY"}
ROUTING_FIELDS = {"maintenance_mode", "dashboard_mode", "publication_policy"}
STORE_CURRENT = (
    "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
)
FIELDS = frozenset(
    {
        "schema_version",
        "market",
        "strategy_id",
        "trade_date",
        "graph_sha256",
        "release_install_ref",
        "root_request_ref",
        "collection_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "previous_completion_ref",
        "previous_store_terminal_ref",
        "previous_store_pointer_ref",
        "recipe_ref",
        "execution_request_ref",
        "authority",
    }
)

_DASHBOARD_INPUT_PATHS = {
    "benchmark_ref": "portfolio_dashboard/inputs/cn_index_benchmark.csv",
    "risk_free_ref": "portfolio_dashboard/inputs/cn_govt_bond_yield.csv",
}
_DASHBOARD_MODES = frozenset({0o400, 0o600, 0o644})
_DASHBOARD_MAXIMUM_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class _DashboardObservation:
    profile: str
    role: str
    path: str
    expected_sha256: str
    data: bytes
    stat_identity: tuple[int, ...]
    maximum_bytes: int


class BindingSources:
    """Stable checked bytes, retained across derivation and publication."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.storage = SecureSystemStorage(workspace)
        self.observed = {}
        self._dashboard_observed = {}

    def raw(self, ref):
        ref = validate_ref(ref)
        value = self.storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if value.byte_sha256 != ref["sha256"]:
            raise ContractError("CATCHUP_BINDING_SOURCE_SHA_MISMATCH")
        old = self.observed.get(ref["path"])
        if old is not None and old != value.data:
            raise ContractError("CATCHUP_BINDING_SOURCE_CONFLICT")
        self.observed[ref["path"]] = value.data
        return value.data

    def document(self, ref):
        return parse_canonical_json_bytes(self.raw(ref))

    def _dashboard_bytes(self, role, ref):
        ref = validate_ref(ref)
        if type(role) is not str or _DASHBOARD_INPUT_PATHS.get(role) != ref["path"]:
            raise ContractError("CATCHUP_DASHBOARD_INPUT_ROLE_PATH_INVALID")
        try:
            return read_source_bytes(
                self.storage,
                ref["path"],
                allowed_modes=_DASHBOARD_MODES,
                maximum_bytes=_DASHBOARD_MAXIMUM_BYTES,
                require_owner=True,
                require_single_link=True,
                require_non_executable=True,
            )
        except SourceBytesChanged as exc:
            raise ContractError("CATCHUP_BINDING_SOURCE_CHANGED") from exc

    def dashboard_input(self, role, ref):
        """Only the two exact Dashboard inputs gain the checked-in input mode."""
        value = self._dashboard_bytes(role, ref)
        if value.byte_sha256 != ref["sha256"]:
            raise ContractError("CATCHUP_BINDING_SOURCE_SHA_MISMATCH")
        observed = _DashboardObservation(
            "DASHBOARD_INPUT",
            role,
            ref["path"],
            ref["sha256"],
            value.data,
            value.stat_identity,
            _DASHBOARD_MAXIMUM_BYTES,
        )
        if self._dashboard_observed.setdefault(ref["path"], observed) != observed:
            raise ContractError("CATCHUP_BINDING_SOURCE_CHANGED")
        return value.data

    def recheck(self):
        for path, raw in self.observed.items():
            current = self.storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024)
            if current.data != raw:
                raise ContractError("CATCHUP_BINDING_SOURCE_CHANGED")
        for path, observed in self._dashboard_observed.items():
            if (
                observed.profile != "DASHBOARD_INPUT"
                or observed.path != path
                or observed.maximum_bytes != _DASHBOARD_MAXIMUM_BYTES
                or _DASHBOARD_INPUT_PATHS.get(observed.role) != path
            ):
                raise ContractError("CATCHUP_BINDING_SOURCE_CHANGED")
            current = self._dashboard_bytes(
                observed.role, {"path": path, "sha256": observed.expected_sha256}
            )
            if current.data != observed.data or current.stat_identity != observed.stat_identity:
                raise ContractError("CATCHUP_BINDING_SOURCE_CHANGED")


def binding_path(day, root_request_ref, version="v1"):
    _validate_day(day)
    validate_ref(root_request_ref)
    if version not in {"v1", "v2", "v3"}:
        raise ContractError("CATCHUP_BINDING_VERSION_INVALID")
    return str(
        PurePosixPath("results/operations/daily_production/CN")
        / day
        / "catchup"
        / root_request_ref["sha256"]
        / ("binding." + version + ".json")
    )


def read_catchup_collection(*, workspace, request_ref, source=None):
    source = source or BindingSources(workspace)
    request = source.document(request_ref)
    validate_production_request(request, release_install_ref=request["release_install_ref"])
    if request["schema_version"] != ROOT_SCHEMA or request["action"] != "CATCH_UP":
        raise ContractError("CATCHUP_ROOT_REQUEST_REQUIRED")
    collection = source.document(request["recipe_ref"])
    version = collection_version(collection)
    fields = {"schema_version", "recipes"} | (
        {"publication_policy"} if version in {"v2", "v3"} else set()
    )
    if set(collection) != fields or type(collection["recipes"]) is not dict:
        raise ContractError("CATCHUP_COLLECTION_FIELDS_INVALID")
    if version in {"v2", "v3"} and (
        type(collection["publication_policy"]) is not str
        or collection["publication_policy"] not in PUBLICATION_POLICIES
    ):
        raise ContractError("CATCHUP_PUBLICATION_POLICY_INVALID")
    previous = PurePosixPath(request["previous_completion_ref"]["path"]).parent.name
    plan = plan_catchup(
        workspace=workspace,
        calendar_ref=request["calendar_ref"],
        raw_calendar_ref=request["raw_calendar_ref"],
        previous_trade_date=previous,
        target_trade_date=request["target_trade_date"],
        day_input_refs=request["day_input_refs"],
    )
    recipes = collection["recipes"]
    if set(recipes) - set(plan["ordered_trade_dates"]) or set(recipes) & set(
        request["day_input_refs"]
    ):
        raise ContractError("CATCHUP_COLLECTION_DATE_OWNERSHIP_INVALID")
    source.raw(request["calendar_ref"])
    source.raw(request["raw_calendar_ref"])
    calendar = replay_close_session_authority(
        source.document(request["calendar_ref"]), source.raw(request["raw_calendar_ref"])
    ).receipt
    if utc_stamp(
        datetime.strptime(calendar["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    ) > datetime.now(timezone.utc):
        raise ContractError("CATCHUP_CALENDAR_OBSERVATION_IN_FUTURE")
    predecessor = previous
    routing = {}
    for day in plan["ordered_trade_dates"]:
        if version in {"v2", "v3"}:
            routing[day] = route_day(
                day=day,
                previous=predecessor,
                request=request,
                collection=collection,
                calendar=calendar,
                raw=source.raw(request["raw_calendar_ref"]),
            )
        if day in recipes:
            validate_catchup_template(
                recipes[day], request={**request, "action": "EXECUTE", "target_trade_date": day}
            )
            if version == "v2" and recipes[day]["schema_version"] != "cn-daily-execute-recipe.v4":
                raise ContractError("CATCHUP_V2_REQUIRES_SEALED_DASHBOARD_RECIPE")
            if version == "v3":
                from .research_timing import CURRENT, HISTORICAL

                expected_mode = (
                    CURRENT if routing[day]["maintenance_mode"] == "CURRENT" else HISTORICAL
                )
                if (
                    recipes[day]["schema_version"]
                    not in {"cn-daily-execute-recipe.v5", "cn-daily-execute-recipe.v6"}
                    or recipes[day]["research_timing"]["mode"] != expected_mode
                ):
                    raise ContractError("CATCHUP_V3_TIMING_PROFILE_REQUIRED")
            if version == "v1" or routing[day]["maintenance_mode"] == "HISTORICAL":
                prepared = prepare_historical_maintenance_input(
                    workspace=workspace,
                    value={
                        "target_trade_date": day,
                        "previous_trade_date": predecessor,
                        "calendar_ref": request["calendar_ref"],
                        "raw_calendar_ref": request["raw_calendar_ref"],
                    },
                    now=datetime.now(timezone.utc),
                )
                prepared["recheck"]()
        predecessor = day
    source.recheck()
    return {
        "request": request,
        "collection": collection,
        "plan": plan,
        "sources": source,
        "routing": routing,
        "calendar": calendar,
    }


def _check_strict_refs(source, value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            source.raw(value)
        else:
            for child in value.values():
                _check_strict_refs(source, child)
    elif type(value) is list:
        for child in value:
            _check_strict_refs(source, child)


def check_template_sources(source, template, *, profile="FRESH"):
    """Read a full validated template under a code-selected fresh/recovery policy."""
    from . import execution_recipe as recipes

    versions = {
        recipes.SCHEMA: recipes.FIELDS,
        recipes.SCHEMA_V2: recipes.FIELDS_V2,
        recipes.SCHEMA_V3: recipes.FIELDS_V3,
        recipes.SCHEMA_V4: recipes.FIELDS_V4,
        recipes.SCHEMA_V5: recipes.FIELDS_V5,
        recipes.SCHEMA_V6: recipes.FIELDS_V6,
    }
    if (
        type(template) is not dict
        or type(template.get("schema_version")) is not str
        or template["schema_version"] not in versions
        or set(template) != versions[template["schema_version"]]
    ):
        raise ContractError("CATCHUP_TEMPLATE_FIELDS_INVALID")
    if type(profile) is not str or profile not in {"FRESH", "FROZEN_REGISTERED_RECOVERY"}:
        raise ContractError("CATCHUP_TEMPLATE_READ_PROFILE_INVALID")
    frozen = profile == "FROZEN_REGISTERED_RECOVERY"
    if frozen and template["schema_version"] != recipes.SCHEMA_V6:
        raise ContractError("CATCHUP_TEMPLATE_READ_PROFILE_INVALID")
    modern = template["schema_version"] in {recipes.SCHEMA_V4, recipes.SCHEMA_V5, recipes.SCHEMA_V6}
    if not modern:
        _check_strict_refs(source, template)
        return
    dashboard = template["dashboard_sources"]
    if type(dashboard) is not dict or set(dashboard) != set(_DASHBOARD_INPUT_PATHS):
        raise ContractError("CATCHUP_TEMPLATE_FIELDS_INVALID")
    for key, value in template.items():
        if frozen and key == "store_preimages":
            continue
        if key != "dashboard_sources":
            _check_strict_refs(source, value)
            continue
        for role, ref in dashboard.items():
            validate_ref(ref)
            if frozen and role == "benchmark_ref":
                continue
            if _DASHBOARD_INPUT_PATHS[role] == ref["path"]:
                source.dashboard_input(role, ref)
            else:
                source.raw(ref)


def derive_catchup_binding(*, workspace, request_ref, day, previous_completion_ref, context=None):
    context = context or read_catchup_collection(workspace=workspace, request_ref=request_ref)
    source, root = context["sources"], context["request"]
    dates = context["plan"]["ordered_trade_dates"]
    if day not in context["collection"]["recipes"]:
        raise ContractError("CATCHUP_TEMPLATE_DATE_MISSING")
    index = dates.index(day)
    previous = dates[index - 1] if index else context["plan"]["previous_trade_date"]
    expected_path = f"results/operations/daily_production/CN/{previous}/completion.v1.json"
    if validate_ref(previous_completion_ref)["path"] != expected_path or (
        index == 0 and previous_completion_ref != root["previous_completion_ref"]
    ):
        raise ContractError("CATCHUP_BINDING_PREDECESSOR_INVALID")
    completion = source.document(previous_completion_ref)
    if (
        completion.get("schema_version")
        not in {"cn-daily-eod-completion.v1", "cn-daily-eod-completion.v2"}
        or completion.get("status") != "SUCCEEDED"
        or completion.get("trade_date") != previous
        or any(completion.get(k) != root[k] for k in ("market", "strategy_id", "graph_sha256"))
        or not _false_authority(completion.get("authority"))
    ):
        raise ContractError("CATCHUP_BINDING_PREVIOUS_EOD_INVALID")
    terminal_ref = validate_ref(completion["node_terminal_refs"]["store"])
    terminal = source.document(terminal_ref)
    terminal_path = PurePosixPath(terminal_ref["path"])
    if (
        terminal.get("schema_version") != "cn-daily-node-terminal.v1"
        or terminal.get("state") != "SUCCEEDED"
        or not _false_authority(terminal.get("authority"))
        or str(terminal_path.parent.parent.parent)
        != str(PurePosixPath(expected_path).parent / "nodes/store")
        or terminal_path.parent.parent.name != terminal.get("request_key")
        or terminal_path.name != "terminal.json"
    ):
        raise ContractError("CATCHUP_BINDING_STORE_TERMINAL_INVALID")
    pointer_ref = validate_ref(terminal["output_refs"]["pointer"])
    if not pointer_ref["path"].endswith("/committed-pointer.v1.json"):
        raise ContractError("CATCHUP_BINDING_IMMUTABLE_STORE_POINTER_REQUIRED")
    source.raw(pointer_ref)
    recipe = deepcopy(context["collection"]["recipes"][day])
    recipe["previous_completion_ref"] = dict(previous_completion_ref)
    recipe["store_preimages"]["store_pointer_ref"] = {
        "path": STORE_CURRENT,
        "sha256": pointer_ref["sha256"],
    }
    if recipe["schema_version"] == "cn-daily-execute-recipe.v6":
        from scripts.registered_daily_event_sources import read_declaration

        registered = read_declaration(
            workspace=workspace, declaration_ref=recipe["registered_event_declaration_ref"]
        )
        declaration = registered["declaration"]
        if (
            declaration["baseline_store_pointer_ref"] != pointer_ref
            or declaration["trade_date"].replace("-", "") != day
        ):
            raise ContractError("CATCHUP_REGISTERED_BASELINE_MISMATCH")
        recipe["store_preimages"]["store_pointer_ref"] = {
            "path": STORE_CURRENT,
            "sha256": declaration["writer_store_pointer_ref"]["sha256"],
        }
        for reference in registered["source_refs"]:
            source.raw(reference)
    version = collection_version(context["collection"])
    mode = None if version == "v1" else context["routing"][day]
    if mode is not None:
        recipe["publish_current_dashboard"] = mode["dashboard_mode"] == "CURRENT_LATEST_EOD"
    directory = PurePosixPath(binding_path(day, request_ref, version)).parent
    recipe_raw = canonical_json_bytes(recipe)
    recipe_ref = {
        "path": str(directory / "recipe.json"),
        "sha256": hashlib.sha256(recipe_raw).hexdigest(),
    }
    request = {
        **root,
        "schema_version": (
            ROOT_SCHEMA
            if mode is not None and mode["maintenance_mode"] == "CURRENT"
            else REQUEST_SCHEMA
        ),
        "action": "EXECUTE",
        "target_trade_date": day,
        "recipe_ref": recipe_ref,
        "calendar_ref": None,
        "raw_calendar_ref": None,
        "previous_completion_ref": None,
        "day_input_refs": {},
    }
    validate_production_request(request, release_install_ref=root["release_install_ref"])
    validate_execution_recipe(recipe, request=request)
    request_raw = canonical_json_bytes(request)
    execution_ref = {
        "path": str(directory / "request.json"),
        "sha256": hashlib.sha256(request_raw).hexdigest(),
    }
    binding = {
        "schema_version": {"v1": SCHEMA, "v2": SCHEMA_V2, "v3": SCHEMA_V3}[version],
        **{k: root[k] for k in ("market", "strategy_id", "graph_sha256", "release_install_ref")},
        "trade_date": day,
        "root_request_ref": dict(request_ref),
        "collection_ref": root["recipe_ref"],
        "calendar_ref": root["calendar_ref"],
        "raw_calendar_ref": root["raw_calendar_ref"],
        "previous_completion_ref": dict(previous_completion_ref),
        "previous_store_terminal_ref": terminal_ref,
        "previous_store_pointer_ref": pointer_ref,
        "recipe_ref": recipe_ref,
        "execution_request_ref": execution_ref,
        "authority": FALSE_AUTHORITY,
    }
    if mode is not None:
        binding.update(mode)
    source.recheck()
    return {
        "binding": binding,
        "recipe": recipe,
        "request": request,
        "recipe_raw": recipe_raw,
        "request_raw": request_raw,
        "sources": source,
    }


def read_catchup_binding(*, workspace, binding_ref):
    source = BindingSources(workspace)
    value = source.document(binding_ref)
    if (
        type(value) is not dict
        or value.get("schema_version") not in {SCHEMA, SCHEMA_V2, SCHEMA_V3}
        or set(value)
        != FIELDS
        | (ROUTING_FIELDS if value.get("schema_version") in {SCHEMA_V2, SCHEMA_V3} else set())
        or not _false_authority(value["authority"])
    ):
        raise ContractError("CATCHUP_BINDING_FIELDS_INVALID")
    version = {SCHEMA: "v1", SCHEMA_V2: "v2", SCHEMA_V3: "v3"}[value["schema_version"]]
    for key in FIELDS:
        if key.endswith("_ref"):
            validate_ref(value[key])
    if binding_ref["path"] != binding_path(value["trade_date"], value["root_request_ref"], version):
        raise ContractError("CATCHUP_BINDING_PATH_INVALID")
    context = read_catchup_collection(
        workspace=workspace, request_ref=value["root_request_ref"], source=source
    )
    derived = derive_catchup_binding(
        workspace=workspace,
        request_ref=value["root_request_ref"],
        day=value["trade_date"],
        previous_completion_ref=value["previous_completion_ref"],
        context=context,
    )
    if value != derived["binding"]:
        raise ContractError("CATCHUP_BINDING_DERIVATION_MISMATCH")
    if (
        source.raw(value["recipe_ref"]) != derived["recipe_raw"]
        or source.raw(value["execution_request_ref"]) != derived["request_raw"]
    ):
        raise ContractError("CATCHUP_BINDING_GENERATED_BYTES_MISMATCH")
    source.recheck()
    return derived


def persist_catchup_binding(*, journal, derived):
    """Short day lock only; no native callbacks and no orphan request authority."""
    journal._require_lock()
    value, source = derived["binding"], derived["sources"]
    if journal.trade_date != value["trade_date"]:
        raise ContractError("CATCHUP_BINDING_JOURNAL_MISMATCH")
    source.recheck()
    version = {SCHEMA: "v1", SCHEMA_V2: "v2", SCHEMA_V3: "v3"}[value["schema_version"]]
    path = binding_path(value["trade_date"], value["root_request_ref"], version)
    rows = [
        (value["recipe_ref"]["path"], derived["recipe_raw"]),
        (value["execution_request_ref"]["path"], derived["request_raw"]),
        (path, canonical_json_bytes(value)),
    ]
    for destination, raw in rows:
        old = journal.storage.read(destination)
        if old is not None and old.data != raw:
            raise ContractError("CATCHUP_BINDING_IMMUTABLE_CONFLICT")
    for destination, raw in rows:
        journal.storage.write(destination, raw)
    source.recheck()
    return {"path": path, "sha256": hashlib.sha256(rows[-1][1]).hexdigest()}


def historical_input(binding):
    return {
        "target_trade_date": binding["trade_date"],
        "previous_trade_date": PurePosixPath(
            binding["previous_completion_ref"]["path"]
        ).parent.name,
        "calendar_ref": binding["calendar_ref"],
        "raw_calendar_ref": binding["raw_calendar_ref"],
    }


def verify_bound_historical_core(*, derived, proof, calendar, raw):
    """Compare replayed Calendar semantics; attempt-local source paths may differ."""
    binding, source = derived["binding"], derived["sources"]
    original_raw = source.raw(binding["raw_calendar_ref"])
    original = replay_close_session_authority(
        source.document(binding["calendar_ref"]), original_raw
    ).receipt
    attempt = replay_close_session_authority(calendar, raw).receipt
    ignored = {"raw_response_path", "validated_at"}
    if (
        raw != original_raw
        or {k: v for k, v in original.items() if k not in ignored}
        != {k: v for k, v in attempt.items() if k not in ignored}
        or proof["requested_trade_date"] != binding["trade_date"]
        or proof["previous_trade_date"] != historical_input(binding)["previous_trade_date"]
        or utc_stamp(proof["observed_at"])
        != datetime.strptime(original["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    ):
        raise ContractError("CATCHUP_BINDING_HISTORICAL_CORE_MISMATCH")
    source.recheck()


def collection_version(value):
    if type(value) is not dict or value.get("schema_version") not in {
        COLLECTION_SCHEMA,
        COLLECTION_SCHEMA_V2,
        COLLECTION_SCHEMA_V3,
    }:
        raise ContractError("CATCHUP_COLLECTION_FIELDS_INVALID")
    return {COLLECTION_SCHEMA: "v1", COLLECTION_SCHEMA_V2: "v2", COLLECTION_SCHEMA_V3: "v3"}[
        value["schema_version"]
    ]


def route_day(*, day, previous, request, collection, calendar, raw):
    from quant_investor.market.requested_session import (
        classify_current_session_edge,
        classify_catchup_session,
    )

    observed = calendar["observed_local_time"][:10].replace("-", "")
    if day < observed:
        classify_catchup_session(
            requested_trade_date=day, previous_trade_date=previous, receipt=calendar, raw=raw
        )
        maintenance = "HISTORICAL"
    elif day == observed:
        classify_current_session_edge(
            requested_trade_date=day, previous_trade_date=previous, receipt=calendar, raw=raw
        )
        maintenance = "CURRENT"
    else:
        raise ContractError("CATCHUP_ROUTING_FUTURE_DATE")
    current = (
        maintenance == "CURRENT"
        and day == request["target_trade_date"] == calendar["target_trade_date"]
        and collection["publication_policy"] == "CURRENT_OBSERVED_CLOSE_ONLY"
    )
    return {
        "maintenance_mode": maintenance,
        "dashboard_mode": "CURRENT_LATEST_EOD" if current else "HISTORICAL_CAPTURE",
        "publication_policy": collection["publication_policy"],
    }


def validate_routed_native_input(value, *, day, previous, routing, version="v2"):
    from .native_input_contract import validate_native_input_shape
    from .dashboard_serving_contract import POLICY

    validate_native_input_shape(value)
    if version not in {"v2", "v3"}:
        raise ContractError("CATCHUP_NATIVE_INPUT_VERSION_INVALID")
    if (
        value["schema_version"]
        not in (
            {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}
            if version == "v3"
            else {"cn-daily-native-inputs.v5"}
        )
        or value["trade_date"] != day
        or value["previous_trade_date"] != previous
        or value["dashboard_publication_policy"] != POLICY
        or value["publish_current_dashboard"]
        is not (routing["dashboard_mode"] == "CURRENT_LATEST_EOD")
    ):
        raise ContractError("CATCHUP_NATIVE_INPUT_ROUTING_CONFLICT")


def verify_completed_edge(
    *, workspace, day, completion_ref, previous, previous_ref=None, inspected=None
):
    """Call after full native EOD replay; prove its own recorded Calendar edge."""
    from pathlib import Path
    from quant_investor.migration.canonical import parse_json_bytes
    from quant_investor.market.requested_session import require_calendar_predecessor
    from .completion_readback import inspect_recorded_completion
    from .native_input_contract import validate_native_input_shape

    source = BindingSources(workspace)
    inspected = inspected or inspect_recorded_completion(
        workspace=workspace, trade_date=day, completion_ref=completion_ref
    )
    recorded = inspected["recorded_completion"]
    inputs = source.document(recorded["native_inputs_ref"])
    validate_native_input_shape(inputs)
    if inputs["trade_date"] != day or inputs["previous_trade_date"] != previous:
        raise ContractError("CATCHUP_COMPLETED_PREDECESSOR_MISMATCH")
    calendar = parse_json_bytes(
        source.raw(inputs["calendar_ref"]), label="completed Calendar", require_canonical=False
    )
    snapshot = inspected.get("completed_handoff_snapshot")
    if snapshot is not None:
        handoff = snapshot.document("handoff")
        if handoff["calendar_ref"] != inputs["calendar_ref"]:
            raise ContractError("CATCHUP_COMPLETED_CALENDAR_BINDING_MISMATCH")
        raw_ref = handoff["raw_calendar_ref"]
        if (
            previous_ref is not None
            and snapshot.document("recipe")["previous_completion_ref"] != previous_ref
        ):
            raise ContractError("CATCHUP_COMPLETED_PREDECESSOR_REF_MISMATCH")
    else:
        path = Path(calendar["raw_response_path"])
        if path.is_absolute():
            path = path.relative_to(Path(workspace).resolve(strict=True))
        raw_ref = {"path": path.as_posix(), "sha256": calendar["raw_response_sha256"]}
    raw = source.raw(raw_ref)
    require_calendar_predecessor(
        requested_trade_date=day, previous_trade_date=previous, receipt=calendar, raw=raw
    )
    source.recheck()
    if snapshot is not None:
        snapshot.recheck()
    return inputs


def requires_current_serving(context, day):
    """Only the exact native-observed current target has current presentation scope."""
    if collection_version(context["collection"]) == "v1":
        return True
    calendar = context["calendar"]
    return context["collection"][
        "publication_policy"
    ] == "CURRENT_OBSERVED_CLOSE_ONLY" and day == context["request"][
        "target_trade_date"
    ] == calendar[
        "target_trade_date"
    ] == calendar[
        "observed_local_time"
    ][
        :10
    ].replace(
        "-", ""
    )
