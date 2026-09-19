"""Prepare immutable launcher inputs from native local sources; never run producers."""

from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.factors.production_authority import (
    FactorProductionStore,
    FACTOR_ACTIVE_POINTER_PATH,
    FACTOR_PRODUCTION_MARKER_PATH,
    FACTOR_AUTHORITY_ACTIVE,
)
from quant_investor.market import cn_benchmark_store as benchmark
from quant_investor.market.close_session_authority import replay_close_session_authority
from quant_investor.market.requested_session import classify_requested_session
from quant_investor.strategy_records import event_store, performance, store
from quant_investor.strategy_records.event_contracts import SYMBOLIC_RECEIPT
from quant_investor.strategy_records.event_receipts import resolve_catalog_event_receipt
from .automatic_catchup_contract import document_ref, completion_day
from .automatic_catchup_resolution import optional_bytes
from .automatic_catchup_storage import AutomaticRunStorage
from .catchup_binding import BindingSources
from ._source_bytes import SourceBytesChanged, read_source_bytes
from .daily_contract import utc_stamp, validate_ref
from .daily_journal import FALSE_AUTHORITY
from .daily_preparation_contract import (
    ADMISSION,
    BENCHMARK_CSV,
    COMMITMENT_SCHEMA,
    COMMITMENT_SCHEMA_V2,
    CONSTRUCTION_SCHEMA_V2,
    RESULT_SCHEMA,
    PREIMAGES,
    RECORD_ROOT,
    REF_FIELDS,
    PreparationError,
    assemble_objects,
    preparation_root,
    validate_config,
    validate_locator,
)
from .dashboard_serving_contract import PREFIX, HEAD_JSON, HEAD_JS, head_bytes, head_js, digest
from .execution_controls import verify_recipe_static_controls
from .journal_storage import JournalStorage
from .research_corporate_inputs import validate_corporate_template
from .theme_acquisition import validate_theme_acquisition_policy, validate_theme_policy_profile


class Sources(BindingSources):
    """Remember byte identities and absences until fresh registration ends."""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.identities = {}
        self.absent = set()

    def raw(self, ref):
        validate_ref(ref)
        stored = self._read(ref["path"])
        raw = stored.data
        if stored.byte_sha256 != ref["sha256"]:
            raise PreparationError("PREPARATION_SOURCE_SHA_MISMATCH")
        if self.observed.setdefault(ref["path"], raw) != raw:
            raise PreparationError("PREPARATION_SOURCE_CHANGED")
        old = self.identities.setdefault(ref["path"], stored.stat_identity)
        if old != stored.stat_identity:
            raise PreparationError("PREPARATION_SOURCE_CHANGED")
        return raw

    def _read(self, path):
        try:
            return read_source_bytes(
                self.storage,
                path,
                allowed_modes=frozenset({0o400, 0o600, 0o644}),
                maximum_bytes=8 * 1024 * 1024,
                require_owner=True,
                require_single_link=True,
                require_non_executable=True,
            )
        except SourceBytesChanged as exc:
            raise PreparationError("PREPARATION_SOURCE_CHANGED") from exc

    def pin(self, path):
        stored = self._read(path)
        ref = {"path": path, "sha256": stored.byte_sha256}
        if self.raw(ref) != stored.data:
            raise PreparationError("PREPARATION_SOURCE_CHANGED")
        return ref

    def optional(self, path):
        raw = optional_bytes(self, path)
        if raw is None:
            self.absent.add(path)
            return None
        if path in self.absent or self.raw(self.pin(path)) != raw:
            raise PreparationError("PREPARATION_SOURCE_CHANGED")
        return raw

    def recheck(self):
        for path, identity in self.identities.items():
            stored = self._read(path)
            if stored.stat_identity != identity or stored.data != self.observed[path]:
                raise PreparationError("PREPARATION_SOURCE_CHANGED")
        if any(optional_bytes(self, path) is not None for path in self.absent):
            raise PreparationError("PREPARATION_SOURCE_CHANGED")


def _walk_refs(source, value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            source.raw(value)
        else:
            for child in value.values():
                _walk_refs(source, child)
    elif type(value) is list:
        for child in value:
            _walk_refs(source, child)


def _locator(source, config, calendar):
    raw = source.optional(f"{PREFIX}/{HEAD_JSON}")
    mirror = source.optional(f"{PREFIX}/{HEAD_JS}")
    observed = None
    if raw is not None or mirror is not None:
        if raw is None or mirror is None or head_js(raw) != mirror:
            raise PreparationError("PREPARATION_HEAD_PAIR_INVALID")
        observed = {
            "document": head_bytes(raw),
            "json_sha256": digest(raw),
            "mirror_sha256": digest(mirror),
        }
    seed = config["seed_completion_ref"]
    selected = None if observed is not None else seed
    value = {
        "mode": "HEAD" if observed is not None else "SEED" if seed is not None else "NONE",
        "completion_ref": observed["document"]["completion_ref"] if observed is not None else seed,
        "observed_head": observed,
        "configured_seed_ref": seed,
        "selected_seed_ref": selected,
    }
    validate_locator(value, config=config, calendar=calendar)
    if value["completion_ref"] is not None:
        source.raw(value["completion_ref"])
    return value


def _native_sources(source, calendar):
    from scripts.registered_daily_event_sources import select_daily_source

    root = Path(source.workspace)
    refs = {key: source.pin(path) for key, path in PREIMAGES.items()}
    target = calendar["target_trade_date"]
    registered = select_daily_source(
        workspace=source.workspace,
        store_pointer_ref=refs["store_pointer_ref"],
        trade_date=f"{target[:4]}-{target[4:6]}-{target[6:]}",
    )
    if registered["state"] == "BLOCKED_FINALIZED_V2_INPUT_CUSTODY_MISSING":
        raise PreparationError(registered["state"])
    registered_day = target if registered["state"] == "REGISTERED_INTRADAY" else None
    for reference in registered.get("source_refs", []):
        source.raw(reference)
    loaded = store.load_registered_catalog(root / RECORD_ROOT)
    if loaded is None or loaded[1]["schema_id"] != store.CATALOG_SCHEMA_V3:
        raise PreparationError("PREPARATION_STORE_V3_REQUIRED")
    pointer, catalog = loaded
    source.raw(
        {"path": RECORD_ROOT + "/" + pointer["catalog_path"], "sha256": pointer["catalog_sha256"]}
    )
    history_ref = catalog["performance_history_ref"]
    for key in ("manifest", "series", "owner_declaration"):
        ref = history_ref[key]
        source.raw({"path": RECORD_ROOT + "/" + ref["path"], "sha256": ref["sha256"]})
    history = performance.load_performance_history(root / RECORD_ROOT, history_ref)
    frontier = str(history["rows"][-1]["valuation_date"]).replace("-", "")
    days, target = calendar["ordered_open_dates"], calendar["target_trade_date"]
    if frontier not in days or frontier > target:
        raise PreparationError("PREPARATION_STORE_CALENDAR_COVERAGE_INVALID")
    required = [day for day in days if frontier < day <= target]
    required = sorted(set(required + [target]))
    events = event_store.load_generation(root / RECORD_ROOT / "_event_store")
    indices = benchmark.load_generation(root / "data/parquet/cn/benchmarks")
    if (
        events["pointer_sha256"] != refs["event_pointer_ref"]["sha256"]
        or indices["pointer_sha256"] != refs["benchmark_pointer_ref"]["sha256"]
    ):
        raise PreparationError("PREPARATION_NATIVE_POINTER_CHANGED")
    event_ref = events["pointer"]["generation"]
    source.raw(
        {"path": RECORD_ROOT + "/_event_store/" + event_ref["path"], "sha256": event_ref["sha256"]}
    )
    for key in ("manifest", "series"):
        ref = indices["pointer"][key]
        source.raw({"path": "data/parquet/cn/benchmarks/" + ref["path"], "sha256": ref["sha256"]})
    acquisition = indices["manifest"].get("acquisition_receipt_ref")
    if acquisition is not None:
        source.raw(acquisition)
    by_date = {row["trade_date"].replace("-", ""): row for row in events["closures"]}
    keys = {(row["date"].strftime("%Y%m%d"), row["ts_code"]) for row in indices["rows"]}
    missing_events = [day for day in required if day not in by_date and day != registered_day]
    missing_benchmark = [
        day for day in required if any((day, code) not in keys for code in benchmark.REQUIRED_CODES)
    ]
    if missing_events or missing_benchmark:
        raise PreparationError(
            "PREPARATION_NATIVE_DATES_MISSING",
            missing_event_dates=missing_events,
            missing_benchmark_dates=missing_benchmark,
        )
    for day in required:
        if day == registered_day:
            continue
        closure = by_date[day]
        for key in ("policy_ref", "owner_declaration_ref"):
            source.raw(closure[key])
        receipt = closure["source_receipt_ref"]
        if receipt is not None:
            if SYMBOLIC_RECEIPT.fullmatch(receipt["path"]):
                resolve_catalog_event_receipt(workspace=root, closure=closure)
            else:
                source.raw(receipt)
                from quant_investor.strategy_records.daily_event_source import (
                    validate_daily_closure_source,
                )

                proof = validate_daily_closure_source(workspace=root, closure=closure)
                if proof is not None:
                    for reference in proof["source_refs"]:
                        source.raw(reference)
    compatibility = source.pin(BENCHMARK_CSV)
    if source.raw(compatibility) != benchmark.compatibility_csv_bytes(indices["rows"]):
        raise PreparationError("PREPARATION_BENCHMARK_COMPATIBILITY_CHANGED")
    value = {
        "store_preimages": refs,
        "benchmark_ref": compatibility,
        "previous_trade_date": None,
        "factor_parent_pointer_sha256": None,
    }
    if registered_day is not None:
        value.update(
            schema_version=CONSTRUCTION_SCHEMA_V2,
            **{
                k: registered[k]
                for k in (
                    "registered_event_declaration_ref",
                    "decision_baseline_pointer_ref",
                    "writer_pointer_ref",
                )
            },
        )
    return value


def _registered_predecessor(source, construction, locator, calendar, *, prepared_at=None):
    if construction is None or construction.get("schema_version") != CONSTRUCTION_SCHEMA_V2:
        return
    from scripts.registered_daily_event_sources import read_declaration
    from scripts.daily_completion_store import replay_completed_store

    previous = locator["completion_ref"]
    target = calendar["target_trade_date"]
    days = calendar["ordered_open_dates"]
    if previous is None or target not in days or days.index(target) == 0:
        raise PreparationError("REGISTERED_PREVIOUS_EOD_REQUIRED")
    day = days[days.index(target) - 1]
    if completion_day(previous) != day:
        raise PreparationError("REGISTERED_PREVIOUS_EOD_DATE_MISMATCH")
    proof = read_declaration(
        workspace=source.workspace, declaration_ref=construction["registered_event_declaration_ref"]
    )
    declared = proof["declaration"]
    if prepared_at is not None and utc_stamp(prepared_at) < utc_stamp(declared["registered_at"]):
        raise PreparationError("PREPARATION_BEFORE_REGISTERED_DECLARATION")
    if (
        declared["baseline_store_pointer_ref"] != construction["decision_baseline_pointer_ref"]
        or declared["writer_store_pointer_ref"] != construction["writer_pointer_ref"]
        or declared["trade_date"].replace("-", "") != target
        or proof["baseline"]["record"]["data_date"].replace("-", "") != day
    ):
        raise PreparationError("PREPARATION_REGISTERED_BINDING_INVALID")
    completed = replay_completed_store(
        workspace=source.workspace, trade_date=day, completion_ref=previous
    )
    if completed["output_refs"]["pointer"] != declared["baseline_store_pointer_ref"]:
        raise PreparationError("REGISTERED_PREVIOUS_EOD_BASELINE_MISMATCH")
    source.raw(previous)
    for reference in proof["source_refs"]:
        source.raw(reference)
    source.recheck()


def _bootstrap_parent(source, calendar, construction):
    days, day = calendar["ordered_open_dates"], calendar["target_trade_date"]
    if day not in days or days.index(day) == 0:
        raise PreparationError("PREPARATION_PREVIOUS_OPEN_REQUIRED")
    previous = days[days.index(day) - 1]
    for date in (previous, day):
        if (
            source.optional(f"results/operations/daily_production/CN/{date}/completion.v1.json")
            is not None
        ):
            raise PreparationError("PREPARATION_EXISTING_EOD_REQUIRES_SEED")
    factors = FactorProductionStore(source.workspace)
    pointer = factors.read(FACTOR_ACTIVE_POINTER_PATH)
    marker = factors.read(FACTOR_PRODUCTION_MARKER_PATH)
    verified = factors.verify_active()
    if (
        pointer is None
        or marker is None
        or (
            verified.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE
            or verified.get("as_of") != previous
            or verified.get("factor_pointer_byte_sha256") != pointer.byte_sha256
        )
    ):
        raise PreparationError("PREPARATION_FACTOR_PARENT_INVALID")

    def recheck():
        if (
            factors.read(FACTOR_ACTIVE_POINTER_PATH) != pointer
            or factors.read(FACTOR_PRODUCTION_MARKER_PATH) != marker
        ):
            raise PreparationError("PREPARATION_FACTOR_PARENT_CHANGED")

    recheck()
    construction.update(
        previous_trade_date=previous, factor_parent_pointer_sha256=pointer.byte_sha256
    )
    return recheck


def _static(source, config, objects):
    for key in REF_FIELDS - {"seed_completion_ref"}:
        if config[key] is not None:
            source.raw(config[key])
    for key in ("industry_source_ref", "corporate_action_template_ref"):
        _walk_refs(source, source.document(config[key]))
    validate_corporate_template(source.document(config["corporate_action_template_ref"]))
    policy = validate_theme_acquisition_policy(source.document(config["theme_acquisition_ref"]))
    generated = {item["ref"]["path"]: item for item in objects}

    def document(ref):
        item = generated.get(ref["path"])
        if item is None:
            return source.document(ref)
        if item["ref"] != ref:
            raise PreparationError("PREPARATION_GENERATED_REF_CONFLICT")
        return item["document"]

    recipe = next(
        item["document"] for item in objects if item["ref"]["path"].endswith("/recipe.json")
    )
    validate_theme_policy_profile(recipe, policy)
    verify_recipe_static_controls(workspace=source.workspace, recipe=recipe, document=document)


def _pending(lock, request_ref=None):
    pending = lock.pending()
    if (
        pending is not None
        and pending["state"] == "ACTIVE"
        and pending["auto_request_ref"] != request_ref
    ):
        raise PreparationError(
            "PREPARATION_PENDING_REQUEST", pending_request_ref=pending["auto_request_ref"]
        )


def _validate_commitment(value, *, config, config_ref, calendar, calendar_ref, raw_calendar_ref):
    fields = {
        "schema_version",
        "trade_date",
        "config_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "prepared_at",
        "locator",
        "construction",
        "objects",
        "store_policy_admission",
        "authority",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or (
            value["schema_version"] not in {COMMITMENT_SCHEMA, COMMITMENT_SCHEMA_V2}
            or value["config_ref"] != config_ref
            or value["calendar_ref"] != calendar_ref
            or value["raw_calendar_ref"] != raw_calendar_ref
            or value["trade_date"] != calendar["target_trade_date"]
            or value["store_policy_admission"] != ADMISSION
            or value["authority"] != FALSE_AUTHORITY
        )
    ):
        raise PreparationError("PREPARATION_COMMITMENT_CONFLICT")
    registered = (
        type(value["construction"]) is dict
        and value["construction"].get("schema_version") == CONSTRUCTION_SCHEMA_V2
    )
    if (value["schema_version"] == COMMITMENT_SCHEMA_V2) != registered:
        raise PreparationError("PREPARATION_COMMITMENT_VERSION_MISMATCH")
    observed = datetime.strptime(calendar["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    prepared = utc_stamp(value["prepared_at"])
    if (
        prepared < observed
        or prepared.astimezone(ZoneInfo("Asia/Shanghai")).date() != observed.date()
    ):
        raise PreparationError("PREPARATION_COMMITMENT_TIME_INVALID")
    objects = assemble_objects(
        config=config,
        config_ref=config_ref,
        calendar=calendar,
        calendar_ref=calendar_ref,
        raw_calendar_ref=raw_calendar_ref,
        locator=value["locator"],
        construction=value["construction"],
    )
    if canonical_json_bytes(objects) != canonical_json_bytes(value["objects"]):
        raise PreparationError("PREPARATION_COMMITMENT_OBJECTS_INVALID")
    return objects


def _existing_commitment(storage, root):
    values = [(v, storage.read(root + f"/commitment.v{v}.json")) for v in (1, 2)]
    existing = [(v, stored) for v, stored in values if stored is not None]
    if len(existing) > 1:
        raise PreparationError("PREPARATION_COMMITMENT_VERSION_CONFLICT")
    if not existing:
        return None, None
    version, stored = existing[0]
    value = parse_canonical_json_bytes(stored.data)
    if value.get("schema_version") != (COMMITMENT_SCHEMA if version == 1 else COMMITMENT_SCHEMA_V2):
        raise PreparationError("PREPARATION_COMMITMENT_PATH_VERSION_MISMATCH")
    return root + f"/commitment.v{version}.json", stored


def _register(storage, path, commitment, objects, source, lock, recheck, source_binding=None):
    # Detect every conflict before repairing any object.
    for item in objects:
        stored = storage.read(item["ref"]["path"])
        if stored is not None and stored.data != canonical_json_bytes(item["document"]):
            raise PreparationError("PREPARATION_OBJECT_CONFLICT")
    source.recheck()
    recheck()
    lock.require_lock()
    raw = canonical_json_bytes(commitment)
    storage.write(path, raw)
    if source_binding is not None:
        from .source_slot_storage import publish_prepared_locator

        expected = source_binding["expected_preimages"]
        if (
            expected is not None
            and commitment["construction"] is not None
            and commitment["construction"]["store_preimages"] != expected
        ):
            raise PreparationError("SOURCE_PREPARATION_PREIMAGES_CHANGED")
        publish_prepared_locator(
            workspace=source.workspace,
            config_ref=commitment["config_ref"],
            prior_sha256=source_binding["locator_sha256"],
            commitment_ref=document_ref(path, commitment),
            commitment=commitment,
            request_ref=objects[-1]["ref"],
            lock=lock,
        )
    for item in objects:
        lock.require_lock()
        storage.write(item["ref"]["path"], canonical_json_bytes(item["document"]))
    if storage.read(path).data != raw:
        raise PreparationError("PREPARATION_COMMITMENT_CHANGED")
    for item in objects:
        if storage.read(item["ref"]["path"]).data != canonical_json_bytes(item["document"]):
            raise PreparationError("PREPARATION_OBJECT_CHANGED")
    source.recheck()
    recheck()
    lock.require_lock()


def prepare_daily_request(
    *, workspace, config_ref, calendar_ref, raw_calendar_ref, now=None, _source_binding=None
):
    """Return registered input refs; all native execution admission is deferred."""
    source = Sources(str(Path(workspace).resolve(strict=True)))
    config = validate_config(source.document(config_ref))
    calendar = replay_close_session_authority(
        json.loads(source.raw(calendar_ref)), source.raw(raw_calendar_ref)
    ).receipt
    observed = datetime.strptime(calendar["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z")
    day = observed.strftime("%Y%m%d")
    root = preparation_root(config_ref, day)
    storage = JournalStorage(source.workspace)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "REGISTERED_INPUTS_ONLY",
        "trade_date": day,
        "config_ref": config_ref,
        "calendar_ref": calendar_ref,
        "raw_calendar_ref": raw_calendar_ref,
        "commitment_ref": None,
        "request_ref": None,
        "store_policy_admission": ADMISSION,
        "authority": dict(FALSE_AUTHORITY),
    }
    path, existing = _existing_commitment(storage, root)
    if existing is None:
        actual = now or datetime.now(timezone.utc)
        if (
            actual.tzinfo is None
            or actual.utcoffset() is None
            or actual < observed
            or actual.astimezone(ZoneInfo("Asia/Shanghai")).date() != observed.date()
        ):
            raise PreparationError("PREPARATION_FRESH_CALENDAR_REQUIRED")
        session = classify_requested_session(
            requested_trade_date=day, receipt=calendar, raw=source.raw(raw_calendar_ref)
        )
        if session["classification"] == "CONFIRMED_CLOSED":
            source.recheck()
            return {**result, "status": "NON_TRADING_DAY"}
    existed_before_lock = existing is not None
    with AutomaticRunStorage(source.workspace).locked() as lock:
        path, existing = _existing_commitment(storage, root)
        if existing is None and existed_before_lock:
            raise PreparationError("PREPARATION_COMMITMENT_DISAPPEARED")

        def no_parent_recheck():
            return None

        recheck = no_parent_recheck

        if existing is not None:
            commitment = parse_canonical_json_bytes(existing.data)
            objects = _validate_commitment(
                commitment,
                config=config,
                config_ref=config_ref,
                calendar=calendar,
                calendar_ref=calendar_ref,
                raw_calendar_ref=raw_calendar_ref,
            )
            _pending(lock, objects[-1]["ref"])
            _registered_predecessor(
                source,
                commitment["construction"],
                commitment["locator"],
                calendar,
                prepared_at=commitment["prepared_at"],
            )
        else:
            _pending(lock)
            locator = _locator(source, config, calendar)
            empty = (
                locator["completion_ref"] is not None
                and completion_day(locator["completion_ref"]) == day
            )
            construction = None if empty else _native_sources(source, calendar)
            _registered_predecessor(
                source,
                construction,
                locator,
                calendar,
                prepared_at=actual.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            if locator["mode"] == "NONE":
                recheck = _bootstrap_parent(source, calendar, construction)
            objects = assemble_objects(
                config=config,
                config_ref=config_ref,
                calendar=calendar,
                calendar_ref=calendar_ref,
                raw_calendar_ref=raw_calendar_ref,
                locator=locator,
                construction=construction,
            )
            if not empty:
                _static(source, config, objects)
            commitment = {
                "schema_version": (
                    COMMITMENT_SCHEMA_V2
                    if construction is not None
                    and construction.get("schema_version") == CONSTRUCTION_SCHEMA_V2
                    else COMMITMENT_SCHEMA
                ),
                "trade_date": day,
                "config_ref": config_ref,
                "calendar_ref": calendar_ref,
                "raw_calendar_ref": raw_calendar_ref,
                "prepared_at": actual.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "locator": locator,
                "construction": construction,
                "objects": objects,
                "store_policy_admission": ADMISSION,
                "authority": dict(FALSE_AUTHORITY),
            }
            _validate_commitment(
                commitment,
                config=config,
                config_ref=config_ref,
                calendar=calendar,
                calendar_ref=calendar_ref,
                raw_calendar_ref=raw_calendar_ref,
            )
            version = 2 if commitment["schema_version"] == COMMITMENT_SCHEMA_V2 else 1
            path = root + f"/commitment.v{version}.json"
        _register(storage, path, commitment, objects, source, lock, recheck, _source_binding)
        result.update(commitment_ref=document_ref(path, commitment), request_ref=objects[-1]["ref"])
    return result
