"""Close only expired automatic work proven unstarted under native exclusion."""

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import _verify_file
from quant_investor.market.close_session_authority import CloseSessionAuthorityError
from quant_investor.market.tushare_transport import TushareHttpsError
from .automatic_catchup_contract import completion_day, document_ref, run_path, digest
from .automatic_catchup_resolution import _replay
from .catchup_binding import derive_catchup_binding
from .daily_contract import ContractError, utc_stamp
from .daily_journal import FALSE_AUTHORITY
from .journal_storage import JournalStorage
from .maintenance_readback import existing_maintenance_lock, _walk, _read

RUN_ROOT = "data/private/cn_daily_maintenance"
CLOSURE_SCHEMA = "cn-daily-catchup-closure.v1"
CLOSURE_FIELDS = {
    "schema_version",
    "resolution_ref",
    "state",
    "completed_prefix_refs",
    "unresolved_trade_dates",
    "checked_at",
    "authority",
}


def completion_state(context, *, synthetic):
    source, resolution = context["sources"], context["resolution"]
    anchor = resolution["anchor_ref"]
    days = [day for day in resolution["ordered_trade_dates"] if day > completion_day(anchor)]
    completed, unresolved = [], []
    storage = JournalStorage(source.workspace)
    for day in days:
        path = f"results/operations/daily_production/CN/{day}/completion.v1.json"
        stored = storage.read(path)
        if stored is None:
            unresolved.append(day)
            continue
        if unresolved:
            raise ContractError("AUTO_COMPLETION_BEYOND_GAP")
        ref = {"path": path, "sha256": stored.byte_sha256}
        _replay(source, ref, context["request"], synthetic, anchor)
        completed.append(ref)
        anchor = ref
    source.recheck()
    return completed, unresolved, anchor


def prime_day_locks(storage, resolution):
    """Execution setup only: create coordinator locks before activating the lease."""
    storage.require_lock()
    journal = JournalStorage(storage.workspace)
    anchor = completion_day(resolution["anchor_ref"])
    for day in resolution["ordered_trade_dates"]:
        if day <= anchor:
            continue
        try:
            with journal.lock(
                f"results/operations/daily_production/CN/{day}/.lock", nonblocking=True
            ):
                pass
        except BlockingIOError as exc:
            raise ContractError("AUTO_DAY_BUSY") from exc


@contextmanager
def _existing_day(storage, day):
    journal = JournalStorage(storage.workspace)
    path = f"results/operations/daily_production/CN/{day}/.lock"
    parent, leaf, _ = journal._parent(path, create=False)
    fd = None
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        metadata = os.fstat(fd)
        _verify_file(metadata)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("AUTO_DAY_BUSY") from exc
        named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ContractError("AUTO_DAY_LOCK_CHANGED")
        yield journal, parent
        current, _, _ = journal._parent(path, create=False)
        try:
            a, b = os.fstat(parent), os.fstat(current)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ContractError("AUTO_DAY_ROOT_CHANGED")
        finally:
            os.close(current)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _directory_names(io, root_fd, parts):
    try:
        fd = _walk(io, root_fd, tuple(parts))
    except FileNotFoundError:
        return []
    try:
        return os.listdir(fd)
    finally:
        os.close(fd)


def _native_absence(io, root_fd, workspace, unresolved):
    tasks = _directory_names(io, root_fd, ("logical_tasks",))
    bound_attempts = set()
    native_root = Path(workspace) / RUN_ROOT
    for key in tasks:
        if any(key.startswith(day + "-") for day in unresolved):
            raise ContractError("AUTO_EXPIRY_NATIVE_CLAIM_PRESENT")
        # Inspect ownership, not latest time, to detect any orphan native attempt.
        names = _directory_names(io, root_fd, ("logical_tasks", key))
        if set(names) - {"claim.json", "attempt-1.json", "attempt-2.json"}:
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
        if "claim.json" not in names:
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
        claim_path = PurePosixPath("logical_tasks") / key / "claim.json"
        claim_raw = _read(io, root_fd, claim_path)
        claim = json.loads(claim_raw)
        if (
            type(claim) is not dict
            or claim.get("schema_version") != "cn-daily-logical-task.v1"
            or claim.get("logical_key") != key
            or claim.get("attempt_budget") != 2
        ):
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
        claim_ref = {"path": str(native_root / claim_path), "sha256": digest(claim_raw)}
        predecessor_started = None
        for name in ("attempt-1.json", "attempt-2.json"):
            if name not in names:
                continue
            row = json.loads(_read(io, root_fd, PurePosixPath("logical_tasks") / key / name))
            if (
                type(row) is not dict
                or set(row)
                != {"logical_key", "attempt_root", "predecessor_started_ref", "claim_ref"}
                or row.get("logical_key") != key
                or row["claim_ref"] != claim_ref
                or row["predecessor_started_ref"] != predecessor_started
                or (name == "attempt-2.json" and "attempt-1.json" not in names)
                or type(row.get("attempt_root")) is not str
            ):
                raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
            attempt = Path(row["attempt_root"])
            if attempt.parent != native_root / "attempts" or attempt.name in bound_attempts:
                raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
            # Missing or unsafe start records cannot establish a bound old attempt.
            start_raw = _read(
                io, root_fd, PurePosixPath("attempts") / attempt.name / "started.json"
            )
            started = json.loads(start_raw)
            if type(started) is not dict or started.get("state") != "STARTED":
                raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
            try:
                targets = _attempt_targets(io, root_fd, attempt.name, key)
            except (CloseSessionAuthorityError, TushareHttpsError) as exc:
                raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED") from exc
            if targets & set(unresolved):
                raise ContractError("AUTO_EXPIRY_NATIVE_TARGET_PRESENT")
            bound_attempts.add(attempt.name)
            predecessor_started = {
                "path": str(attempt / "started.json"),
                "sha256": digest(start_raw),
            }
    attempts = set(_directory_names(io, root_fd, ("attempts",)))
    if attempts != bound_attempts:
        raise ContractError("AUTO_EXPIRY_UNBOUND_NATIVE_ATTEMPT")


def _attempt_targets(io, root_fd, name, logical_key):
    """Ordinary weekend maintenance can target Friday despite a Saturday claim."""
    from quant_investor.market.historical_session import (
        FILENAME,
        CALENDAR_FILENAME,
        RAW_FILENAME,
        verify_historical_session,
    )
    from quant_investor.market.close_session_authority import replay_close_session_authority

    base = PurePosixPath("attempts") / name
    calendar_raw = _read(io, root_fd, base / CALENDAR_FILENAME)
    raw = _read(io, root_fd, base / RAW_FILENAME)
    try:
        proof_raw = _read(io, root_fd, base / FILENAME)
    except FileNotFoundError:
        proof_raw = None
    if proof_raw is not None:
        proof = verify_historical_session(
            proof_bytes=proof_raw, calendar_bytes=calendar_raw, raw=raw
        )
        target = proof["requested_trade_date"]
        if logical_key[:8] != target:
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
    else:
        calendar = replay_close_session_authority(json.loads(calendar_raw), raw).receipt
        if calendar["observed_local_time"][:10].replace("-", "") != logical_key[:8]:
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
        target = calendar["target_trade_date"]
    targets = {target}
    for leaf in ("attempt.json", "core-completion.json"):
        try:
            record = json.loads(_read(io, root_fd, base / leaf))
        except FileNotFoundError:
            continue
        if type(record) is not dict:
            raise ContractError("AUTO_EXPIRY_NATIVE_OWNERSHIP_UNCONFIRMED")
        if record.get("target_date") is not None:
            from .daily_journal import _validate_day

            _validate_day(record["target_date"])
            targets.add(record["target_date"])
    return targets


def _day_absence(context, day, journal, fd, previous):
    names = set(os.listdir(fd))
    if names - {".lock", "catchup", "executions", "preparation"} or (
        "executions" in names and "catchup" not in names
    ):
        raise ContractError("AUTO_EXPIRY_DAY_START_OR_UNKNOWN_EVIDENCE")
    prepared_recheck = None
    if "preparation" in names:
        from .source_preparation_absence import prepared_only

        prepared_recheck = prepared_only(
            context, day, journal, lambda parts: _directory_names(journal._io, fd, parts)
        )
    if "catchup" not in names:
        return prepared_recheck
    resolution, workspace = context["resolution"], context["sources"].workspace
    root_ref = resolution["derived_request_ref"]
    path = f"results/operations/daily_production/CN/{day}/catchup"
    # Only this resolution's deterministic preparatory binding files are allowed.
    directory, _, _ = journal._parent(path + "/unused", create=False)
    try:
        if os.listdir(directory) != [root_ref["sha256"]]:
            raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
    finally:
        os.close(directory)

    derived = derive_catchup_binding(
        workspace=workspace, request_ref=root_ref, day=day, previous_completion_ref=previous
    )
    origin_recheck = None
    if "executions" in names:
        origin_recheck = _origin_only_execution(context, day, journal, fd, derived)
    version = {"cn-daily-catchup-binding.v2": 2, "cn-daily-catchup-binding.v3": 3}.get(
        derived["binding"]["schema_version"]
    )
    if version is None:
        raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
    expected = {
        "recipe.json": derived["recipe_raw"],
        "request.json": derived["request_raw"],
        f"binding.v{version}.json": canonical_json_bytes(derived["binding"]),
    }
    prefix = path + "/" + root_ref["sha256"]
    directory, _, _ = journal._parent(prefix + "/unused", create=False)
    try:
        files = os.listdir(directory)
        if set(files) - set(expected):
            raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
        for name in files:
            stored = journal.read(prefix + "/" + name)
            if stored is None or stored.data != expected[name]:
                raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
    finally:
        os.close(directory)

    def recheck():
        if prepared_recheck is not None:
            prepared_recheck()
        if origin_recheck is not None:
            origin_recheck()

    return recheck


def _origin_only_execution(context, day, journal, fd, derived):
    """Only the exact pre-maintenance origin may accompany an unstarted v3 binding."""
    from .automatic_origin import origin_path, read_automatic_origin, recheck_origin

    if (
        derived["recipe"]["schema_version"] != "cn-daily-execute-recipe.v6"
        or derived["binding"].get("maintenance_mode") != "CURRENT"
    ):
        raise ContractError("AUTO_EXPIRY_DAY_START_OR_UNKNOWN_EVIDENCE")
    request_ref = derived["binding"]["execution_request_ref"]
    prefix = ("executions", request_ref["sha256"])
    for parts, expected in (
        (("executions",), [request_ref["sha256"]]),
        (prefix, ["inputs"]),
        ((*prefix, "inputs"), ["automatic-origin.v1.json"]),
    ):
        if _directory_names(journal._io, fd, parts) != expected:
            raise ContractError("AUTO_EXPIRY_DAY_START_OR_UNKNOWN_EVIDENCE")
    stored = journal.read(origin_path(day, request_ref))
    if stored is None:
        raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
    origin = read_automatic_origin(
        workspace=context["sources"].workspace,
        reference={"path": stored.relative_path, "sha256": stored.byte_sha256},
    )
    if (
        origin["resolution"]["resolution"] != context["resolution"]
        or origin["bound"]["binding"] != derived["binding"]
    ):
        raise ContractError("AUTO_EXPIRY_DAY_OWNERSHIP_UNCONFIRMED")
    recheck_origin(origin)
    return lambda: recheck_origin(origin)


def expire_unstarted(storage, context, *, resolution_ref, synthetic, now=None):
    """Persist terminal closure only under all absence-proof locks; never rewrite a run."""
    storage.require_lock()
    resolution = context["resolution"]
    if resolution_ref != document_ref(
        run_path(resolution["auto_request_ref"], "resolution.v1.json"), resolution
    ):
        raise ContractError("AUTO_CLOSURE_RESOLUTION_MISMATCH")
    actual = now or datetime.now(timezone.utc)
    today = actual.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    if not any(
        scope["maintenance_mode"] == "CURRENT" and day < today
        for day, scope in resolution["day_scopes"].items()
    ):
        raise ContractError("AUTO_PENDING_REQUEST_CONFLICT")
    completed, unresolved, previous = completion_state(context, synthetic=synthetic)
    if not unresolved:
        raise ContractError("AUTO_EXPIRY_REQUIRES_UNFINISHED_WORK")
    if any(day >= today for day in unresolved):
        raise ContractError("AUTO_PENDING_REQUEST_CONFLICT")
    path = run_path(resolution["auto_request_ref"], "closure.v1.json")
    with (
        existing_maintenance_lock(workspace=storage.workspace, run_root=RUN_ROOT) as (io, root_fd),
        ExitStack() as stack,
    ):
        locks = {day: stack.enter_context(_existing_day(storage, day)) for day in unresolved}
        _native_absence(io, root_fd, storage.workspace, unresolved)
        rechecks = [
            _day_absence(context, day, journal, fd, previous)
            for day, (journal, fd) in locks.items()
        ]
        for recheck in rechecks:
            if recheck is not None:
                recheck()
        context["sources"].recheck()
        stored = storage.read(path)
        actual = now or datetime.now(timezone.utc)
        stamp = actual.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if stored is not None:
            old = parse_canonical_json_bytes(stored.data)
            if type(old) is not dict or set(old) != CLOSURE_FIELDS:
                raise ContractError("AUTO_CLOSURE_INVALID")
            stamp = old["checked_at"]
        if not utc_stamp(resolution["resolved_at"]) <= utc_stamp(stamp) <= actual:
            raise ContractError("AUTO_CLOSURE_TIME_INVALID")
        checked_day = utc_stamp(stamp).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        if any(day >= checked_day for day in unresolved):
            raise ContractError("AUTO_CLOSURE_BEFORE_EXPIRY")
        value = {
            "schema_version": CLOSURE_SCHEMA,
            "resolution_ref": resolution_ref,
            "state": "EXPIRED_UNSTARTED",
            "completed_prefix_refs": completed,
            "unresolved_trade_dates": unresolved,
            "checked_at": stamp,
            "authority": dict(FALSE_AUTHORITY),
        }
        if stored is not None and stored.data != canonical_json_bytes(value):
            raise ContractError("AUTO_CLOSURE_CONFLICT")
        storage.write(path, canonical_json_bytes(value))
        pending = storage.pending()
        if pending is None or pending["resolution_ref"] != resolution_ref:
            raise ContractError("AUTO_PENDING_RESOLUTION_CONFLICT")
        storage.set_pending({**pending, "state": "IDLE"})
    return document_ref(path, value)
