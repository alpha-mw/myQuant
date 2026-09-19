"""Non-creating maintenance exclusion and descriptor-bound auxiliary receipt reads."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage, _verify_directory
from .daily_contract import ContractError, validate_ref
from .daily_journal import _validate_day

_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _walk(
    storage: SecureSystemStorage, parent: int, parts: tuple[str, ...], *, governed: bool = True
) -> int:
    fd = os.dup(parent)
    try:
        for part in parts:
            storage._reject_casefold_alias(fd, part)
            child = os.open(part, _DIR, dir_fd=fd)
            try:
                _verify_directory(os.fstat(child), governed=governed)
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _file_stat(fd):
    value = os.fstat(fd)
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise ContractError("MAINTENANCE_RECEIPT_UNSAFE")
    return value


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@contextmanager
def existing_maintenance_lock(*, workspace: str, run_root: str):
    """Yield the verified run-root fd; consumers must read through this descriptor."""
    validate_ref({"path": run_root, "sha256": "0" * 64})
    storage = SecureSystemStorage(workspace)
    workspace_fd = storage._open_workspace()
    root_fd = lock_fd = None
    try:
        root_fd = _walk(storage, workspace_fd, PurePosixPath(run_root).parts, governed=False)
        _verify_directory(os.fstat(root_fd), governed=True)
        try:
            lock_fd = os.open(
                ".daily-maintenance.lock",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=root_fd,
            )
        except FileNotFoundError as exc:
            raise ContractError("MAINTENANCE_LOCK_MISSING") from exc
        metadata = _file_stat(lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("MAINTENANCE_AUXILIARY_RUNNING") from exc
        named = os.stat(".daily-maintenance.lock", dir_fd=root_fd, follow_symlinks=False)
        if _identity(named) != _identity(metadata):
            raise ContractError("MAINTENANCE_LOCK_CHANGED")
        yield storage, root_fd
        again = _walk(storage, workspace_fd, PurePosixPath(run_root).parts, governed=False)
        try:
            if (os.fstat(again).st_dev, os.fstat(again).st_ino) != (
                os.fstat(root_fd).st_dev,
                os.fstat(root_fd).st_ino,
            ):
                raise ContractError("MAINTENANCE_ROOT_CHANGED")
        finally:
            os.close(again)
        if _identity(
            os.stat(".daily-maintenance.lock", dir_fd=root_fd, follow_symlinks=False)
        ) != _identity(metadata):
            raise ContractError("MAINTENANCE_LOCK_CHANGED")
    except OSError as exc:
        raise ContractError("MAINTENANCE_LOCK_UNSAFE") from exc
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(workspace_fd)


def _read(storage: SecureSystemStorage, root_fd: int, path: PurePosixPath) -> bytes:
    parent = _walk(storage, root_fd, path.parts[:-1])
    fd = None
    try:
        storage._reject_casefold_alias(parent, path.name)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = _file_stat(fd)
        if before.st_size > 8 * 1024 * 1024:
            raise ContractError("MAINTENANCE_RECEIPT_TOO_LARGE")
        chunks = []
        remaining = 8 * 1024 * 1024 + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = _file_stat(fd)
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(named)
        ):
            raise ContractError("MAINTENANCE_RECEIPT_CHANGED")
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def read_auxiliary_stage_records(*, workspace: str, run_root: str, core_ref: dict) -> dict:
    """Capture only exact stage bytes; native source semantics still require replay."""
    validate_ref(core_ref)
    root = PurePosixPath(run_root)
    relative = PurePosixPath(core_ref["path"]).relative_to(root)
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "attempts"
        or relative.name != "core-completion.json"
    ):
        raise ContractError("MAINTENANCE_CORE_PATH_INVALID")
    with existing_maintenance_lock(workspace=workspace, run_root=run_root) as (storage, fd):
        raw = _read(storage, fd, relative)
        if hashlib.sha256(raw).hexdigest() != core_ref["sha256"]:
            raise ContractError("MAINTENANCE_CORE_SHA_MISMATCH")
        rows: dict[str, dict] = {}
        for key, stage in (("fundamental", "FUNDAMENTAL"), ("macro", "MACRO_RELEASE")):
            path = relative.parent / f"stage-{stage}.json"
            try:
                raw = _read(storage, fd, path)
            except FileNotFoundError:
                rows[key] = {"state": "MISSING", "ref": None, "document": None}
                continue
            value = parse_canonical_json_bytes(raw)
            if (
                type(value) is not dict
                or set(value) != {"state", "result"}
                or value["state"] != "STAGE_COMPLETED"
                or type(value["result"]) is not dict
            ):
                raise ContractError("MAINTENANCE_STAGE_RECEIPT_INVALID")
            rows[key] = {
                "state": "RECORDED",
                "ref": {"path": str(root / path), "sha256": hashlib.sha256(raw).hexdigest()},
                "document": value,
            }
        return {"stages": rows, "native_replay_required": True, "execution_authorized": False}


@contextmanager
def locked_finalized_maintenance_replay(
    *,
    workspace: str,
    run_root: str,
    run_date: str,
    _catchup_binding_ref=None,
    expected_previous_trade_date=None,
):
    """Retain native maintenance exclusion while a caller repairs a missing core anchor."""
    from quant_investor.market.maintenance_journal import (
        logical_task_claim,
        _verify_finalized_attempt,
    )
    from quant_investor.market.daily_maintenance import DailyMaintenanceError

    bound = None
    if _catchup_binding_ref is not None:
        from .catchup_binding import read_catchup_binding

        bound = read_catchup_binding(workspace=workspace, binding_ref=_catchup_binding_ref)
        if bound["binding"]["trade_date"] != run_date:
            raise ContractError("FINALIZED_CATCHUP_DATE_MISMATCH")

    _validate_day(run_date)
    root = Path(workspace).resolve(strict=True) / run_root
    key = run_date + "-2020-execute"
    task = PurePosixPath("logical_tasks") / key
    observed = {}
    with existing_maintenance_lock(workspace=workspace, run_root=run_root) as (storage, fd):

        def read(path):
            relative = PurePosixPath(Path(path).relative_to(root).as_posix())
            raw = _read(storage, fd, relative)
            if relative in observed and observed[relative] != raw:
                raise ContractError("FINALIZED_MAINTENANCE_CHANGED")
            observed[relative] = raw
            return raw

        def ref(path):
            return {"path": str(path), "sha256": hashlib.sha256(read(path)).hexdigest()}

        claim_path = root / task / "claim.json"
        claim_raw = read(claim_path)
        if claim_raw != canonical_json_bytes(
            logical_task_claim(logical_key=key, slot="2020", mode="execute")
        ):
            raise ContractError("FINALIZED_MAINTENANCE_CLAIM_DRIFT")
        claim_ref = ref(claim_path)
        attempts = []
        for number in (1, 2):
            try:
                raw = read(root / task / f"attempt-{number}.json")
            except FileNotFoundError:
                continue
            row = parse_canonical_json_bytes(raw)
            if (
                type(row) is not dict
                or set(row)
                != {"logical_key", "attempt_root", "predecessor_started_ref", "claim_ref"}
                or row["logical_key"] != key
                or row["claim_ref"] != claim_ref
                or number != len(attempts) + 1
            ):
                raise ContractError("FINALIZED_MAINTENANCE_ATTEMPT_BINDING_INVALID")
            attempt = Path(row["attempt_root"])
            if attempt.parent != root / "attempts":
                raise ContractError("FINALIZED_MAINTENANCE_ATTEMPT_PATH_INVALID")
            expected_parent = ref(attempts[-1] / "started.json") if attempts else None
            if row["predecessor_started_ref"] != expected_parent:
                raise ContractError("FINALIZED_MAINTENANCE_PREDECESSOR_CHANGED")
            attempts.append(attempt)
        result = None
        if attempts:
            try:
                read(attempts[-1] / "attempt.json")
            except FileNotFoundError:
                pass
            else:
                result = _verify_finalized_attempt(
                    workspace=Path(workspace).resolve(strict=True),
                    attempt=attempts[-1],
                    claim_ref=claim_ref,
                    mode="execute",
                    read=read,
                    ref=ref,
                    error=DailyMaintenanceError,
                )
                if result is not None and bound is None:
                    from quant_investor.market.maintenance_journal import _project_requested_replay

                    result = _project_requested_replay(
                        receipt=result,
                        requested_trade_date=run_date,
                        read=read,
                        error=DailyMaintenanceError,
                        expected_previous_trade_date=expected_previous_trade_date,
                    )
                closed = (
                    result is not None
                    and result.get("requested_session_result", {}).get("classification")
                    == "CONFIRMED_CLOSED"
                )
                if result is not None and not closed and result.get("target_date") != run_date:
                    raise ContractError("FINALIZED_MAINTENANCE_TARGET_MISMATCH")
                if (
                    result is not None
                    and result.get("requested_session_result", {}).get("classification")
                    != "CONFIRMED_CLOSED"
                ):
                    expected_core = ref(attempts[-1] / "core-completion.json")
                    if result.get("core_completion_ref") != expected_core:
                        raise ContractError("FINALIZED_MAINTENANCE_CORE_REF_MISMATCH")
                    if bound is not None:
                        from quant_investor.factors.production_rollover import (
                            validate_daily_maintenance_receipt,
                        )
                        from quant_investor.market.historical_session import read_historical_core
                        from .catchup_binding import verify_bound_historical_core

                        native = validate_daily_maintenance_receipt(
                            workspace_root=Path(workspace).resolve(strict=True),
                            receipt_path=Path(expected_core["path"]),
                            expected_receipt_sha256=expected_core["sha256"],
                        )
                        core = parse_canonical_json_bytes(read(Path(expected_core["path"])))
                        if (
                            core.get("schema_version") != "cn-daily-maintenance-core.v2"
                            or native["target_date"] != run_date
                        ):
                            raise ContractError("FINALIZED_CATCHUP_CORE_INVALID")
                        attempt = attempts[-1]
                        evidence = read_historical_core(
                            attempt_root=attempt,
                            proof_ref=core["historical_session_ref"],
                            close_ref=core["close_session_receipt_ref"],
                            started_ref=core["started_ref"],
                            target=run_date,
                            read=lambda path, **kwargs: read(path),
                        )
                        verify_bound_historical_core(
                            derived=bound,
                            proof=evidence["historical_session"],
                            calendar=parse_canonical_json_bytes(
                                read(attempt / "close-session-receipt.json")
                            ),
                            raw=read(attempt / "close-session.raw.json"),
                        )
        for path, raw in observed.items():
            if _read(storage, fd, path) != raw:
                raise ContractError("FINALIZED_MAINTENANCE_CHANGED")
        if bound is not None:
            bound["sources"].recheck()
        yield result
        if bound is not None:
            bound["sources"].recheck()
        for path, raw in observed.items():
            if _read(storage, fd, path) != raw:
                raise ContractError("FINALIZED_MAINTENANCE_CHANGED")
