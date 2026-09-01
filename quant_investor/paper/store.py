"""Owner-only immutable Paper account store with pointer CAS."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Callable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import canonical_json_bytes

from .contracts import PAPER_ROOT, PaperError, seal_document, validate_registration

FaultHook = Callable[[str], None]

_LEDGER_SCHEMA = pa.schema(
    [
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("shares", pa.int64(), nullable=False),
        pa.field("settled_shares", pa.int64(), nullable=False),
        pa.field("avg_cost", pa.decimal128(20, 4), nullable=False),
        pa.field("cost_basis", pa.decimal128(20, 4), nullable=False),
        pa.field("realized_pnl", pa.decimal128(20, 4), nullable=False),
        pa.field("cumulative_fees", pa.decimal128(20, 4), nullable=False),
        pa.field("last_trade_date", pa.string(), nullable=False),
        pa.field("last_fill_id", pa.string(), nullable=False),
        pa.field("acquisition_lots_json", pa.string(), nullable=False),
    ]
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_file(path: Path, *, required: bool = True) -> tuple[bytes, os.stat_result] | None:
    try:
        before = path.lstat()
    except FileNotFoundError:
        if required:
            raise PaperError("PAPER_STORAGE_NOT_FOUND", str(path)) from None
        return None
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise PaperError("PAPER_STORAGE_SECURITY", str(path))
    first = path.read_bytes()
    middle = path.lstat()
    second = path.read_bytes()
    after = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        identity(before) != identity(middle)
        or identity(middle) != identity(after)
        or first != second
    ):
        raise PaperError("PAPER_STORAGE_STABLE_READ_FAILED", str(path))
    return first, after


def _safe_dir(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PaperError("PAPER_STORAGE_SECURITY", str(path)) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PaperError("PAPER_STORAGE_SECURITY", str(path))
    return path


def _write_new(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _safe_file(path)
        if existing is None or existing[0] != raw:
            raise PaperError("PAPER_IMMUTABLE_CONFLICT", str(path))
        return
    try:
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PaperStore:
    """One independent Paper authority rooted below the supplied workspace."""

    def __init__(
        self, workspace_root: str | os.PathLike[str], *, fault_hook: FaultHook | None = None
    ):
        self.workspace = Path(workspace_root).resolve(strict=True)
        self.root = self.workspace / PAPER_ROOT
        self._fault_hook = fault_hook or (lambda _point: None)

    def account_root(self, account_id: str) -> Path:
        if not account_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in account_id
        ):
            raise PaperError("PAPER_ACCOUNT_ID_INVALID", account_id)
        path = self.root / account_id
        if self.root != path.parent:
            raise PaperError("PAPER_STORAGE_SECURITY", "account escapes root")
        return path

    def account_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        _safe_dir(self.root)
        rows = []
        for child in self.root.iterdir():
            if child.name.startswith("."):
                continue
            _safe_dir(child)
            rows.append(child.name)
        return sorted(rows, key=lambda text: text.encode("ascii"))

    @contextmanager
    def lock(self, account_id: str, *, create: bool) -> Iterator[Path]:
        account = self.account_root(account_id)
        if create:
            _safe_dir(self.root, create=True)
            _safe_dir(account, create=True)
        else:
            _safe_dir(account)
        lock_path = account / ".writer.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise PaperError("PAPER_STORAGE_SECURITY", "lock unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PaperError("PAPER_ACCOUNT_LOCKED", account_id) from None
            yield account
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError as exc:
            raise PaperError("PAPER_STORAGE_SECURITY", "path escapes workspace") from exc

    def _ref(self, path: Path) -> dict[str, str]:
        stored = _safe_file(path)
        if stored is None:
            raise PaperError("PAPER_STORAGE_NOT_FOUND", str(path))
        return {"path": self._relative(path), "sha256": _sha(stored[0])}

    def _read_json(self, path: Path) -> dict[str, Any]:
        stored = _safe_file(path)
        if stored is None:
            raise PaperError("PAPER_STORAGE_NOT_FOUND", str(path))
        try:
            value = json.loads(stored[0])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PaperError("PAPER_STORAGE_INVALID", str(path)) from exc
        if type(value) is not dict or canonical_json_bytes(value) != stored[0]:
            raise PaperError("PAPER_STORAGE_INVALID", str(path))
        return value

    def _ledger_rows(self, path: Path) -> list[dict[str, Any]]:
        stored = _safe_file(path)
        if stored is None:
            raise PaperError("PAPER_STORAGE_NOT_FOUND", str(path))
        table = pq.read_table(pa.BufferReader(stored[0]))
        if table.schema != _LEDGER_SCHEMA:
            raise PaperError("PAPER_LEDGER_VERIFY_FAILED", "schema differs")
        rows = table.to_pylist()
        symbols = [row["symbol"] for row in rows]
        if symbols != sorted(set(symbols), key=lambda text: text.encode("ascii")):
            raise PaperError("PAPER_LEDGER_VERIFY_FAILED", "row order differs")
        return rows

    def _write_ledger(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "account_id": row["account_id"],
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "shares": int(row["shares"]),
                    "settled_shares": int(row["settled_shares"]),
                    "avg_cost": Decimal(str(row["avg_cost"])),
                    "cost_basis": Decimal(str(row["cost_basis"])),
                    "realized_pnl": Decimal(str(row["realized_pnl"])),
                    "cumulative_fees": Decimal(str(row["cumulative_fees"])),
                    "last_trade_date": str(row.get("last_trade_date") or ""),
                    "last_fill_id": str(row.get("last_fill_id") or ""),
                    "acquisition_lots_json": (
                        str(row["acquisition_lots_json"])
                        if "acquisition_lots_json" in row
                        else json.dumps(
                            row["acquisition_lots"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                }
            )
        table = pa.Table.from_pylist(normalized, schema=_LEDGER_SCHEMA)
        pq.write_table(table, path, compression="zstd", version="2.6", write_statistics=True)
        os.chmod(path, 0o600)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _current_path(self, account: Path) -> Path:
        return account / "_record_store/current.v1.json"

    def _load_current(self, account: Path) -> tuple[bytes, dict[str, Any], str]:
        stored = _safe_file(self._current_path(account))
        if stored is None:
            raise PaperError("PAPER_CURRENT_POINTER_INVALID", "pointer absent")
        raw = stored[0]
        value = self._read_json(self._current_path(account))
        if value.get("schema_version") != "paper-current-pointer.v1":
            raise PaperError("PAPER_CURRENT_POINTER_INVALID", "schema differs")
        return raw, value, _sha(raw)

    def load_account(self, account_id: str) -> dict[str, Any]:
        account = self.account_root(account_id)
        _safe_dir(account)
        registration = self._read_json(account / "registration.v1.json")
        validate_registration(registration)
        pointer_raw, pointer, pointer_sha = self._load_current(account)
        closure_ref = pointer.get("active_closure_ref")
        if type(closure_ref) is not dict:
            raise PaperError("PAPER_CURRENT_POINTER_INVALID", "closure ref absent")
        closure_path = self.workspace / closure_ref["path"]
        closure = self._read_json(closure_path)
        if _sha(canonical_json_bytes(closure)) != closure_ref["sha256"]:
            raise PaperError("PAPER_CURRENT_POINTER_INVALID", "closure SHA differs")
        state_path = self.workspace / closure["account_state_ref"]["path"]
        ledger_path = self.workspace / closure["ledger_ref"]["path"]
        state = self._read_json(state_path)
        ledger = self._ledger_rows(ledger_path)
        if (
            self._ref(state_path)["sha256"] != closure["account_state_ref"]["sha256"]
            or self._ref(ledger_path)["sha256"] != closure["ledger_ref"]["sha256"]
        ):
            raise PaperError("PAPER_CURRENT_POINTER_INVALID", "state or ledger SHA differs")
        transaction_ref = closure.get("transaction_receipt_ref")
        if transaction_ref is not None:
            transaction_path = self.workspace / transaction_ref["path"]
            transaction = self._read_json(transaction_path)
            if self._ref(transaction_path)["sha256"] != transaction_ref["sha256"]:
                raise PaperError("PAPER_CURRENT_POINTER_INVALID", "transaction SHA differs")
            if (
                transaction.get("account_id") != account_id
                or transaction.get("sequence") != pointer.get("sequence")
                or transaction.get("account_state_ref") != closure["account_state_ref"]
                or transaction.get("ledger_ref") != closure["ledger_ref"]
            ):
                raise PaperError("PAPER_CURRENT_POINTER_INVALID", "transaction closure differs")
        if state.get("account_id") != account_id or state.get("sequence") != pointer.get(
            "sequence"
        ):
            raise PaperError("PAPER_ACCOUNT_STATE_INVALID", "state/pointer differs")
        return {
            "account_root": account,
            "registration": registration,
            "registration_ref": self._ref(account / "registration.v1.json"),
            "pointer": pointer,
            "pointer_raw": pointer_raw,
            "pointer_sha256": pointer_sha,
            "closure": closure,
            "closure_ref": dict(closure_ref),
            "state": state,
            "ledger": ledger,
        }

    def register(self, registration: Mapping[str, Any]) -> dict[str, Any]:
        value = validate_registration(registration)
        account_id = value["account_id"]
        account = self.account_root(account_id)
        if account.exists():
            raise PaperError("PAPER_ACCOUNT_ALREADY_REGISTERED", account_id)
        with self.lock(account_id, create=True):
            self._fault_hook("REGISTER_AFTER_LOCK")
            _safe_dir(account / "records", create=True)
            _safe_dir(account / "_record_store", create=True)
            _safe_dir(account / "_record_store/pointer_history", create=True)
            registration_raw = canonical_json_bytes(dict(registration))
            _write_new(account / "registration.v1.json", registration_raw)
            record = account / "records/000000-genesis"
            _safe_dir(record, create=True)
            ledger_rows = [
                {
                    "account_id": account_id,
                    **position,
                    "last_trade_date": "",
                    "last_fill_id": "",
                }
                for position in value["initial_positions"]
            ]
            self._write_ledger(record / "ledger_after.parquet", ledger_rows)
            state = seal_document(
                {
                    "schema_version": "paper-account-state.v1",
                    "account_id": account_id,
                    "sequence": 0,
                    "as_of_trade_date": "00000000",
                    "cash": value["initial_cash"],
                    "realized_pnl": "0.0000",
                    "cumulative_fees": "0.0000",
                    "positions": value["initial_positions"],
                    "applied_source_intents": {},
                    "applied_economic_actions": {},
                    "pending_intents": {},
                    "broker": False,
                    "real_order": False,
                    "actual_holdings_mutation": False,
                }
            )
            _write_new(record / "account_state_after.v1.json", canonical_json_bytes(state))
            registration_ref = self._ref(account / "registration.v1.json")
            writer_path = record / "writer-registration.v1.json"
            from .contracts import writer_registration

            _write_new(writer_path, canonical_json_bytes(writer_registration()))
            closure = seal_document(
                {
                    "schema_version": "paper-closure.v1",
                    "closure_id": "paper-closure-genesis-" + account_id,
                    "account_id": account_id,
                    "sequence": 0,
                    "predecessor_closure_ref": None,
                    "transaction_receipt_ref": None,
                    "account_state_ref": self._ref(record / "account_state_after.v1.json"),
                    "ledger_ref": self._ref(record / "ledger_after.parquet"),
                    "source_intent_ref": None,
                    "eligibility_ref": None,
                    "policy_ref": dict(value["policy_ref"]),
                    "registration_ref": registration_ref,
                    "writer_registration_ref": self._ref(writer_path),
                }
            )
            _write_new(record / "closure.v1.json", canonical_json_bytes(closure))
            pointer = seal_document(
                {
                    "schema_version": "paper-current-pointer.v1",
                    "account_id": account_id,
                    "sequence": 0,
                    "active_closure_ref": self._ref(record / "closure.v1.json"),
                    "previous_pointer_sha256": "EMPTY",
                }
            )
            self._fault_hook("REGISTER_BEFORE_POINTER")
            _write_new(self._current_path(account), canonical_json_bytes(pointer))
            _fsync_dir(account / "_record_store")
            _fsync_dir(account)
            loaded = self.load_account(account_id)
            return {
                "command_status": "REGISTERED",
                "account_id": account_id,
                "pointer_sha256": loaded["pointer_sha256"],
                "sequence": 0,
            }

    def commit(
        self,
        *,
        account_id: str,
        expected_pointer_sha256: str,
        intent: Mapping[str, Any],
        intent_ref: Mapping[str, str],
        eligibility: Mapping[str, Any],
        eligibility_ref: Mapping[str, str],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self.lock(account_id, create=False):
            before = self.load_account(account_id)
            if before["pointer_sha256"] != expected_pointer_sha256:
                raise PaperError("PAPER_COMPARE_AND_SWAP_CONFLICT", "pointer preimage differs")
            source_id = intent["source_intent_id"]
            economic_key = intent["economic_action_key_sha256"]
            intent_sha = intent_ref["sha256"]
            applied_source = dict(before["state"].get("applied_source_intents") or {})
            applied_economic = dict(before["state"].get("applied_economic_actions") or {})
            pending_map = dict(before["state"].get("pending_intents") or {})
            for mapping, key in ((applied_source, source_id), (applied_economic, economic_key)):
                if key in mapping:
                    if mapping[key].get("intent_sha256") == intent_sha:
                        return {
                            "command_status": "NO_ACTION_ALREADY_APPLIED",
                            "account_id": account_id,
                            "pointer_sha256": before["pointer_sha256"],
                            "sequence": before["pointer"]["sequence"],
                        }
                    raise PaperError("PAPER_IDEMPOTENCY_CONFLICT", key)
            existing_pending = pending_map.get(source_id)
            if existing_pending and existing_pending.get("intent_sha256") != intent_sha:
                raise PaperError("PAPER_IDEMPOTENCY_CONFLICT", source_id)

            sequence = int(before["pointer"]["sequence"]) + 1
            identity_input = {
                "account_id": account_id,
                "sequence": sequence,
                "previous_pointer_sha256": before["pointer_sha256"],
                "intent_sha256": intent_sha,
                "eligibility_sha256": eligibility_ref["sha256"],
                "outcome": outcome["outcome"],
            }
            identity = _sha(canonical_json_bytes(identity_input))
            record_name = f"{sequence:06d}-{identity[:16]}"
            final = before["account_root"] / "records" / record_name
            final_relative = self._relative(final)
            stage = (
                before["account_root"] / "records" / f".stage-{record_name}-{secrets.token_hex(6)}"
            )
            if final.exists():
                raise PaperError("PAPER_ORPHAN_CONFLICT", record_name)
            _safe_dir(stage, create=True)
            self._fault_hook("AFTER_STAGE_CREATE")
            _write_new(stage / "intents.v1.json", canonical_json_bytes(dict(intent)))
            _write_new(stage / "input-eligibility.v1.json", canonical_json_bytes(dict(eligibility)))

            ledger = [dict(row) for row in before["ledger"]]
            state = dict(before["state"])
            state.pop("semantic_sha256", None)
            state["sequence"] = sequence
            state["as_of_trade_date"] = eligibility["evaluated_trade_date"]
            state["positions"] = [dict(item) for item in state["positions"]]
            order_ref = None
            fill_ref = None
            pending_ref = None
            status = outcome["outcome"]
            if outcome["outcome"] == "FILLED":
                _write_new(stage / "orders.v1.json", canonical_json_bytes(outcome["order"]))
                order_ref = {
                    "path": f"{final_relative}/orders.v1.json",
                    "sha256": _sha(canonical_json_bytes(outcome["order"])),
                }
                fill_value = dict(outcome["fill"])
                fill_value["order_ref"] = order_ref
                _write_new(stage / "fills.v1.json", canonical_json_bytes(fill_value))
                fill_ref = {
                    "path": f"{final_relative}/fills.v1.json",
                    "sha256": _sha(canonical_json_bytes(fill_value)),
                }
                symbol = intent["symbol"]
                target = next((row for row in ledger if row["symbol"] == symbol), None)
                if target is None:
                    raise PaperError("PAPER_POSITION_MISMATCH", symbol)
                accounting = outcome["accounting"]
                target["shares"] = accounting["shares_after"]
                target["settled_shares"] = min(target["settled_shares"], target["shares"])
                target["cost_basis"] = accounting["cost_basis_after"]
                target["realized_pnl"] = format(
                    Decimal(str(target["realized_pnl"]))
                    + Decimal(accounting["realized_pnl_delta"]),
                    ".4f",
                )
                target["cumulative_fees"] = format(
                    Decimal(str(target["cumulative_fees"]))
                    + Decimal(accounting["cumulative_fees_delta"]),
                    ".4f",
                )
                target["last_trade_date"] = eligibility["evaluated_trade_date"]
                target["last_fill_id"] = fill_value["fill_id"]
                state["cash"] = accounting["cash_after"]
                state["realized_pnl"] = format(
                    Decimal(state["realized_pnl"]) + Decimal(accounting["realized_pnl_delta"]),
                    ".4f",
                )
                state["cumulative_fees"] = format(
                    Decimal(state["cumulative_fees"])
                    + Decimal(accounting["cumulative_fees_delta"]),
                    ".4f",
                )
                applied_source[source_id] = {"intent_sha256": intent_sha, "outcome": "FILLED"}
                applied_economic[economic_key] = {"intent_sha256": intent_sha, "outcome": "FILLED"}
                pending_map.pop(source_id, None)
            else:
                _write_new(stage / "pending.v1.json", canonical_json_bytes(outcome["pending"]))
                pending_ref = {
                    "path": f"{final_relative}/pending.v1.json",
                    "sha256": _sha(canonical_json_bytes(outcome["pending"])),
                }
                if outcome["outcome"] == "EXPIRED":
                    applied_source[source_id] = {"intent_sha256": intent_sha, "outcome": "EXPIRED"}
                    applied_economic[economic_key] = {
                        "intent_sha256": intent_sha,
                        "outcome": "EXPIRED",
                    }
                    pending_map.pop(source_id, None)
                else:
                    pending_map[source_id] = {
                        "intent_sha256": intent_sha,
                        "economic_key": economic_key,
                        "pending_ref": pending_ref,
                        "evaluated_open_session_count": outcome["pending"][
                            "evaluated_open_session_count"
                        ],
                    }
            state["applied_source_intents"] = applied_source
            state["applied_economic_actions"] = applied_economic
            state["pending_intents"] = pending_map
            state["positions"] = [
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "shares": row["shares"],
                    "settled_shares": row["settled_shares"],
                    "avg_cost": str(row["avg_cost"]),
                    "cost_basis": str(row["cost_basis"]),
                    "realized_pnl": str(row["realized_pnl"]),
                    "cumulative_fees": str(row["cumulative_fees"]),
                    "acquisition_lots": (
                        json.loads(row["acquisition_lots_json"])
                        if "acquisition_lots_json" in row
                        else row["acquisition_lots"]
                    ),
                }
                for row in ledger
            ]
            state = seal_document(state)
            self._write_ledger(stage / "ledger_after.parquet", ledger)
            _write_new(stage / "account_state_after.v1.json", canonical_json_bytes(state))
            writer_path = stage / "writer-registration.v1.json"
            from .contracts import writer_registration

            _write_new(writer_path, canonical_json_bytes(writer_registration()))
            receipt = seal_document(
                {
                    "schema_version": "paper-transaction-receipt.v1",
                    "transaction_id": "paper-transaction-" + identity,
                    "account_id": account_id,
                    "sequence": sequence,
                    "previous_pointer_sha256": before["pointer_sha256"],
                    "registration_ref": before["registration_ref"],
                    "writer_registration_ref": {
                        "path": f"{final_relative}/writer-registration.v1.json",
                        "sha256": _sha(canonical_json_bytes(writer_registration())),
                    },
                    "policy_ref": dict(intent["policy_ref"]),
                    "intent_ref": dict(intent_ref),
                    "eligibility_ref": dict(eligibility_ref),
                    "order_ref": order_ref,
                    "fill_ref": fill_ref,
                    "pending_ref": pending_ref,
                    "ledger_ref": {
                        "path": f"{final_relative}/ledger_after.parquet",
                        "sha256": self._ref(stage / "ledger_after.parquet")["sha256"],
                    },
                    "account_state_ref": {
                        "path": f"{final_relative}/account_state_after.v1.json",
                        "sha256": _sha(canonical_json_bytes(state)),
                    },
                    "write_set": sorted(
                        [
                            "account_state_after.v1.json",
                            "input-eligibility.v1.json",
                            "intents.v1.json",
                            "ledger_after.parquet",
                            "writer-registration.v1.json",
                        ]
                        + (["orders.v1.json", "fills.v1.json"] if fill_ref else ["pending.v1.json"])
                    ),
                    "command_status": status,
                    "broker": False,
                    "real_order": False,
                    "actual_holdings_mutation": False,
                }
            )
            _write_new(stage / "transaction_receipt.v1.json", canonical_json_bytes(receipt))
            closure = seal_document(
                {
                    "schema_version": "paper-closure.v1",
                    "closure_id": "paper-closure-" + identity,
                    "account_id": account_id,
                    "sequence": sequence,
                    "predecessor_closure_ref": before["closure_ref"],
                    "transaction_receipt_ref": {
                        "path": f"{final_relative}/transaction_receipt.v1.json",
                        "sha256": _sha(canonical_json_bytes(receipt)),
                    },
                    "account_state_ref": {
                        "path": f"{final_relative}/account_state_after.v1.json",
                        "sha256": _sha(canonical_json_bytes(state)),
                    },
                    "ledger_ref": {
                        "path": f"{final_relative}/ledger_after.parquet",
                        "sha256": self._ref(stage / "ledger_after.parquet")["sha256"],
                    },
                    "source_intent_ref": dict(intent_ref),
                    "eligibility_ref": dict(eligibility_ref),
                    "policy_ref": dict(intent["policy_ref"]),
                    "registration_ref": before["registration_ref"],
                    "writer_registration_ref": {
                        "path": f"{final_relative}/writer-registration.v1.json",
                        "sha256": _sha(canonical_json_bytes(writer_registration())),
                    },
                }
            )
            _write_new(stage / "closure.v1.json", canonical_json_bytes(closure))
            self._fault_hook("BEFORE_RECORD_RENAME")
            _fsync_dir(stage)
            os.rename(stage, final)
            _fsync_dir(final.parent)
            self._fault_hook("AFTER_RECORD_RENAME")
            current_path = self._current_path(before["account_root"])
            current_raw = _safe_file(current_path)
            if current_raw is None or _sha(current_raw[0]) != expected_pointer_sha256:
                raise PaperError("PAPER_COMPARE_AND_SWAP_CONFLICT", "pointer changed before CAS")
            history = (
                before["account_root"]
                / "_record_store/pointer_history"
                / f"{expected_pointer_sha256}.json"
            )
            _write_new(history, current_raw[0])
            _fsync_dir(history.parent)
            pointer = seal_document(
                {
                    "schema_version": "paper-current-pointer.v1",
                    "account_id": account_id,
                    "sequence": sequence,
                    "active_closure_ref": self._ref(final / "closure.v1.json"),
                    "previous_pointer_sha256": expected_pointer_sha256,
                }
            )
            pointer_raw = canonical_json_bytes(pointer)
            temporary = current_path.parent / f".current-{secrets.token_hex(8)}"
            _write_new(temporary, pointer_raw)
            self._fault_hook("BEFORE_POINTER_CAS")
            os.replace(temporary, current_path)
            _fsync_dir(current_path.parent)
            self._fault_hook("AFTER_POINTER_CAS")
            loaded = self.load_account(account_id)
            if loaded["pointer"]["active_closure_ref"] != self._ref(final / "closure.v1.json"):
                raise PaperError("PAPER_POINTER_READBACK_FAILED", account_id)
            return {
                "command_status": status,
                "account_id": account_id,
                "sequence": sequence,
                "pointer_sha256": loaded["pointer_sha256"],
                "record_path": self._relative(final),
                "broker": False,
                "real_order": False,
                "actual_holdings_mutation": False,
            }


__all__ = ["PaperStore"]
