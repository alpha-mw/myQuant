"""In-process immutable capability for completed-history handoff reads."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError

_SEAL = object()
_ROLES = frozenset(
    {
        "completion",
        "ledger",
        "materialization",
        "handoff",
        "recipe",
        "loop_context",
        "release_install_input",
        "logical_claim",
    }
)


@dataclass(frozen=True, init=False)
class CompletedHandoffSnapshot:
    workspace: str
    trade_date: str
    documents: tuple[tuple[str, str, str, bytes], ...]
    _seal: object

    def __init__(self, *args, **kwargs):
        raise ContractError("COMPLETED_SNAPSHOT_FACTORY_REQUIRED")

    def document(self, role: str) -> dict:
        _require_snapshot(self)
        for name, _, _, raw in self.documents:
            if name == role:
                return parse_canonical_json_bytes(raw)
        raise ContractError("COMPLETED_SNAPSHOT_ROLE_INVALID")

    def reference(self, role: str) -> dict:
        _require_snapshot(self)
        for name, path, sha, _ in self.documents:
            if name == role:
                return {"path": path, "sha256": sha}
        raise ContractError("COMPLETED_SNAPSHOT_ROLE_INVALID")

    def recheck(self) -> None:
        _require_snapshot(self)
        reader = SecureSystemStorage(self.workspace)
        for _, path, sha, raw in self.documents:
            current = reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024)
            if current.byte_sha256 != sha or current.data != raw:
                raise ContractError("COMPLETED_SNAPSHOT_CHANGED")


def _require_snapshot(value) -> None:
    if type(value) is not CompletedHandoffSnapshot or getattr(value, "_seal", None) is not _SEAL:
        raise ContractError("COMPLETED_SNAPSHOT_CAPABILITY_REQUIRED")


def _mint_snapshot(
    *, workspace: str, trade_date: str, documents: tuple
) -> CompletedHandoffSnapshot:
    """Private factory called solely after completed binding validation."""
    roles = {row[0] for row in documents}
    if len(documents) != len(roles) or roles not in (_ROLES, _ROLES | {"automatic_origin"}):
        raise ContractError("COMPLETED_SNAPSHOT_ROLES_INVALID")
    for _, _, sha, raw in documents:
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != sha:
            raise ContractError("COMPLETED_SNAPSHOT_BYTES_INVALID")
    handoff = parse_canonical_json_bytes(next(row[3] for row in documents if row[0] == "handoff"))
    if (handoff.get("schema_version") == "cn-daily-maintenance-handoff.v4") != (
        "automatic_origin" in roles
    ):
        raise ContractError("COMPLETED_SNAPSHOT_ORIGIN_ROLE_INVALID")
    if "automatic_origin" in roles:
        _, path, sha, _ = next(row for row in documents if row[0] == "automatic_origin")
        if handoff["automatic_origin_ref"] != {"path": path, "sha256": sha}:
            raise ContractError("COMPLETED_SNAPSHOT_ORIGIN_REF_MISMATCH")
    result = object.__new__(CompletedHandoffSnapshot)
    object.__setattr__(result, "workspace", str(Path(workspace).resolve(strict=True)))
    object.__setattr__(result, "trade_date", trade_date)
    object.__setattr__(result, "documents", tuple(tuple(row) for row in documents))
    object.__setattr__(result, "_seal", _SEAL)
    return result
