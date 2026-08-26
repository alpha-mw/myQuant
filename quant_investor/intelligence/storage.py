"""Immutable research-only policy and daily Factor-pool publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Final

from quant_investor.contracts import artifact_byte_sha256, canonical_json_bytes
from quant_investor.migration.canonical import read_stable_regular_file

from ._common import (
    IntelligenceError,
    artifact_ref,
    build_artifact,
    business_identity,
    identifier,
    sha256,
)
from .daily import (
    build_daily_research_policy,
    validate_daily_research_policy,
    validate_factor_research_rank,
)
from .theme_governance import (
    EFFECTIVE_SIGNAL_DATE,
    OWNER_APPROVED_AT,
    TECHNOLOGY_THEME_IDS,
    approved_theme_governance_policy,
    validate_theme_governance_policy,
)

PHASE_A_POLICY_RELATIVE_PATH: Final = (
    "results/policies/research/aggressive_tech_manufacturing/v1.json"
)
THEME_POLICY_V2_RELATIVE_PATH: Final = (
    "results/policies/research/aggressive_tech_manufacturing/v2.json"
)
THEME_GOVERNANCE_POLICY_RELATIVE_PATH: Final = (
    "results/policies/research/aggressive_tech_manufacturing/theme-governance.v1.json"
)
POOL_ROOT_RELATIVE_PATH: Final = "results/intelligence/research_pool"
POOL_STRATEGY_ID: Final = "aggressive_tech_manufacturing"
POOL_LEAF_NAMES: Final = (
    "factor_research_rank.json",
    "manifest.json",
    "publish_receipt.json",
    "selected_symbols.json",
)
_DARWIN_RENAME_EXCL: Final = 0x00000004
_LINUX_RENAME_NOREPLACE: Final = 1


def approved_phase_a_policy() -> dict[str, Any]:
    """Return the exact owner-approved prospective Phase A policy."""

    return build_daily_research_policy(
        strategy_id=POOL_STRATEGY_ID,
        effective_from="2026-08-21T16:00:00Z",
        effective_signal_date="20260822",
        effective_to=None,
        factor_rows=[
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "LOW",
                "factor_id": "pv_low_dollar_volume_5d",
                "weight": "0.5",
            },
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "W80",
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "weight": "0.5",
            },
        ],
        pool_policy={
            "minimum_cohort": 3000,
            "missing_rule": "BLOCK_ON_ANY_MISSING_OR_NONFINITE",
            "normalization": "AVERAGE_TIE_PERCENTILE_ASCENDING_ZERO_ONE",
            "pool_boundary_rule": "EXACT_LIMIT_ASCII_SYMBOL_TIEBREAK",
            "pool_size": 100,
            "sort_key": "DESC_COMBINED_PERCENTILE_ASCII_SYMBOL",
            "tie_rule": "AVERAGE_ORDINAL_PERCENTILE",
        },
        decision_thresholds={
            "paper_candidate": "0.90",
            "research_approved": "0.80",
        },
        technology_theme_ids=[],
        technology_policy_state="UNCONFIGURED",
        theme_provider_precedence=["TUSHARE_DC", "TUSHARE_TDX"],
        fundamental_freshness={"policy": "ADVISORY_NO_FIXED_MAXIMUM"},
        created_at="2026-08-21T16:00:00Z",
    )


def approved_theme_policy_v2() -> dict[str, Any]:
    """Return the later-effective ACTIVE Theme policy approved by Maxwell."""

    return build_daily_research_policy(
        strategy_id=POOL_STRATEGY_ID,
        effective_from=OWNER_APPROVED_AT,
        effective_signal_date=EFFECTIVE_SIGNAL_DATE,
        effective_to=None,
        factor_rows=[
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "LOW",
                "factor_id": "pv_low_dollar_volume_5d",
                "weight": "0.5",
            },
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "W80",
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "weight": "0.5",
            },
        ],
        pool_policy={
            "minimum_cohort": 3000,
            "missing_rule": "BLOCK_ON_ANY_MISSING_OR_NONFINITE",
            "normalization": "AVERAGE_TIE_PERCENTILE_ASCENDING_ZERO_ONE",
            "pool_boundary_rule": "EXACT_LIMIT_ASCII_SYMBOL_TIEBREAK",
            "pool_size": 100,
            "sort_key": "DESC_COMBINED_PERCENTILE_ASCII_SYMBOL",
            "tie_rule": "AVERAGE_ORDINAL_PERCENTILE",
        },
        decision_thresholds={
            "paper_candidate": "0.90",
            "research_approved": "0.80",
        },
        technology_theme_ids=list(TECHNOLOGY_THEME_IDS),
        technology_policy_state="ACTIVE",
        theme_provider_precedence=["TUSHARE_DC", "TUSHARE_TDX"],
        fundamental_freshness={"policy": "ADVISORY_NO_FIXED_MAXIMUM"},
        created_at=OWNER_APPROVED_AT,
    )


def _workspace(value: str | os.PathLike[str]) -> Path:
    try:
        root = Path(value).resolve(strict=True)
        observed = os.lstat(root)
    except OSError as exc:
        raise IntelligenceError("workspace root is invalid") from exc
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise IntelligenceError("workspace root is invalid")
    return root


def _verify_owned_directory(path: Path, *, exact_mode: bool) -> None:
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or (exact_mode and stat.S_IMODE(observed.st_mode) != 0o700)
    ):
        raise IntelligenceError("research store directory is unsafe")


def _store_parent(root: Path, parts: tuple[str, ...]) -> Path:
    current = root / "results"
    if current.exists():
        _verify_owned_directory(current, exact_mode=False)
    else:
        current.mkdir(mode=0o700)
    for part in parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _verify_owned_directory(current, exact_mode=True)
    return current


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(raw):
        written += os.write(descriptor, view[written:])


def _verify_policy_file(path: Path, expected: bytes) -> None:
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or read_stable_regular_file(path, label="daily research policy") != expected
    ):
        raise IntelligenceError("daily research policy immutable conflict")


def _publish_phase_a_policy(
    workspace_root: str | os.PathLike[str],
    *,
    before_publish: Callable[[], None],
) -> dict[str, Any]:
    root = _workspace(workspace_root)
    policy = approved_phase_a_policy()
    raw = canonical_json_bytes(policy)
    parent = _store_parent(root, ("policies", "research", POOL_STRATEGY_ID))
    path = parent / "v1.json"
    try:
        os.lstat(path)
    except FileNotFoundError:
        exists = False
    else:
        exists = True
    if exists:
        _verify_policy_file(path, raw)
        created = False
    else:
        temporary = parent / f".v1.json.publish-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, raw)
            os.fsync(descriptor)
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or observed.st_nlink != 1
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise IntelligenceError("daily research policy temporary file is unsafe")
        finally:
            os.close(descriptor)
        _verify_policy_file(temporary, raw)
        before_publish()
        try:
            _atomic_no_replace(temporary, path)
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            created = True
        except FileExistsError:
            _verify_policy_file(path, raw)
            temporary.unlink()
            created = False
        _verify_policy_file(path, raw)
    observed = read_stable_regular_file(path, label="daily research policy")
    if validate_daily_research_policy(observed) != policy:
        raise IntelligenceError("daily research policy readback differs")
    return {
        "command_status": "PUBLISHED" if created else "NO_ACTION",
        "policy_path": PHASE_A_POLICY_RELATIVE_PATH,
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_ref": artifact_ref(policy),
        "research_only": True,
        "production": False,
        "grants_trading_authority": False,
    }


def publish_phase_a_policy(workspace_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Publish the one code-owned immutable v1 policy or replay it exactly."""

    return _publish_phase_a_policy(workspace_root, before_publish=lambda: None)


def _publish_exact_policy_artifact(
    *,
    root: Path,
    relative_path: str,
    artifact: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any] | bytes], dict[str, Any]],
) -> dict[str, Any]:
    raw = canonical_json_bytes(artifact)
    parent = _store_parent(root, ("policies", "research", POOL_STRATEGY_ID))
    path = root / relative_path
    if path.parent != parent:
        raise IntelligenceError("research policy path is outside the governed parent")
    if path.exists():
        _verify_policy_file(path, raw)
        created = False
    else:
        temporary = parent / f".{path.name}.publish-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _verify_policy_file(temporary, raw)
        try:
            _atomic_no_replace(temporary, path)
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            created = True
        except FileExistsError:
            _verify_policy_file(path, raw)
            temporary.unlink()
            created = False
        _verify_policy_file(path, raw)
    observed = read_stable_regular_file(path, label=f"research policy {path.name}")
    if validator(observed) != artifact:
        raise IntelligenceError("research policy readback differs")
    return {
        "created": created,
        "path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ref": artifact_ref(artifact),
    }


def publish_theme_policy_v2(workspace_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Publish the approved governance artifact, then the ACTIVE daily v2 policy."""

    root = _workspace(workspace_root)
    governance = approved_theme_governance_policy()
    daily = approved_theme_policy_v2()
    if daily["payload"]["technology_theme_ids"] != governance["payload"]["technology_theme_ids"]:
        raise IntelligenceError("daily and governance Theme IDs differ")
    governance_result = _publish_exact_policy_artifact(
        root=root,
        relative_path=THEME_GOVERNANCE_POLICY_RELATIVE_PATH,
        artifact=governance,
        validator=validate_theme_governance_policy,
    )
    daily_result = _publish_exact_policy_artifact(
        root=root,
        relative_path=THEME_POLICY_V2_RELATIVE_PATH,
        artifact=daily,
        validator=validate_daily_research_policy,
    )
    return {
        "command_status": (
            "PUBLISHED" if governance_result["created"] or daily_result["created"] else "NO_ACTION"
        ),
        "daily_policy_path": daily_result["path"],
        "daily_policy_sha256": daily_result["sha256"],
        "daily_policy_ref": daily_result["ref"],
        "effective_signal_date": EFFECTIVE_SIGNAL_DATE,
        "governance_policy_path": governance_result["path"],
        "governance_policy_sha256": governance_result["sha256"],
        "governance_policy_ref": governance_result["ref"],
        "research_only": True,
        "production": False,
        "grants_trading_authority": False,
    }


def approved_pool_policy(relative_path: str) -> dict[str, Any]:
    if relative_path == PHASE_A_POLICY_RELATIVE_PATH:
        return approved_phase_a_policy()
    if relative_path == THEME_POLICY_V2_RELATIVE_PATH:
        return approved_theme_policy_v2()
    raise IntelligenceError("research pool policy path is not approved")


def _selected_symbols(rank: Mapping[str, Any], *, created_at: str) -> dict[str, Any]:
    payload = rank["payload"]
    symbols = [row["symbol"] for row in payload["pool_rows"]]
    symbol_set_sha256 = hashlib.sha256(canonical_json_bytes(symbols)).hexdigest()
    return build_artifact(
        kind="daily_research_selected_symbols",
        identity_field="selection_id",
        identity=business_identity(
            kind="daily_research_selected_symbols",
            identity_inputs={
                "rank_id": rank["artifact_id"],
                "signal_date": payload["signal_date"],
                "strategy_id": payload["strategy_id"],
            },
        ),
        created_at=created_at,
        fields={
            "ordered_symbols": symbols,
            "rank_ref": artifact_ref(rank),
            "signal_date": payload["signal_date"],
            "strategy_id": payload["strategy_id"],
            "symbol_count": len(symbols),
            "symbol_set_sha256": symbol_set_sha256,
        },
    )


def _pool_documents(
    *,
    rank: Mapping[str, Any],
    policy: Mapping[str, Any],
    policy_path: str,
    policy_sha256: str,
) -> dict[str, dict[str, Any]]:
    rank_artifact = validate_factor_research_rank(rank, policy=policy)
    policy_artifact = validate_daily_research_policy(policy)
    if (
        policy_path not in {PHASE_A_POLICY_RELATIVE_PATH, THEME_POLICY_V2_RELATIVE_PATH}
        or hashlib.sha256(canonical_json_bytes(policy_artifact)).hexdigest() != policy_sha256
        or policy_artifact != approved_pool_policy(policy_path)
    ):
        raise IntelligenceError("research pool policy path or byte SHA is invalid")
    rank_payload = rank_artifact["payload"]
    if (
        policy_artifact["payload"]["technology_policy_state"] not in {"ACTIVE", "UNCONFIGURED"}
        or rank_payload["strategy_id"] != POOL_STRATEGY_ID
        or len(rank_payload["pool_rows"]) != 100
        or rank_payload["common_symbol_count"] < 3000
    ):
        raise IntelligenceError("Phase A research pool input is invalid")
    created_at = rank_artifact["created_at"]
    selected = _selected_symbols(rank_artifact, created_at=created_at)
    selected_raw = canonical_json_bytes(selected)
    rank_raw = canonical_json_bytes(rank_artifact)
    manifest = build_artifact(
        kind="daily_research_pool_manifest",
        identity_field="manifest_id",
        identity=business_identity(
            kind="daily_research_pool_manifest",
            identity_inputs={
                "factor_pointer_sha256": rank_payload["factor_pointer_sha256"],
                "policy_byte_sha256": policy_sha256,
                "signal_date": rank_payload["signal_date"],
                "strategy_id": POOL_STRATEGY_ID,
            },
        ),
        created_at=created_at,
        fields={
            "common_symbol_count": rank_payload["common_symbol_count"],
            "common_symbol_set_sha256": rank_payload["common_symbol_set_sha256"],
            "expected_leaf_names": list(POOL_LEAF_NAMES),
            "factor_admission_route": "BOOTSTRAP_EXCEPTION",
            "factor_generation_ref": rank_payload["factor_generation_ref"],
            "factor_pointer_sha256": rank_payload["factor_pointer_sha256"],
            "factor_research_pool": True,
            "observation_refs": rank_payload["observation_refs"],
            "policy_byte_sha256": sha256(policy_sha256, label="policy byte SHA"),
            "policy_path": policy_path,
            "policy_ref": artifact_ref(policy_artifact),
            "pool_size": len(rank_payload["pool_rows"]),
            "prospective_admission_state": "NOT_CLAIMED",
            "rank_byte_sha256": hashlib.sha256(rank_raw).hexdigest(),
            "rank_ref": artifact_ref(rank_artifact),
            "selected_symbols_byte_sha256": hashlib.sha256(selected_raw).hexdigest(),
            "selected_symbols_ref": artifact_ref(selected),
            "selected_symbol_set_sha256": selected["payload"]["symbol_set_sha256"],
            "signal_date": rank_payload["signal_date"],
            "strategy_id": POOL_STRATEGY_ID,
            "technology_gate": (
                "UNAVAILABLE"
                if policy_artifact["payload"]["technology_policy_state"] == "UNCONFIGURED"
                else "PENDING_SOURCE_REPLAY"
            ),
            "technology_shortlist": False,
            "theme_gate_executed": False,
        },
    )
    manifest_raw = canonical_json_bytes(manifest)
    root_relative = (
        f"{POOL_ROOT_RELATIVE_PATH}/{POOL_STRATEGY_ID}/"
        f"{rank_payload['signal_date'][0:4]}-{rank_payload['signal_date'][4:6]}-"
        f"{rank_payload['signal_date'][6:8]}"
    )
    receipt = build_artifact(
        kind="daily_research_pool_receipt",
        identity_field="receipt_id",
        identity=business_identity(
            kind="daily_research_pool_receipt",
            identity_inputs={"manifest_id": manifest["artifact_id"]},
        ),
        created_at=created_at,
        fields={
            "manifest_byte_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_ref": artifact_ref(manifest),
            "root_relative_path": root_relative,
            "signal_date": rank_payload["signal_date"],
            "state": "COMPLETE",
            "strategy_id": POOL_STRATEGY_ID,
        },
    )
    return {
        "factor_research_rank.json": rank_artifact,
        "manifest.json": manifest,
        "publish_receipt.json": receipt,
        "selected_symbols.json": selected,
    }


def _verify_file(path: Path, expected: bytes) -> None:
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or read_stable_regular_file(path, label=f"research pool {path.name}") != expected
    ):
        raise IntelligenceError("research pool leaf validation failed")


def _validate_root(path: Path, raw_documents: Mapping[str, bytes]) -> None:
    _verify_owned_directory(path, exact_mode=True)
    leaves = sorted(entry.name for entry in path.iterdir())
    if leaves != list(POOL_LEAF_NAMES):
        raise IntelligenceError("research pool leaf inventory differs")
    for name in POOL_LEAF_NAMES:
        _verify_file(path / name, raw_documents[name])


def _atomic_no_replace(source: Path, destination: Path) -> None:
    source_raw = source.name.encode("ascii", errors="strict")
    destination_raw = destination.name.encode("ascii", errors="strict")
    parent_fd = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            operation = getattr(libc, "renameatx_np", None)
            flags = _DARWIN_RENAME_EXCL
        elif sys.platform.startswith("linux"):
            operation = getattr(libc, "renameat2", None)
            flags = _LINUX_RENAME_NOREPLACE
        else:
            operation = None
            flags = 0
        if operation is None:
            raise IntelligenceError("atomic no-replace directory rename is unavailable")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        ctypes.set_errno(0)
        if operation(parent_fd, source_raw, parent_fd, destination_raw, flags) == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise IntelligenceError("atomic research pool publication failed")
    finally:
        os.close(parent_fd)


class DailyResearchPoolStore:
    """Exact-once strategy-scoped immutable Factor research-pool store."""

    def __init__(self, workspace_root: str | os.PathLike[str]) -> None:
        self.workspace_root = _workspace(workspace_root)

    def publish(
        self,
        *,
        rank: Mapping[str, Any],
        expected_policy_sha256: str,
        before_publish: Callable[[], None],
        policy_path: str = PHASE_A_POLICY_RELATIVE_PATH,
    ) -> dict[str, Any]:
        approved = approved_pool_policy(policy_path)
        policy_file = self.workspace_root / policy_path
        policy_raw = canonical_json_bytes(approved)
        _verify_policy_file(policy_file, policy_raw)
        if hashlib.sha256(policy_raw).hexdigest() != expected_policy_sha256:
            raise IntelligenceError("published research policy SHA differs")
        policy = validate_daily_research_policy(policy_raw)
        if rank["payload"]["signal_date"] < policy["payload"]["effective_signal_date"]:
            raise IntelligenceError("Factor signal date predates daily research policy")
        documents = _pool_documents(
            rank=rank,
            policy=policy,
            policy_path=policy_path,
            policy_sha256=expected_policy_sha256,
        )
        raw_documents = {
            name: canonical_json_bytes(document) for name, document in documents.items()
        }
        signal_date = documents["manifest.json"]["payload"]["signal_date"]
        parent = _store_parent(
            self.workspace_root,
            ("intelligence", "research_pool", POOL_STRATEGY_ID),
        )
        target = parent / (f"{signal_date[0:4]}-{signal_date[4:6]}-{signal_date[6:8]}")
        if target.exists():
            _validate_root(target, raw_documents)
            status = "NO_ACTION"
        else:
            staging = parent / (f".{target.name}.staging-{os.getpid()}-{secrets.token_hex(8)}")
            staging.mkdir(mode=0o700)
            for name in POOL_LEAF_NAMES:
                try:
                    descriptor = os.open(
                        staging / name,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                    )
                    try:
                        os.fchmod(descriptor, 0o600)
                        _write_all(descriptor, raw_documents[name])
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                except OSError as exc:
                    raise IntelligenceError("research pool staging write failed") from exc
            staging_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            _validate_root(staging, raw_documents)
            before_publish()
            try:
                _atomic_no_replace(staging, target)
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                status = "PUBLISHED"
            except FileExistsError:
                _validate_root(target, raw_documents)
                for name in POOL_LEAF_NAMES:
                    (staging / name).unlink()
                staging.rmdir()
                status = "NO_ACTION"
            _validate_root(target, raw_documents)
        return {
            "command_status": status,
            "manifest_path": (
                f"{POOL_ROOT_RELATIVE_PATH}/{POOL_STRATEGY_ID}/{target.name}/manifest.json"
            ),
            "manifest_sha256": hashlib.sha256(raw_documents["manifest.json"]).hexdigest(),
            "pool_root": f"{POOL_ROOT_RELATIVE_PATH}/{POOL_STRATEGY_ID}/{target.name}",
            "receipt_path": (
                f"{POOL_ROOT_RELATIVE_PATH}/{POOL_STRATEGY_ID}/{target.name}/"
                "publish_receipt.json"
            ),
            "receipt_sha256": hashlib.sha256(raw_documents["publish_receipt.json"]).hexdigest(),
            "signal_date": signal_date,
            "strategy_id": POOL_STRATEGY_ID,
            "research_only": True,
            "production": False,
            "grants_trading_authority": False,
        }


__all__ = [
    "DailyResearchPoolStore",
    "PHASE_A_POLICY_RELATIVE_PATH",
    "THEME_GOVERNANCE_POLICY_RELATIVE_PATH",
    "THEME_POLICY_V2_RELATIVE_PATH",
    "POOL_ROOT_RELATIVE_PATH",
    "POOL_STRATEGY_ID",
    "approved_pool_policy",
    "approved_phase_a_policy",
    "approved_theme_policy_v2",
    "publish_phase_a_policy",
    "publish_theme_policy_v2",
]
