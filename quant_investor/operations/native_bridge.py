"""Closed installed-release import context for fixed native script composition.

Only the installed package selects repository modules. This internal context is
not a public command, a provider permission, or a substitute for native gates.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
from importlib.util import spec_from_loader
from pathlib import Path
import stat
import sys
import tempfile
import threading
from types import MappingProxyType

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.release_install import verify_running_release_install_input, _git
from .daily_contract import ContractError, validate_ref

# Audit changes to this closed set with the native import-side-effect test.
_MODULES = MappingProxyType(
    {
        "scripts": "scripts/__init__.py",
        "check_cn_dashboard_export": "scripts/check_cn_dashboard_export.py",
        "close_cn_dashboard_official_valuation": "scripts/close_cn_dashboard_official_valuation.py",
        "cn_dashboard_common": "scripts/cn_dashboard_common.py",
        "cn_dashboard_v2": "scripts/cn_dashboard_v2.py",
        "cn_dashboard_v2_selector": "scripts/cn_dashboard_v2_selector.py",
        "cn_official_close_batch": "scripts/cn_official_close_batch.py",
        "daily_dashboard_history": "scripts/daily_dashboard_history.py",
        "export_cn_aggressive_dashboard_data": "scripts/export_cn_aggressive_dashboard_data.py",
        "scripts.cn_dashboard_common": "scripts/cn_dashboard_common.py",
        "scripts.cn_dashboard_v2_selector": "scripts/cn_dashboard_v2_selector.py",
        "scripts.cn_official_close_batch": "scripts/cn_official_close_batch.py",
        "scripts.export_cn_aggressive_dashboard_data": (
            "scripts/export_cn_aggressive_dashboard_data.py"
        ),
        "scripts.daily_completion": "scripts/daily_completion.py",
        "scripts.daily_catchup": "scripts/daily_catchup.py",
        "scripts.daily_materialization": "scripts/daily_materialization.py",
        "scripts.daily_production": "scripts/daily_production.py",
        "scripts.daily_automatic_catchup": "scripts/daily_automatic_catchup.py",
        "scripts.daily_launch_inspection": "scripts/daily_launch_inspection.py",
        "scripts.daily_bootstrap_launch": "scripts/daily_bootstrap_launch.py",
        "scripts.daily_registered_recovery": "scripts/daily_registered_recovery.py",
        "scripts.daily_source_inputs": "scripts/daily_source_inputs.py",
        "scripts.daily_store_materialization": "scripts/daily_store_materialization.py",
        "scripts.daily_ledger": "scripts/daily_ledger.py",
        "scripts.daily_ledger_replay": "scripts/daily_ledger_replay.py",
        "scripts.daily_completion_dashboard": "scripts/daily_completion_dashboard.py",
        "scripts.daily_completion_replay": "scripts/daily_completion_replay.py",
        "scripts.daily_completion_store": "scripts/daily_completion_store.py",
        "scripts.daily_dashboard_adapter": "scripts/daily_dashboard_adapter.py",
        "scripts.daily_dashboard_capture": "scripts/daily_dashboard_capture.py",
        "scripts.daily_dashboard_history": "scripts/daily_dashboard_history.py",
        "scripts.daily_dashboard_publication": "scripts/daily_dashboard_publication.py",
        "scripts.daily_dashboard_sealed": "scripts/daily_dashboard_sealed.py",
        "scripts.daily_store_adoption": "scripts/daily_store_adoption.py",
        "scripts.daily_morning_consumer": "scripts/daily_morning_consumer.py",
        "scripts.daily_morning_seal": "scripts/daily_morning_seal.py",
        "scripts.daily_morning_history": "scripts/daily_morning_history.py",
        "scripts.daily_morning_cutover": "scripts/daily_morning_cutover.py",
        "scripts.daily_native_inputs": "scripts/daily_native_inputs.py",
        "scripts.daily_native_registry": "scripts/daily_native_registry.py",
        "scripts.daily_production_store_adapter": "scripts/daily_production_store_adapter.py",
        "scripts.manage_cn_strategy_records": "scripts/manage_cn_strategy_records.py",
        "scripts.registered_daily_event_sources": "scripts/registered_daily_event_sources.py",
    }
)
_OPERATIONS = MappingProxyType(
    {
        "daily_close": ("scripts.daily_production", "dispatch_daily_request"),
        "automatic_resolution": ("scripts.daily_automatic_catchup", "inspect_automatic_resolution"),
        "daily_launch_inspection": ("scripts.daily_launch_inspection", "inspect_daily_launch"),
        "daily_source_inputs": ("scripts.daily_source_inputs", "configured_source_inputs"),
        "morning_cutover": ("scripts.daily_morning_cutover", "recommend_morning_cutover"),
        "morning": ("scripts.daily_morning_consumer", "prepare_morning_consumer"),
        "morning_seal": ("scripts.daily_morning_seal", "seal_morning_consumer"),
        "morning_receipt": ("scripts.daily_morning_seal", "read_morning_consumer_receipt"),
        "catchup": ("scripts.daily_catchup", "run_native_catchup"),
        "execute_and_seal": ("scripts.daily_completion", "run_materialized_native_input"),
        "completion_replay": ("scripts.daily_completion_replay", "replay_native_completion"),
    }
)
_IMPORT_LOCK = threading.Lock()
_ACTIVE_COMPLETION_CONTEXT: ContextVar[tuple | None] = ContextVar(
    "verified_completion_context", default=None
)


def _invoke_active_completion_replay(
    *, release_input_sha256, repository_root, final_commit, workspace, trade_date, completion_ref
):
    """Invoke one fixed read only inside the matching verified context and thread."""
    active = _ACTIVE_COMPLETION_CONTEXT.get()
    identity = (release_input_sha256, repository_root, final_commit, threading.get_ident())
    if active is None or active[:4] != identity:
        return False, None
    result = active[4]["completion_replay"](
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    if type(result) is not dict:
        raise ContractError("NATIVE_BRIDGE_COMPLETION_RESULT_INVALID")
    return True, result


def _under(path, root):
    if not isinstance(path, str) or not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


class _SourceLoader(Loader):
    def __init__(self, path: Path, raw: bytes):
        self.path, self.raw = path, raw

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Execute the exact already Git-verified bytes, not a later path reopening
        # or an ignored pycache. Native __file__-relative assets remain unchanged.
        module.__file__ = str(self.path)
        exec(compile(self.raw, str(self.path), "exec"), module.__dict__)


class _ClosedFinder(MetaPathFinder):
    def __init__(self, root: Path, sources: dict):
        self.root, self.sources, self.sealed = root, sources, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _MODULES:
            if self.sealed:
                raise ContractError("NATIVE_BRIDGE_LATE_REPOSITORY_IMPORT")
            relative = _MODULES[fullname]
            source = self.root / relative
            loader = _SourceLoader(source, self.sources[relative])
            return spec_from_loader(
                fullname, loader, origin=str(source), is_package=fullname == "scripts"
            )
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise ContractError("NATIVE_BRIDGE_MODULE_NOT_ALLOWED")
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and _under(spec.origin, self.root):
            raise ContractError("NATIVE_BRIDGE_MODULE_NOT_ALLOWED")
        return None


def _verified_sources(root: Path, commit: str) -> dict:
    sources = {}
    for relative in sorted(set(_MODULES.values())):
        path = root / relative
        if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
            raise ContractError("NATIVE_BRIDGE_SOURCE_UNSAFE")
        raw = path.read_bytes()
        if raw != _git(root, "show", f"{commit}:{relative}"):
            raise ContractError("NATIVE_BRIDGE_SOURCE_BLOB_MISMATCH")
        sources[relative] = raw
    return sources


def _audit_loaded(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if name in _MODULES:
            if origin != str(root / _MODULES[name]):
                raise ContractError("NATIVE_BRIDGE_IMPORT_ORIGIN_MISMATCH")
        elif name.startswith("scripts.") or _under(origin, root):
            raise ContractError("NATIVE_BRIDGE_MODULE_NOT_ALLOWED")


@contextmanager
def verified_native_context(
    *, release_input_raw: bytes, expected_sha256: str, repository_root: str
):
    """Yield only fixed callables after two native running-release verifications.

    Callers validate immutable business request bytes/schema before entering, then
    validate the handler's exact result before returning from this context.
    """
    validate_ref({"path": "release-input.json", "sha256": expected_sha256})
    if (
        type(release_input_raw) is not bytes
        or hashlib.sha256(release_input_raw).hexdigest() != expected_sha256
    ):
        raise ContractError("NATIVE_BRIDGE_RELEASE_SHA_MISMATCH")
    if not _IMPORT_LOCK.acquire(blocking=False):
        raise ContractError("NATIVE_BRIDGE_CONCURRENT_IMPORT_CONTEXT")
    try:
        with _import_context(release_input_raw, repository_root) as operations:
            yield operations
    finally:
        _IMPORT_LOCK.release()


def _verify_runtime(raw, root):
    result = verify_running_release_install_input(raw, repository_root=root)
    if result.get("state") != "PASS":
        raise ContractError("NATIVE_BRIDGE_RUNTIME_NOT_VERIFIED")


@contextmanager
def _import_context(raw: bytes, repository_root: str):
    _verify_runtime(raw, repository_root)
    root = Path(repository_root).resolve(strict=True)
    if any(name in _MODULES or name.startswith("scripts.") for name in sys.modules):
        raise ContractError("NATIVE_BRIDGE_PRELOADED_REPOSITORY_MODULE")
    value = parse_canonical_json_bytes(raw)
    commit = value["release_install_evidence"]["payload"]["final_commit"]
    sources = _verified_sources(root, commit)
    finder = _ClosedFinder(root, sources)
    old_path, old_meta, old_cache = (
        list(sys.path),
        list(sys.meta_path),
        dict(sys.path_importer_cache),
    )
    old_prefix, old_modules = sys.pycache_prefix, set(sys.modules)
    try:
        with tempfile.TemporaryDirectory(prefix="myquant-native-import-") as cache:
            sys.pycache_prefix = cache
            sys.path[:] = [p for p in old_path if not _under(p or str(Path.cwd()), root)]
            sys.meta_path.insert(0, finder)
            for name in _MODULES:
                importlib.import_module(name)
            _audit_loaded(root)
            if _verified_sources(root, commit) != sources:
                raise ContractError("NATIVE_BRIDGE_SOURCE_CHANGED_DURING_PRELOAD")
            _verify_runtime(raw, root)
            finder.sealed = True
            operations = MappingProxyType(
                {
                    key: getattr(sys.modules[module], function)
                    for key, (module, function) in _OPERATIONS.items()
                }
            )
            token = _ACTIVE_COMPLETION_CONTEXT.set(
                (
                    hashlib.sha256(raw).hexdigest(),
                    str(root),
                    commit,
                    threading.get_ident(),
                    operations,
                )
            )
            try:
                yield operations
                _audit_loaded(root)
            finally:
                _ACTIVE_COMPLETION_CONTEXT.reset(token)
    finally:
        for name in set(sys.modules) - old_modules:
            module = sys.modules[name]
            if (
                name in _MODULES
                or name.startswith("scripts.")
                or _under(getattr(module, "__file__", None), root)
            ):
                del sys.modules[name]
        sys.path[:] = old_path
        sys.meta_path[:] = old_meta
        sys.pycache_prefix = old_prefix
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(old_cache)
