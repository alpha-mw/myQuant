"""One private publication lock and scoped, nonserializable sealed-EOD capability."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
import os
import json
from types import MappingProxyType

from quant_investor.system.errors import SystemSecurityError
from .journal_storage import JournalStorage
from .daily_contract import ContractError
from .dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    EVIDENCE_JSON,
    EVIDENCE_JS,
    SELECTOR_JSON,
    SELECTOR_JS,
    head_bytes,
    head_js,
    canonical_json_bytes,
    digest,
    instant,
)

_ACTIVE = ContextVar("myquant_dashboard_publication", default=None)
_KEY = object()


class _DashboardLockStorage(JournalStorage):
    """Reuse descriptor-safe lock I/O, confined to one private lock filename."""

    @staticmethod
    def _path(value):
        if value != PREFIX + "/.publication.lock":
            raise SystemSecurityError("DASHBOARD_LOCK_PATH_INVALID")
        return PurePosixPath(value)

    @staticmethod
    def _governed_directory(path):
        # Existing native Dashboard directories may be 0755. Keep workspace
        # ownership/no-writable-alias checks; the lock itself must still be 0600.
        return False


class SealedPublication:
    def __init__(self, root, proof, key):
        if key is not _KEY:
            raise ContractError("DASHBOARD_PUBLICATION_CAPABILITY_INVALID")
        self.root, self.proof = root, proof
        self.active = True
        self._expected = MappingProxyType({})
        self._head = None
        self._head_ref = None

    def __reduce__(self):
        raise TypeError("Dashboard publication capabilities cannot be serialized")

    def require(self):
        if not self.active or _ACTIVE.get() is not self:
            raise ContractError("DASHBOARD_PUBLICATION_CAPABILITY_EXPIRED")
        self.proof.require()

    @property
    def expected(self):
        return MappingProxyType(self._expected)

    def select_head(self, raw):
        self.require()
        from scripts.daily_dashboard_publication import _serving_data_files

        head = head_bytes(raw)
        proof = self.proof
        if instant(head["registered_at"]) < instant(
            proof.completion["native_validation_completed_at"]
        ):
            raise ContractError("DASHBOARD_HEAD_BEFORE_EOD_SEAL")
        if head["trade_date"] != proof.day or head["completion_ref"] != proof.completion_ref:
            raise ContractError("DASHBOARD_PUBLICATION_HEAD_DIFFERS")
        if raw != proof.head_preimage and head["previous_head_sha256"] != (
            None if proof.head_preimage is None else digest(proof.head_preimage)
        ):
            raise ContractError("DASHBOARD_PUBLICATION_HEAD_PREIMAGE_DIFFERS")
        ref = {
            "path": (
                f"results/operations/daily_production/CN/{proof.day}/dashboard/"
                f"completed-heads/{digest(raw)}.json"
            ),
            "sha256": digest(raw),
        }
        self._head, self._head_ref = head, ref
        expected = {HEAD_JSON: raw, HEAD_JS: head_js(raw), **_serving_data_files(proof, ref)}
        if self._expected and dict(self._expected) != expected:
            raise ContractError("DASHBOARD_PUBLICATION_EXPECTED_BYTES_CONFLICT")
        self._expected = MappingProxyType(expected)

    def select_selector(self, commit_at):
        self.require()
        if self._head is None:
            raise ContractError("DASHBOARD_PUBLICATION_HEAD_REQUIRED")
        from scripts.daily_dashboard_publication import _selector_files, datetime, timezone

        if (
            not instant(self._head["registered_at"])
            <= instant(commit_at)
            <= datetime.now(timezone.utc)
            <= instant(json.loads(self.proof.v2)["freshness"]["valid_through"])
        ):
            raise ContractError("DASHBOARD_SELECTOR_COMMIT_TIME_INVALID")

        selectors = _selector_files(
            self.proof, self._head, canonical_json_bytes(self._head), self._expected, commit_at
        )
        if any(
            name in self._expected and self._expected[name] != raw
            for name, raw in selectors.items()
        ):
            raise ContractError("DASHBOARD_PUBLICATION_EXPECTED_BYTES_CONFLICT")
        self._expected = MappingProxyType({**self._expected, **selectors})

    def bytes_for(self, path):
        self.require()
        from scripts.cn_dashboard_v2_selector import require_exact_private_dashboard_output_path

        name = Path(path).name
        if name not in self.expected:
            raise ContractError("DASHBOARD_PUBLICATION_PATH_NOT_AUTHORIZED")
        require_exact_private_dashboard_output_path(
            project_root=self.root, path=Path(path), filename=name
        )
        return self.expected[name]

    def validate_bytes(self, path, raw):
        if self.bytes_for(path) != raw:
            raise ContractError("DASHBOARD_PUBLICATION_SEALED_BYTES_DIFFER")


@contextmanager
def _lock(root):
    # All public writer entry points use this context; nested native writes reuse it.
    with _DashboardLockStorage(root).lock(PREFIX + "/.publication.lock"):
        yield


def _head_present(root):
    if any(
        os.path.lexists(root / PREFIX / name)
        for name in (HEAD_JSON, HEAD_JS, EVIDENCE_JSON, EVIDENCE_JS)
    ):
        return True
    from scripts.daily_dashboard_publication import _optional
    from scripts.cn_dashboard_v2_selector import validate_selector, render_js

    # Only a positively validated legacy selector can retain legacy access.
    # Corrupt v3 remnants (including a mirror alone) keep the boundary closed.
    raw = _optional(root, PREFIX + "/" + SELECTOR_JSON)
    mirror = _optional(root, PREFIX + "/" + SELECTOR_JS)
    if raw is None:
        return mirror is not None
    try:
        value = json.loads(raw)
        if value.get("schema_version") == "cn_aggressive_dashboard_selector.v3":
            return True
        if validate_selector(value):
            return True
        return mirror is not None and mirror != render_js(value)
    except (ValueError, TypeError, AttributeError):
        return True


@contextmanager
def publication_scope(project_root, capability=None):
    from .dashboard_replay_sources import require_live_dashboard_publication

    require_live_dashboard_publication()
    root = Path(project_root).resolve(strict=True)
    active = _ACTIVE.get()
    if active is not None:
        if type(active) is SealedPublication:
            if capability is not active or active.root != root:
                raise ContractError("DASHBOARD_EOD_PUBLICATION_REQUIRED")
            active.require()
        elif capability is not None or active != root:
            raise ContractError("DASHBOARD_PUBLICATION_LOCK_CONTEXT_MISMATCH")
        yield capability
        return
    if capability is not None:
        raise ContractError("DASHBOARD_PUBLICATION_CAPABILITY_EXPIRED")
    with _lock(root):
        if _head_present(root):
            raise ContractError("DASHBOARD_EOD_PUBLICATION_REQUIRED")
        token = _ACTIVE.set(root)
        try:
            yield None
        finally:
            _ACTIVE.reset(token)


@contextmanager
def sealed_publication_scope(project_root, proof):
    from scripts.daily_dashboard_publication import VerifiedDashboard
    from .dashboard_replay_sources import require_live_dashboard_publication

    require_live_dashboard_publication()
    root = Path(project_root).resolve(strict=True)
    if type(proof) is not VerifiedDashboard or proof.workspace != root or _ACTIVE.get() is not None:
        raise ContractError("DASHBOARD_SEALED_PUBLICATION_CONTEXT_INVALID")
    proof.require()
    with _lock(root):
        proof.require()
        capability = SealedPublication(root, proof, _KEY)
        token = _ACTIVE.set(capability)
        try:
            yield capability
        finally:
            capability.active = False
            _ACTIVE.reset(token)
