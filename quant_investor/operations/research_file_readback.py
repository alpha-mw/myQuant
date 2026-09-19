"""Recheck research files through the same native reader used by their producer."""

from pathlib import Path
from quant_investor.cli.unified import _daily_source_file, _DailySourceFileError
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.contracts.core import CanonicalJSONError
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.migration.errors import UnifiedCutoverError, UNPARSEABLE_JSON
from quant_investor.system.errors import SystemStorageError, SystemNotFound
from .daily_contract import ContractError, validate_ref
from .dependency_diagnostics import DependencyInputError

_SOURCE_LABELS = frozenset(
    {
        "CUTOFF_SOURCE_REF_INVALID",
        "CUTOFF_FOCUS_PIT_REF_INVALID",
        "CUTOFF_EVENT_POINTER_INVALID",
        "CUTOFF_EVENT_SOURCE_INVALID",
        "CUTOFF_ANNOUNCEMENT_REF_INVALID",
        "PORTFOLIO_SOURCE_REF_INVALID",
    }
)
_SOURCE_FAILURES = {
    "SAFE_SOURCE_MISSING": "RESEARCH_NATIVE_SOURCE_MISSING",
    "SAFE_SOURCE_SHA_MISMATCH": "RESEARCH_NATIVE_SOURCE_SHA_MISMATCH",
}


def read_research_bytes(reader, reference, *, maximum_bytes):
    ref = validate_ref(reference)
    try:
        return reader.read_workspace_file_bytes(ref["path"], maximum_bytes=maximum_bytes)
    except FileNotFoundError as exc:
        raise DependencyInputError("RESEARCH_SOURCE_MISSING") from exc
    except SystemStorageError as exc:
        if type(exc) is SystemNotFound or (
            type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError)
        ):
            raise DependencyInputError("RESEARCH_SOURCE_MISSING") from exc
        raise


def parse_research_json(raw, *, label="research source", canonical=True):
    try:
        if canonical:
            return parse_canonical_json_bytes(raw, label=label)
        return parse_json_bytes(raw, label=label, require_canonical=False)
    except CanonicalJSONError as exc:
        raise DependencyInputError("RESEARCH_SOURCE_JSON_INVALID") from exc
    except UnifiedCutoverError as exc:
        if exc.code == UNPARSEABLE_JSON:
            raise DependencyInputError("RESEARCH_SOURCE_JSON_INVALID") from exc
        raise


class ResearchFileReadback:
    def __init__(self, workspace: str):
        self.root = Path(workspace).resolve(strict=True)
        self.observed = {}

    def source_file(self, reference, *, code):
        ref = validate_ref(reference)
        try:
            path, raw, checked = _daily_source_file(self.root, ref, code=code)
        except _DailySourceFileError as exc:
            if code not in _SOURCE_LABELS:
                raise
            raise DependencyInputError(_SOURCE_FAILURES[exc.source_failure]) from exc
        prior = self.observed.get(ref["path"])
        if prior is not None and prior != (checked, raw):
            raise ContractError("RESEARCH_FILE_READBACK_CONFLICT")
        self.observed[ref["path"]] = (checked, raw)
        return path, raw, checked

    def recheck(self):
        for ref, original in self.observed.values():
            _, raw, _ = _daily_source_file(
                self.root, ref, code="RESEARCH_FILE_CHANGED_DURING_READBACK"
            )
            if raw != original:
                raise ContractError("RESEARCH_FILE_CHANGED_DURING_READBACK")
