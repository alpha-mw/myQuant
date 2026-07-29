"""Offline preparation of current-canonical, dynamic-factor V17 v4 Shadow inputs.

This module deliberately has no provider, maintenance, formal-publication,
canary, selector, portfolio, broker, order, or trade imports.  It accepts only
explicit path/SHA pairs and validated research-control documents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final, NoReturn

import pandas as pd

from quant_investor.factors.governance_literature_incubator_v4 import (
    candidate_catalog_v4,
    evaluate_candidate_v4,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import validate_semantic_sha

SOURCE_LOCATOR_VERSION: Final = "myquant.v17.v4.research-source-locator.v2"
INITIAL_POOL_VERSION: Final = "myquant.v17.v4.research-initial-pool-output.v2"
QUANT_BRANCH_VERSION: Final = "myquant.v17.v4.research-quant-branch-output.v2"
FUNDAMENTAL_BRANCH_VERSION: Final = "myquant.v17.v4.research-fundamental-branch-output.v2"
FACTOR_SET_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set.v1"
INPUT_BUNDLE_VERSION: Final = "myquant.v17.v4.research-factor-input-bundle.v1"
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
SOURCE_ROLES: Final = (
    "v15_run_manifest",
    "manual_execution_manifest",
    "contained_ledger",
    "market_pointer",
    "market_snapshot_manifest",
    "pit_membership",
    "pit_generation_manifest",
    "fundamental_pointer",
    "fundamental_generation_manifest",
    "macro_pointer",
    "macro_generation_manifest",
    "macro_release_calendar",
    "v15_strategy_universe",
)
_JSON_ROLES: Final = frozenset(SOURCE_ROLES) - {
    "contained_ledger",
    "pit_membership",
    "v15_strategy_universe",
}
_CN_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$", re.ASCII)
_DECIMAL_QUANTUM: Final = Decimal("0.0000000000000001")
_CATALOG: Final = tuple(candidate_catalog_v4())
_CATALOG_BY_NAME: Final = {str(row["name"]): row for row in _CATALOG}
_CATALOG_SHA256: Final = hashlib.sha256(canonical_bytes(list(_CATALOG))).hexdigest()
_FUNDAMENTAL_COMPONENTS: Final = (
    "fin_roe",
    "fin_ocf_to_profit",
    "one_minus_fin_debt_to_assets",
)


class ForwardShadowPreparationError(RuntimeError):
    """Raised when explicit offline preparation cannot be proven."""

    exit_code = 2


class TrueCurrentCanonicalInputGap(ForwardShadowPreparationError):
    """A missing or stale canonical input that requires an upstream workflow."""


def _blocked(reason: str) -> NoReturn:
    raise ForwardShadowPreparationError(f"V17_V4_FORWARD_PREPARATION_BLOCKED:{reason}")


def _gap(reason: str) -> NoReturn:
    raise TrueCurrentCanonicalInputGap(
        "V17_V4_FORWARD_PREPARATION_BLOCKED:" f"TRUE_CURRENT_CANONICAL_INPUT_GAP:{reason}"
    )


@dataclass(frozen=True)
class ExactInput:
    role: str
    relative_path: str
    byte_sha256: str
    raw: bytes

    @property
    def source_ref(self) -> dict[str, str]:
        return {
            "byte_sha256": self.byte_sha256,
            "media_type": (
                "application/json" if self.role in _JSON_ROLES else "application/vnd.apache.parquet"
            ),
            "relative_path": self.relative_path,
            "role": self.role,
        }


@dataclass(frozen=True)
class ForwardSourcePreflight:
    """Validated current-canonical source set and its two local data frames."""

    source_locator: dict[str, Any]
    strategy_universe: pd.DataFrame
    contained_ledger: pd.DataFrame
    source_inputs: tuple[ExactInput, ...]


def _require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        _blocked(f"{label}_sha256")
    return value


def _require_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        _blocked(f"{label}_identifier")
    return value


def _require_session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _blocked(f"{label}_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked(f"{label}_date")
    if parsed.isoformat() != value:
        _blocked(f"{label}_date")
    return value


def _require_cutoff(value: Any) -> str:
    if type(value) is not str or not value.endswith("Z"):
        _blocked("cutoff")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _blocked("cutoff")
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond != 0
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        _blocked("cutoff")
    return value


def _workspace_path(
    workspace_root: str | os.PathLike[str],
    value: str | os.PathLike[str],
) -> tuple[Path, str]:
    declared_root = Path(workspace_root)
    if not declared_root.is_absolute():
        _blocked("workspace_root_absolute")
    declared_root = declared_root.absolute()
    root = declared_root.resolve(strict=True)
    candidate = Path(value)
    if not candidate.is_absolute():
        if any(part in {"", ".", ".."} for part in candidate.parts):
            _blocked("source_path_noncanonical")
        candidate = root.joinpath(candidate)
        relative = candidate.relative_to(root)
    else:
        try:
            relative = candidate.relative_to(declared_root)
        except ValueError:
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                _blocked("source_path_outside_workspace")
        candidate = root / relative
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        _blocked("source_path_noncanonical")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            _gap("source_missing")
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            _blocked("source_parent_not_contained_directory")
    return candidate, PurePosixPath(*relative.parts).as_posix()


def _stable_exact_read(
    workspace_root: str | os.PathLike[str],
    *,
    role: str,
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> ExactInput:
    expected = _require_sha(expected_sha256, label=role)
    candidate, relative = _workspace_path(workspace_root, path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(candidate, flags)
    except FileNotFoundError:
        _gap(f"{role}_missing")
    except OSError as exc:
        raise ForwardShadowPreparationError(
            f"V17_V4_FORWARD_PREPARATION_BLOCKED:{role}_unsafe"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            _blocked(f"{role}_not_contained_regular_single_link")
        first = bytearray()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            first.extend(chunk)
        middle = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        second = bytearray()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            second.extend(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    raw = bytes(first)
    if (
        identity(before) != identity(middle)
        or identity(middle) != identity(after)
        or raw != bytes(second)
        or len(raw) != after.st_size
    ):
        _blocked(f"{role}_unstable_double_read")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        _blocked(f"{role}_sha256_mismatch")
    return ExactInput(role, relative, observed, raw)


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                _blocked(f"{label}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForwardShadowPreparationError(
            f"V17_V4_FORWARD_PREPARATION_BLOCKED:{label}_json"
        ) from exc
    if type(value) is not dict:
        _blocked(f"{label}_root")
    return value


def _parquet_frame(raw: bytes, *, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(BytesIO(raw))
    except Exception as exc:
        raise ForwardShadowPreparationError(
            f"V17_V4_FORWARD_PREPARATION_BLOCKED:{label}_parquet"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        _blocked(f"{label}_empty")
    return frame


def _normal_date(value: Any) -> str | None:
    if type(value) is not str or not value:
        return None
    text = value.strip()
    if re.fullmatch(r"[0-9]{8}", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if type(value) is dict:
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(_nested_values(child, key))
    elif type(value) is list:
        for child in value:
            found.extend(_nested_values(child, key))
    return found


def _exact_nested_date(document: Mapping[str, Any], key: str, *, label: str) -> str:
    values = {_normal_date(item) for item in _nested_values(document, key)}
    values.discard(None)
    if len(values) != 1:
        _blocked(f"{label}_{key}_ambiguous")
    return values.pop()


def _declared_ledger(
    workspace_root: str | os.PathLike[str],
    manual_source: ExactInput,
    manual: Mapping[str, Any],
) -> ExactInput:
    provenance = manual.get("ledger_provenance")
    path_values = [
        manual.get("effective_manual_ledger_path"),
        manual.get("next_ledger_path"),
        manual.get("ledger_after_manual_switch_parquet"),
        (provenance.get("declared_next_ledger_path") if type(provenance) is dict else None),
    ]
    paths = {value for value in path_values if type(value) is str and value}
    if len(paths) != 1:
        _blocked("contained_ledger_path_ambiguous")
    declared = paths.pop()
    pure = PurePosixPath(declared)
    if (
        pure.is_absolute()
        or pure.suffix != ".parquet"
        or str(pure) != declared
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _blocked("contained_ledger_not_declared_parquet")
    sha_values = [
        manual.get("next_ledger_sha256"),
        manual.get("ledger_after_manual_switch_parquet_sha256"),
        provenance.get("declared_sha256") if type(provenance) is dict else None,
        provenance.get("parquet_sha256") if type(provenance) is dict else None,
    ]
    shas = {value for value in sha_values if type(value) is str and value}
    if len(shas) != 1:
        _blocked("contained_ledger_sha256_ambiguous")
    return _stable_exact_read(
        workspace_root,
        role="contained_ledger",
        path=Path(workspace_root) / Path(manual_source.relative_path).parent / declared,
        expected_sha256=shas.pop(),
    )


def _numeric_series(frame: pd.DataFrame, names: Sequence[str], *, label: str) -> pd.Series:
    present = [name for name in names if name in frame.columns]
    if len(present) != 1:
        _blocked(f"contained_ledger_{label}_ambiguous")
    values = pd.to_numeric(frame[present[0]], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        _blocked(f"contained_ledger_{label}_invalid")
    return values.astype(float)


def _validate_ledger(
    frame: pd.DataFrame,
    manual: Mapping[str, Any],
    *,
    decision_session: str,
) -> None:
    symbol_columns = [name for name in ("symbol", "ts_code") if name in frame.columns]
    if len(symbol_columns) != 1:
        _blocked("contained_ledger_symbol_ambiguous")
    symbols = frame[symbol_columns[0]]
    if (
        not symbols.map(
            lambda value: type(value) is str and _CN_SYMBOL_RE.fullmatch(value) is not None
        ).all()
        or symbols.duplicated().any()
    ):
        _blocked("contained_ledger_symbol_unique")
    shares = _numeric_series(frame, ("shares", "quantity"), label="shares")
    prices = _numeric_series(
        frame,
        ("current_price", "last_price"),
        label="current_price",
    )
    values = _numeric_series(
        frame,
        ("current_value", "market_value"),
        label="current_value",
    )
    if shares.lt(0).any() or prices.lt(0).any() or values.lt(0).any():
        _blocked("contained_ledger_negative_financial")
    if not (shares.mul(prices).sub(values).abs() <= 0.011).all():
        _blocked("contained_ledger_position_reconciliation")
    for column in (
        "as_of",
        "as_of_date",
        "trade_date",
        "manual_entry_trade_date",
        "trailing_profit_peak_trade_date",
    ):
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            if value == "":
                continue
            normalized = _normal_date(str(value))
            if normalized is None or normalized > decision_session:
                _blocked("contained_ledger_as_of_after_session")
    try:
        market_value = Decimal(str(manual["market_value_after"]))
        cash = Decimal(str(manual["cash_after"]))
        total = Decimal(str(manual["total_value_after"]))
        capital = Decimal(str(manual["capital_cny"]))
        pnl = Decimal(str(manual["portfolio_pnl_after"]))
    except (KeyError, InvalidOperation):
        _blocked("contained_ledger_financial_declarations")
    observed_market = sum((Decimal(str(item)) for item in values), Decimal("0"))
    tolerance = Decimal("0.011")
    if (
        abs(observed_market - market_value) > tolerance
        or abs(cash + market_value - total) > tolerance
        or abs(capital + pnl - total) > tolerance
        or cash < 0
        or total <= 0
    ):
        _blocked("contained_ledger_financial_reconciliation")
    declared_count = manual.get("effective_manual_holding_count")
    if type(declared_count) is int and declared_count != int(shares.gt(0).sum()):
        _blocked("contained_ledger_holding_count")


def _relative_declaration_matches(
    declared: Any,
    source: ExactInput,
    *,
    workspace_root: str | os.PathLike[str],
) -> bool:
    if type(declared) is not str or not declared:
        return False
    try:
        _, relative = _workspace_path(workspace_root, declared)
        return relative == source.relative_path
    except ForwardShadowPreparationError:
        return False


def _validate_source_lineage(
    *,
    workspace_root: str | os.PathLike[str],
    decision_session: str,
    v15: Mapping[str, Any],
    manual: Mapping[str, Any],
    market_pointer: Mapping[str, Any],
    market_manifest: Mapping[str, Any],
    market_manifest_source: ExactInput,
    pit_manifest: Mapping[str, Any],
    pit_membership_source: ExactInput,
    pit_manifest_source: ExactInput,
    fundamental_pointer: Mapping[str, Any],
    fundamental_pointer_source: ExactInput,
    fundamental_manifest: Mapping[str, Any],
    fundamental_manifest_source: ExactInput,
    macro_pointer: Mapping[str, Any],
    macro_manifest: Mapping[str, Any],
    macro_manifest_source: ExactInput,
    macro_calendar: Mapping[str, Any],
    macro_calendar_source: ExactInput,
) -> None:
    if _exact_nested_date(v15, "analysis_trade_date", label="v15") != decision_session:
        _gap("v15_run_manifest_stale")
    if (
        market_pointer.get("status") != "OK"
        or _normal_date(market_pointer.get("latest_complete_trade_date")) != decision_session
        or market_pointer.get("blockers") not in ([], None)
    ):
        _gap("market_pointer_stale_or_blocked")
    coverage = market_pointer.get("coverage")
    if type(coverage) is not dict or coverage.get("complete") is not True:
        _gap("market_pointer_incomplete")
    if (
        market_manifest.get("status") != "OK"
        or market_manifest.get("readback_validated") is not True
        or market_manifest.get("snapshot_id") != market_pointer.get("snapshot_id")
        or _normal_date(market_manifest.get("latest_complete_trade_date")) != decision_session
        or not _relative_declaration_matches(
            market_pointer.get("manifest_path"),
            market_manifest_source,
            workspace_root=workspace_root,
        )
    ):
        _blocked("market_pointer_manifest_binding")
    market_coverage = market_manifest.get("coverage")
    if type(market_coverage) is not dict:
        _blocked("market_manifest_coverage")
    if (
        market_coverage.get("pit_membership_sha256") != pit_membership_source.byte_sha256
        or market_coverage.get("pit_generation_manifest_sha256") != pit_manifest_source.byte_sha256
        or market_coverage.get("pit_generation_id") != pit_manifest.get("generation_id")
        or pit_manifest.get("canonical_sha256") != pit_membership_source.byte_sha256
    ):
        _blocked("pit_market_binding")
    if not _relative_declaration_matches(
        market_coverage.get("pit_membership_path"),
        pit_membership_source,
        workspace_root=workspace_root,
    ) or not _relative_declaration_matches(
        market_coverage.get("pit_generation_manifest_path"),
        pit_manifest_source,
        workspace_root=workspace_root,
    ):
        _blocked("pit_path_binding")
    fundamental_declaration = fundamental_pointer.get("manifest_path")
    if type(fundamental_declaration) is str and not Path(fundamental_declaration).is_absolute():
        fundamental_declaration = (
            Path(fundamental_pointer_source.relative_path).parent / fundamental_declaration
        ).as_posix()
    if (
        fundamental_pointer.get("status") != "OK"
        or fundamental_pointer.get("generation_id") != fundamental_manifest.get("generation_id")
        or fundamental_manifest.get("status") != "OK"
        or not _relative_declaration_matches(
            fundamental_declaration,
            fundamental_manifest_source,
            workspace_root=workspace_root,
        )
    ):
        _blocked("fundamental_pointer_manifest_binding")
    metadata = fundamental_pointer.get("metadata")
    primary = fundamental_pointer.get("primary_provenance")
    if (
        type(metadata) is not dict
        or metadata.get("gate2_passed") is not True
        or type(primary) is not dict
        or not str(primary.get("status") or "").startswith("verified_")
    ):
        _gap("fundamental_generation_not_current_canonical")
    macro_entry: Mapping[str, Any] = macro_pointer
    if type(macro_pointer.get("tables")) is dict:
        candidate = macro_pointer["tables"].get("macro_daily")
        if type(candidate) is dict:
            macro_entry = candidate
    if (
        macro_entry.get("generation_id") != macro_manifest.get("generation_id")
        or macro_entry.get("generation_manifest_sha256") != macro_manifest_source.byte_sha256
        or _normal_date(macro_entry.get("latest_date")) != decision_session
        or _normal_date(macro_manifest.get("as_of")) != decision_session
        or macro_manifest.get("production_eligible") is not True
    ):
        _gap("macro_generation_stale_or_unbound")
    release = macro_manifest.get("macro_release_calendar_generation")
    if type(release) is not dict:
        _blocked("macro_release_calendar_binding")
    schema_version = macro_calendar.get("schema_version")
    if schema_version == "macro-release-calendar-pointer.v1":
        if (
            release.get("macro_release_calendar_generation_id")
            != macro_calendar.get("generation_id")
            or release.get("pointer_sha256") != macro_calendar_source.byte_sha256
            or release.get("manifest_sha256") != macro_calendar.get("manifest_sha256")
        ):
            _blocked("macro_release_calendar_pointer_binding")
    elif schema_version == "macro-release-calendar-generation.v1":
        if (
            release.get("macro_release_calendar_generation_id")
            != macro_calendar.get("generation_id")
            or release.get("manifest_sha256") != macro_calendar_source.byte_sha256
        ):
            _blocked("macro_release_calendar_manifest_binding")
    elif type(macro_calendar.get("events")) is not list:
        _blocked("macro_release_calendar_shape")


def preflight_current_canonical_sources(
    workspace_root: str | os.PathLike[str],
    *,
    locator_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    v15_run_manifest_path: str,
    v15_run_manifest_sha256: str,
    manual_execution_manifest_path: str,
    manual_execution_manifest_sha256: str,
    market_pointer_path: str,
    market_pointer_sha256: str,
    market_manifest_path: str,
    market_manifest_sha256: str,
    pit_membership_path: str,
    pit_membership_sha256: str,
    pit_manifest_path: str,
    pit_manifest_sha256: str,
    fundamental_pointer_path: str,
    fundamental_pointer_sha256: str,
    fundamental_manifest_path: str,
    fundamental_manifest_sha256: str,
    macro_pointer_path: str,
    macro_pointer_sha256: str,
    macro_manifest_path: str,
    macro_manifest_sha256: str,
    macro_release_calendar_path: str,
    macro_release_calendar_sha256: str,
    strategy_universe_path: str,
    strategy_universe_sha256: str,
) -> ForwardSourcePreflight:
    """Validate explicit current inputs without scanning, downloading, or maintaining."""

    strategy = _require_id(strategy_id, label="strategy_id")
    origin = _require_session(decision_session, label="decision_session")
    exact_cutoff = _require_cutoff(cutoff)
    source_arguments = (
        ("v15_run_manifest", v15_run_manifest_path, v15_run_manifest_sha256),
        (
            "manual_execution_manifest",
            manual_execution_manifest_path,
            manual_execution_manifest_sha256,
        ),
        ("market_pointer", market_pointer_path, market_pointer_sha256),
        ("market_snapshot_manifest", market_manifest_path, market_manifest_sha256),
        ("pit_membership", pit_membership_path, pit_membership_sha256),
        ("pit_generation_manifest", pit_manifest_path, pit_manifest_sha256),
        ("fundamental_pointer", fundamental_pointer_path, fundamental_pointer_sha256),
        (
            "fundamental_generation_manifest",
            fundamental_manifest_path,
            fundamental_manifest_sha256,
        ),
        ("macro_pointer", macro_pointer_path, macro_pointer_sha256),
        ("macro_generation_manifest", macro_manifest_path, macro_manifest_sha256),
        (
            "macro_release_calendar",
            macro_release_calendar_path,
            macro_release_calendar_sha256,
        ),
        ("v15_strategy_universe", strategy_universe_path, strategy_universe_sha256),
    )
    loaded = {
        role: _stable_exact_read(
            workspace_root,
            role=role,
            path=path,
            expected_sha256=sha,
        )
        for role, path, sha in source_arguments
    }
    v15 = _json_object(loaded["v15_run_manifest"].raw, label="v15_run_manifest")
    manual = _json_object(
        loaded["manual_execution_manifest"].raw,
        label="manual_execution_manifest",
    )
    if (
        Path(loaded["v15_run_manifest"].relative_path).parent
        != Path(loaded["manual_execution_manifest"].relative_path).parent
    ):
        _blocked("v15_manual_manifest_run_ambiguous")
    ledger_source = _declared_ledger(
        workspace_root,
        loaded["manual_execution_manifest"],
        manual,
    )
    ledger = _parquet_frame(ledger_source.raw, label="contained_ledger")
    _validate_ledger(ledger, manual, decision_session=origin)
    loaded["contained_ledger"] = ledger_source
    documents = {
        role: _json_object(loaded[role].raw, label=role) for role in _JSON_ROLES if role in loaded
    }
    _validate_source_lineage(
        workspace_root=workspace_root,
        decision_session=origin,
        v15=v15,
        manual=manual,
        market_pointer=documents["market_pointer"],
        market_manifest=documents["market_snapshot_manifest"],
        market_manifest_source=loaded["market_snapshot_manifest"],
        pit_manifest=documents["pit_generation_manifest"],
        pit_membership_source=loaded["pit_membership"],
        pit_manifest_source=loaded["pit_generation_manifest"],
        fundamental_pointer=documents["fundamental_pointer"],
        fundamental_pointer_source=loaded["fundamental_pointer"],
        fundamental_manifest=documents["fundamental_generation_manifest"],
        fundamental_manifest_source=loaded["fundamental_generation_manifest"],
        macro_pointer=documents["macro_pointer"],
        macro_manifest=documents["macro_generation_manifest"],
        macro_manifest_source=loaded["macro_generation_manifest"],
        macro_calendar=documents["macro_release_calendar"],
        macro_calendar_source=loaded["macro_release_calendar"],
    )
    universe = _parquet_frame(
        loaded["v15_strategy_universe"].raw,
        label="v15_strategy_universe",
    )
    ordered_inputs = tuple(loaded[role] for role in SOURCE_ROLES)
    locator = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "cutoff": exact_cutoff,
            "formal_activation_eligible": False,
            "locator_id": _require_id(locator_id, label="locator_id"),
            "maintenance_calls_performed": False,
            "origin": origin,
            "performance_evidence_eligible": False,
            "preflight_status": "CURRENT_CANONICAL_READY",
            "protocol_version": PROTOCOL_VERSION,
            "provider_calls_performed": False,
            "shadow_only": True,
            "source_refs": [item.source_ref for item in ordered_inputs],
            "strategy_id": strategy,
            "version": SOURCE_LOCATOR_VERSION,
        }
    )
    return ForwardSourcePreflight(locator, universe, ledger, ordered_inputs)


def classify_current_canonical_preflight(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return only READY or TRUE_CURRENT_CANONICAL_INPUT_GAP classification."""

    try:
        result = preflight_current_canonical_sources(*args, **kwargs)
    except TrueCurrentCanonicalInputGap as exc:
        return {
            "maintenance_calls_performed": False,
            "provider_calls_performed": False,
            "reason": str(exc).rsplit(":", 1)[-1],
            "status": "TRUE_CURRENT_CANONICAL_INPUT_GAP",
        }
    return {
        "maintenance_calls_performed": False,
        "provider_calls_performed": False,
        "source_locator": result.source_locator,
        "status": "CURRENT_CANONICAL_READY",
    }


def artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    """Build one common artifact reference from canonical in-memory bytes."""

    identity_fields = {
        SOURCE_LOCATOR_VERSION: "locator_id",
        INITIAL_POOL_VERSION: "output_id",
        QUANT_BRANCH_VERSION: "output_id",
        FUNDAMENTAL_BRANCH_VERSION: "output_id",
        FACTOR_SET_VERSION: "factor_set_id",
        INPUT_BUNDLE_VERSION: "bundle_id",
    }
    version = document.get("version")
    identity_field = identity_fields.get(str(version))
    if identity_field is None:
        _blocked("artifact_ref_version")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or str(path) != relative_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _blocked("artifact_ref_path")
    raw = canonical_resource_bytes(document)
    return {
        "artifact_id": _require_id(document.get(identity_field), label=identity_field),
        "artifact_version": str(version),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _validate_ref(
    document: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    expected_version: str,
) -> None:
    if document.get("version") != expected_version:
        _blocked("control_document_version")
    validate_semantic_sha(document)
    expected = artifact_ref(
        document,
        relative_path=str(reference.get("relative_path") or ""),
    )
    if dict(reference) != expected:
        _blocked("control_document_reference")


def _selected_factor_rows(
    factor_set: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if (
        factor_set.get("version") != FACTOR_SET_VERSION
        or input_bundle.get("version") != INPUT_BUNDLE_VERSION
    ):
        _blocked("research_control_version")
    for document in (factor_set, input_bundle):
        if (
            document.get("authority") != NO_AUTHORITY
            or document.get("shadow_only") is not True
            or document.get("formal_activation_eligible") is not False
            or document.get("canary_evidence_eligible") is not False
            or document.get("performance_evidence_eligible") is not False
        ):
            _blocked("research_control_authority")
    selected = factor_set.get("selected_factors")
    if type(selected) is not list or not 1 <= len(selected) <= 64:
        _blocked("selected_factor_count")
    if factor_set.get("target_cardinality") != len(selected) or len(
        {row.get("name") for row in selected if type(row) is dict}
    ) != len(selected):
        _blocked("selected_factor_cardinality")
    result: list[dict[str, Any]] = []
    required_union: set[str] = set()
    for raw_row in selected:
        if type(raw_row) is not dict:
            _blocked("selected_factor_row")
        name = raw_row.get("name")
        catalog = _CATALOG_BY_NAME.get(str(name))
        if catalog is None:
            _blocked(f"unsupported_selected_factor_{name}")
        try:
            direction = Decimal(str(raw_row.get("direction")))
            catalog_direction = Decimal(str(catalog["direction"]))
        except InvalidOperation:
            _blocked(f"factor_direction_{name}")
        exact_fields = (
            "family",
            "slot",
            "lookback",
            "definition",
            "definition_sha256",
            "implementation",
            "params",
            "required_fields",
        )
        if direction != catalog_direction or any(
            raw_row.get(field) != catalog.get(field) for field in exact_fields
        ):
            _blocked(f"factor_catalog_definition_mismatch_{name}")
        required_union.update(str(field) for field in catalog["required_fields"])
        result.append(dict(raw_row))
    if factor_set.get("candidate_catalog_sha256") != _CATALOG_SHA256:
        _blocked("candidate_catalog_sha256")
    required = input_bundle.get("required_fields")
    slices = input_bundle.get("field_slices")
    expected_required = sorted(required_union)
    if (
        required != expected_required
        or type(slices) is not list
        or [row.get("field_name") for row in slices if type(row) is dict] != expected_required
    ):
        _blocked("input_bundle_required_fields")
    return tuple(result)


def _slice_frames(
    workspace_root: str | os.PathLike[str],
    *,
    input_bundle: Mapping[str, Any],
    run_id: str,
    cutoff: str,
    decision_session: str,
) -> dict[str, pd.DataFrame]:
    expected_prefix = PurePosixPath("data/private/v17_v4_runs") / run_id / "research_factor_inputs"
    result: dict[str, pd.DataFrame] = {}
    for row in input_bundle["field_slices"]:
        field = str(row["field_name"])
        reference = row.get("slice_ref")
        if type(reference) is not dict:
            _blocked(f"field_slice_ref_{field}")
        path = PurePosixPath(str(reference.get("relative_path") or ""))
        if path.parent != expected_prefix or path.name != f"{field}.parquet":
            _blocked(f"field_slice_path_{field}")
        source = _stable_exact_read(
            workspace_root,
            role=f"field_slice_{field}",
            path=str(path),
            expected_sha256=str(reference.get("byte_sha256") or ""),
        )
        frame = _parquet_frame(source.raw, label=f"field_slice_{field}")
        expected_columns = {"symbol", "trade_date", "available_at", field}
        if set(frame.columns) != expected_columns:
            _blocked(f"field_slice_columns_{field}")
        if (
            len(frame) != row.get("row_count")
            or frame[["trade_date", "symbol"]].duplicated().any()
            or not frame["symbol"]
            .map(lambda value: type(value) is str and _CN_SYMBOL_RE.fullmatch(value) is not None)
            .all()
        ):
            _blocked(f"field_slice_shape_{field}")
        sessions = frame["trade_date"].map(_normal_date)
        if sessions.isna().any():
            _blocked(f"field_slice_session_{field}")
        availability = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        values = pd.to_numeric(frame[field], errors="coerce")
        if (
            availability.isna().any()
            or availability.gt(pd.Timestamp(cutoff)).any()
            or sessions.max() != decision_session
            or sessions.min() != row.get("first_session")
            or sessions.max() != row.get("last_session")
            or str(availability.max().isoformat().replace("+00:00", "Z")) != row.get("available_at")
            or values.map(lambda value: math.isfinite(value) if pd.notna(value) else True)
            .eq(False)
            .any()
        ):
            _blocked(f"field_slice_pit_{field}")
        normalized = pd.DataFrame(
            {
                "symbol": frame["symbol"].astype(str),
                "trade_date": sessions,
                field: values,
            }
        )
        result[field] = normalized.pivot(
            index="trade_date",
            columns="symbol",
            values=field,
        ).sort_index()
        result[field].index = pd.DatetimeIndex(
            pd.to_datetime(result[field].index, format="%Y-%m-%d"),
            name="trade_date",
        )
        result[field] = result[field].sort_index(axis=1)
    return result


def _current_universe(
    frame: pd.DataFrame,
    *,
    decision_session: str,
    cutoff: str,
) -> pd.DataFrame:
    symbol_columns = [name for name in ("symbol", "ts_code") if name in frame.columns]
    if len(symbol_columns) != 1:
        _blocked("strategy_universe_symbol")
    result = frame.copy()
    symbol_column = symbol_columns[0]
    session_columns = [name for name in ("trade_date", "decision_session") if name in result]
    if len(session_columns) > 1:
        normalized_sets = {tuple(result[name].map(_normal_date)) for name in session_columns}
        if len(normalized_sets) != 1:
            _blocked("strategy_universe_session_ambiguous")
    if session_columns:
        sessions = result[session_columns[0]].map(_normal_date)
        if sessions.isna().any() or sessions.gt(decision_session).any():
            _blocked("strategy_universe_session")
        result = result.loc[sessions.eq(decision_session)].copy()
        if result.empty:
            _gap("strategy_universe_stale")
    if "available_at" in result:
        availability = pd.to_datetime(result["available_at"], utc=True, errors="coerce")
        if availability.isna().any() or availability.gt(pd.Timestamp(cutoff)).any():
            _blocked("strategy_universe_availability")
    result["symbol"] = result[symbol_column].astype(str)
    if (
        result["symbol"].duplicated().any()
        or not result["symbol"].map(lambda value: _CN_SYMBOL_RE.fullmatch(value) is not None).all()
        or len(result) < 24
    ):
        _blocked("strategy_universe_domain")
    return result.set_index("symbol", drop=False).sort_index()


def _factor_scores(
    *,
    selected: Sequence[Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
    decision_session: str,
    pool_size: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, dict[str, str]]]:
    if type(pool_size) is not int or not 24 <= pool_size <= 500:
        _blocked("initial_pool_size")
    symbols = list(universe.index)
    factor_values: dict[str, pd.Series] = {}
    origin = pd.Timestamp(decision_session)
    for row in selected:
        name = str(row["name"])
        required = list(row["required_fields"])
        if any(field not in frames for field in required):
            _blocked(f"missing_selected_factor_input_{name}")
        indexes = sorted(set().union(*(frames[field].index for field in required)))
        columns = sorted(set().union(*(frames[field].columns for field in required)))
        inputs = {
            field: frames[field].reindex(index=indexes, columns=columns) for field in required
        }
        mask = pd.DataFrame(True, index=indexes, columns=columns)
        for field in required:
            mask &= inputs[field].notna()
        try:
            evaluated = evaluate_candidate_v4(
                name=name,
                inputs=inputs,
                pit_mask=mask.astype(bool),
            )
        except Exception as exc:
            raise ForwardShadowPreparationError(
                f"V17_V4_FORWARD_PREPARATION_BLOCKED:" f"selected_factor_evaluation_{name}"
            ) from exc
        if origin not in evaluated.index:
            _blocked(f"selected_factor_origin_{name}")
        directed = evaluated.loc[origin].reindex(symbols).mul(float(Decimal(str(row["direction"]))))
        factor_values[name] = directed.rank(
            method="average",
            pct=True,
            ascending=True,
        )
    complete = pd.DataFrame(factor_values).dropna(how="any")
    finite = complete.apply(lambda column: column.map(math.isfinite)).all(axis=1)
    complete = complete.loc[finite]
    if len(complete) < pool_size:
        _gap("selected_factor_complete_domain_below_pool_size")
    factor_names = [str(row["name"]) for row in selected]
    quantized: dict[str, dict[str, str]] = {}
    composite: dict[str, str] = {}
    for symbol, values in complete.iterrows():
        ranks = {
            name: format(Decimal(str(values[name])).quantize(_DECIMAL_QUANTUM), "f")
            for name in factor_names
        }
        quantized[symbol] = ranks
        score = (
            sum((Decimal(ranks[name]) for name in factor_names), Decimal("0"))
            / Decimal(len(factor_names))
        ).quantize(_DECIMAL_QUANTUM)
        composite[symbol] = format(score, "f")
    ordered = sorted(
        composite,
        key=lambda symbol: (-Decimal(composite[symbol]), symbol),
    )
    pool = ordered[:pool_size]
    pool_set = set(pool)
    score_rows = [
        {
            "composite_score": composite[symbol],
            "factor_ranks": [
                {"factor_name": name, "rank": quantized[symbol][name]} for name in factor_names
            ],
            "selected": symbol in pool_set,
            "symbol": symbol,
        }
        for symbol in ordered
    ]
    return pool, score_rows, quantized


def _fundamental_column(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    present = [name for name in aliases if name in frame.columns]
    if not present:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    if len(present) != 1:
        _blocked("strategy_universe_fundamental_ambiguous")
    return pd.to_numeric(frame[present[0]], errors="coerce")


def _fundamental_rows(
    universe: pd.DataFrame,
    pool: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    current = universe.reindex(pool)
    values = pd.DataFrame(
        {
            "fin_roe": _fundamental_column(current, ("fin_roe", "roe")),
            "fin_ocf_to_profit": _fundamental_column(
                current,
                ("fin_ocf_to_profit", "ocf_to_profit", "ocf"),
            ),
            "one_minus_fin_debt_to_assets": (
                1.0
                - _fundamental_column(
                    current,
                    ("fin_debt_to_assets", "debt_to_assets", "leverage"),
                )
            ),
        },
        index=current.index,
    )
    complete = values.notna().all(axis=1)
    finite = values.apply(lambda column: column.map(math.isfinite)).all(axis=1)
    complete &= finite
    ranks = values.loc[complete].rank(method="average", pct=True, ascending=True)
    rows: list[dict[str, Any]] = []
    for symbol in pool:
        if not complete.loc[symbol]:
            rows.append(
                {
                    "component_ranks": [],
                    "coverage": "0.0000000000000000",
                    "evidence_status": "UNAVAILABLE_MISSING_FUNDAMENTAL",
                    "score": "0.0000000000000000",
                    "symbol": symbol,
                }
            )
            continue
        components = [
            {
                "component": name,
                "rank": format(
                    Decimal(str(ranks.loc[symbol, name])).quantize(_DECIMAL_QUANTUM),
                    "f",
                ),
            }
            for name in _FUNDAMENTAL_COMPONENTS
        ]
        score = (
            sum((Decimal(row["rank"]) for row in components), Decimal("0")) / Decimal("3")
        ).quantize(_DECIMAL_QUANTUM)
        rows.append(
            {
                "component_ranks": components,
                "coverage": "1.0000000000000000",
                "evidence_status": "AVAILABLE_COMPLETE_CASE",
                "score": format(score, "f"),
                "symbol": symbol,
            }
        )
    return rows, int(complete.sum())


def _shadow_fields() -> dict[str, bool]:
    return {
        "canary_evidence_eligible": False,
        "formal_activation_eligible": False,
        "performance_evidence_eligible": False,
        "shadow_only": True,
    }


def build_quant_first_forward_shadow(
    *,
    run_id: str,
    factor_set: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    source_locator: Mapping[str, Any],
    source_locator_ref: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    input_bundle_ref: Mapping[str, Any],
    strategy_universe: pd.DataFrame,
    field_frames: Mapping[str, pd.DataFrame],
    initial_pool_size: int,
    initial_pool_id: str,
    quant_output_id: str,
    fundamental_output_id: str,
    initial_pool_relative_path: str,
) -> dict[str, dict[str, Any]]:
    """Build the exact-pool v2 initial, Quant, and Fundamental artifacts."""

    _require_id(run_id, label="run_id")
    _validate_ref(factor_set, factor_set_ref, expected_version=FACTOR_SET_VERSION)
    _validate_ref(input_bundle, input_bundle_ref, expected_version=INPUT_BUNDLE_VERSION)
    _validate_ref(source_locator, source_locator_ref, expected_version=SOURCE_LOCATOR_VERSION)
    source_refs = source_locator.get("source_refs")
    if (
        source_locator.get("authority") != NO_AUTHORITY
        or source_locator.get("shadow_only") is not True
        or source_locator.get("formal_activation_eligible") is not False
        or source_locator.get("canary_evidence_eligible") is not False
        or source_locator.get("performance_evidence_eligible") is not False
        or source_locator.get("provider_calls_performed") is not False
        or source_locator.get("maintenance_calls_performed") is not False
        or type(source_refs) is not list
        or [row.get("role") for row in source_refs if type(row) is dict] != list(SOURCE_ROLES)
        or any(
            type(row) is not dict or _SHA_RE.fullmatch(str(row.get("byte_sha256") or "")) is None
            for row in source_refs
        )
    ):
        _blocked("research_source_locator_contract")
    if input_bundle.get("factor_set_ref") != dict(factor_set_ref) or input_bundle.get(
        "research_source_locator_ref"
    ) != dict(source_locator_ref):
        _blocked("input_bundle_control_binding")
    strategy = str(source_locator["strategy_id"])
    cutoff = str(source_locator["cutoff"])
    origin = str(source_locator["origin"])
    if (
        factor_set.get("strategy_id") != strategy
        or input_bundle.get("strategy_id") != strategy
        or factor_set.get("cutoff") != cutoff
        or input_bundle.get("cutoff") != cutoff
        or input_bundle.get("decision_session") != origin
        or input_bundle.get("run_id") != run_id
    ):
        _blocked("research_control_context")
    selected = _selected_factor_rows(factor_set, input_bundle)
    universe = _current_universe(
        strategy_universe,
        decision_session=origin,
        cutoff=cutoff,
    )
    pool, score_rows, factor_ranks = _factor_scores(
        selected=selected,
        frames=field_frames,
        universe=universe,
        decision_session=origin,
        pool_size=initial_pool_size,
    )
    factor_names = [str(row["name"]) for row in selected]
    initial = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **_shadow_fields(),
            "cutoff": cutoff,
            "factor_set_ref": dict(factor_set_ref),
            "input_bundle_ref": dict(input_bundle_ref),
            "ordered_pool": pool,
            "origin": origin,
            "output_id": _require_id(initial_pool_id, label="initial_pool_id"),
            "pool_size": len(pool),
            "protocol_version": PROTOCOL_VERSION,
            "research_source_locator_ref": dict(source_locator_ref),
            "score_rows": score_rows,
            "scored_universe_count": len(score_rows),
            "selected_factor_names": factor_names,
            "strategy_id": strategy,
            "version": INITIAL_POOL_VERSION,
        }
    )
    initial_ref = artifact_ref(initial, relative_path=initial_pool_relative_path)
    quant = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **_shadow_fields(),
            "branch_kind": "QUANT",
            "coverage_ratio": "1.0000000000000000",
            "cutoff": cutoff,
            "factor_set_ref": dict(factor_set_ref),
            "initial_pool_ref": initial_ref,
            "input_bundle_ref": dict(input_bundle_ref),
            "origin": origin,
            "output_id": _require_id(quant_output_id, label="quant_output_id"),
            "protocol_version": PROTOCOL_VERSION,
            "score_rows": [
                {
                    "factor_ranks": [
                        {"factor_name": name, "rank": factor_ranks[symbol][name]}
                        for name in factor_names
                    ],
                    "score": next(
                        row["composite_score"] for row in score_rows if row["symbol"] == symbol
                    ),
                    "symbol": symbol,
                }
                for symbol in pool
            ],
            "selected_factor_names": factor_names,
            "strategy_id": strategy,
            "version": QUANT_BRANCH_VERSION,
        }
    )
    fundamental_rows, complete_count = _fundamental_rows(universe, pool)
    fundamental_manifest_ref = next(
        row
        for row in source_locator["source_refs"]
        if row["role"] == "fundamental_generation_manifest"
    )
    fundamental_source_ref = {
        "artifact_id": "fundamental-generation-manifest",
        "artifact_version": "cn-fundamental-generation.v1",
        "byte_sha256": fundamental_manifest_ref["byte_sha256"],
        "cutoff": cutoff,
        "relative_path": fundamental_manifest_ref["relative_path"],
        "semantic_sha256": fundamental_manifest_ref["byte_sha256"],
        "strategy_id": strategy,
    }
    fundamental = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **_shadow_fields(),
            "branch_kind": "FUNDAMENTAL",
            "complete_case_count": complete_count,
            "coverage_ratio": format(
                (Decimal(complete_count) / Decimal(len(pool))).quantize(_DECIMAL_QUANTUM),
                "f",
            ),
            "cutoff": cutoff,
            "factor_set_ref": dict(factor_set_ref),
            "fundamental_source_ref": fundamental_source_ref,
            "initial_pool_ref": initial_ref,
            "input_bundle_ref": dict(input_bundle_ref),
            "origin": origin,
            "output_id": _require_id(
                fundamental_output_id,
                label="fundamental_output_id",
            ),
            "protocol_version": PROTOCOL_VERSION,
            "score_rows": fundamental_rows,
            "strategy_id": strategy,
            "unavailable_count": len(pool) - complete_count,
            "version": FUNDAMENTAL_BRANCH_VERSION,
        }
    )
    if [row["symbol"] for row in quant["score_rows"]] != pool or [
        row["symbol"] for row in fundamental["score_rows"]
    ] != pool:
        _blocked("same_pool_branch_order")
    return {
        "fundamental_branch": fundamental,
        "initial_pool": initial,
        "quant_branch": quant,
    }


def prepare_forward_shadow(
    workspace_root: str | os.PathLike[str],
    *,
    factor_set: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    input_bundle_ref: Mapping[str, Any],
    source_locator_relative_path: str,
    run_id: str,
    initial_pool_size: int,
    initial_pool_id: str,
    quant_output_id: str,
    fundamental_output_id: str,
    initial_pool_relative_path: str,
    **source_arguments: Any,
) -> dict[str, Any]:
    """Run explicit preflight, load bundle slices, and build all prep artifacts."""

    preflight = preflight_current_canonical_sources(
        workspace_root,
        **source_arguments,
    )
    locator_ref = artifact_ref(
        preflight.source_locator,
        relative_path=source_locator_relative_path,
    )
    selected = _selected_factor_rows(factor_set, input_bundle)
    del selected
    frames = _slice_frames(
        workspace_root,
        input_bundle=input_bundle,
        run_id=run_id,
        cutoff=str(preflight.source_locator["cutoff"]),
        decision_session=str(preflight.source_locator["origin"]),
    )
    artifacts = build_quant_first_forward_shadow(
        run_id=run_id,
        factor_set=factor_set,
        input_bundle=input_bundle,
        source_locator=preflight.source_locator,
        source_locator_ref=locator_ref,
        factor_set_ref=factor_set_ref,
        input_bundle_ref=input_bundle_ref,
        strategy_universe=preflight.strategy_universe,
        field_frames=frames,
        initial_pool_size=initial_pool_size,
        initial_pool_id=initial_pool_id,
        quant_output_id=quant_output_id,
        fundamental_output_id=fundamental_output_id,
        initial_pool_relative_path=initial_pool_relative_path,
    )
    return {
        "classification": "CURRENT_CANONICAL_READY",
        "maintenance_calls_performed": False,
        "provider_calls_performed": False,
        "research_source_locator": preflight.source_locator,
        **artifacts,
    }


def build_quant_forward_v3(**scoring_inputs: Any) -> dict[str, Any]:
    """Build the additive coverage-aware Quant v3 research payload.

    The legacy v2 replay path above remains unchanged. Callers must provide
    exact PIT neutralizer inputs to the pure v3 scorer.
    """

    from .forward_scoring_v3 import score_quant_forward_v3

    return score_quant_forward_v3(**scoring_inputs)


def build_fundamental_forward_v3(
    **scoring_inputs: Any,
) -> dict[str, Any]:
    """Build the additive evidence-weighted Fundamental v3 payload."""

    from .forward_scoring_v3 import score_fundamental_forward_v3

    return score_fundamental_forward_v3(**scoring_inputs)


__all__ = [
    "FACTOR_SET_VERSION",
    "FUNDAMENTAL_BRANCH_VERSION",
    "ForwardShadowPreparationError",
    "ForwardSourcePreflight",
    "INITIAL_POOL_VERSION",
    "INPUT_BUNDLE_VERSION",
    "QUANT_BRANCH_VERSION",
    "SOURCE_LOCATOR_VERSION",
    "TrueCurrentCanonicalInputGap",
    "artifact_ref",
    "build_fundamental_forward_v3",
    "build_quant_forward_v3",
    "build_quant_first_forward_shadow",
    "classify_current_canonical_preflight",
    "preflight_current_canonical_sources",
    "prepare_forward_shadow",
]
