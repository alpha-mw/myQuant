"""Strict Parquet-backed market data reader for runtime strategy paths."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_investor.agent_protocol import DataQualityIssue
from quant_investor.market.cn_nontrading_evidence import (
    validate_bak_daily_nontrading_evidence,
)
from quant_investor.market.cn_terminal_delisting_evidence import (
    validate_terminal_delisting_evidence,
)
from quant_investor.market.read_result import MarketDataReadResult


class MarketDataUnavailableError(RuntimeError):
    """Raised when strict Parquet market data is not healthy enough to read."""


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _normalize_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_symbol_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol and symbol not in result:
            result.append(symbol)
    return result


def coverage_fingerprint(value: Any) -> str:
    """Return a deterministic digest for a JSON coverage object."""

    coverage = value if isinstance(value, dict) else {}
    encoded = json.dumps(
        coverage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Compatibility alias for existing internal callers.
_coverage_fingerprint = coverage_fingerprint


def _coverage_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(number):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _complete_coverage_blockers(
    coverage: Mapping[str, Any],
    *,
    latest_complete_trade_date: str,
) -> list[str]:
    """Validate the closed-world claims required by complete coverage."""

    if coverage.get("complete") is not True:
        return []

    blockers: list[str] = []
    coverage_trade_date = _normalize_trade_date(coverage.get("coverage_trade_date"))
    if not coverage_trade_date:
        blockers.append("coverage_trade_date_missing")
    elif coverage_trade_date != latest_complete_trade_date:
        blockers.append(
            "coverage_trade_date_mismatch:"
            f"{coverage_trade_date}!={latest_complete_trade_date}"
        )

    expected_scope_sha256 = str(
        coverage.get("expected_scope_sha256") or ""
    ).strip().lower()
    if not expected_scope_sha256:
        blockers.append("coverage_expected_scope_sha256_missing")
    elif len(expected_scope_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_scope_sha256
    ):
        blockers.append("coverage_expected_scope_sha256_invalid")

    allowed_stale_symbols = [
        str(symbol).strip().upper()
        for symbol in list(coverage.get("allowed_stale_symbols", []) or [])
        if str(symbol).strip()
    ]
    if allowed_stale_symbols:
        blockers.append("coverage_unverified_allowed_stale_symbols_not_permitted")

    blocking_incomplete_count = _coverage_integer(
        coverage.get("blocking_incomplete_count")
    )
    if blocking_incomplete_count is None:
        blockers.append("coverage_blocking_incomplete_count_missing_or_invalid")
    elif blocking_incomplete_count != 0:
        blockers.append(
            "coverage_blocking_incomplete_count_nonzero:"
            f"{blocking_incomplete_count}"
        )

    expected_scope_count = _coverage_integer(coverage.get("expected_scope_count"))
    if expected_scope_count is None or expected_scope_count <= 0:
        blockers.append("coverage_expected_scope_count_missing_or_nonpositive")

    coverage_complete_count = _coverage_integer(
        coverage.get("coverage_complete_count")
    )
    if coverage_complete_count is None:
        blockers.append("coverage_complete_count_missing_or_invalid")
    elif (
        expected_scope_count is not None
        and expected_scope_count > 0
        and coverage_complete_count != expected_scope_count
    ):
        blockers.append(
            "coverage_complete_count_mismatch:"
            f"{coverage_complete_count}!={expected_scope_count}"
        )

    try:
        coverage_ratio = float(coverage.get("coverage_ratio"))
    except (TypeError, ValueError, OverflowError):
        coverage_ratio = math.nan
    if not math.isfinite(coverage_ratio) or coverage_ratio != 1.0:
        blockers.append("coverage_ratio_not_one")

    coverage_schema_version = str(
        coverage.get("coverage_schema_version") or ""
    )
    if coverage_schema_version in {
        "cn-full-a-coverage.v2",
        "cn-full-a-coverage.v3",
    }:
        def _symbol_set(key: str) -> set[str]:
            raw = coverage.get(key, []) or []
            if not isinstance(raw, (list, tuple, set)):
                blockers.append(f"coverage_{key}_invalid")
                return set()
            normalized = {
                str(symbol).strip().upper()
                for symbol in raw
                if str(symbol).strip()
            }
            if len(normalized) != len(raw):
                blockers.append(f"coverage_{key}_contains_duplicates_or_empty")
            return normalized

        suspended = _symbol_set("suspended_symbols")
        inactive = _symbol_set("inactive_symbols")
        verified_terminal_delisting = _symbol_set(
            "verified_terminal_delisting_symbols"
        )
        verified_nontrading = (
            _symbol_set("verified_nontrading_bak_daily_zero_symbols")
            if coverage_schema_version == "cn-full-a-coverage.v3"
            else set()
        )
        allowed = _symbol_set("allowed_stale_symbols")
        non_blocking_absent = _symbol_set("non_blocking_absent_symbols")
        true_missing = _symbol_set("true_missing_symbols")
        classification_sets = [
            suspended,
            inactive,
            verified_nontrading,
            allowed,
            true_missing,
        ]
        if coverage.get("classification_sets_disjoint") is not True or any(
            left & right
            for index, left in enumerate(classification_sets)
            for right in classification_sets[index + 1 :]
        ):
            blockers.append("coverage_classification_sets_not_disjoint")
        declared_non_blocking = (
            suspended | inactive | verified_nontrading | allowed
        )
        if declared_non_blocking != non_blocking_absent:
            blockers.append("coverage_non_blocking_absent_union_mismatch")
        if true_missing:
            blockers.append(
                f"coverage_true_missing_symbols_nonempty:{len(true_missing)}"
            )
        observed_bar_count = _coverage_integer(coverage.get("observed_bar_count"))
        if observed_bar_count is None or observed_bar_count < 0:
            blockers.append("coverage_observed_bar_count_missing_or_invalid")
        elif (
            coverage_complete_count is not None
            and observed_bar_count + len(non_blocking_absent)
            != coverage_complete_count
        ):
            blockers.append("coverage_classification_union_count_mismatch")
        if coverage_schema_version == "cn-full-a-coverage.v3" and verified_nontrading:
            evidence_path = str(
                coverage.get("verified_nontrading_evidence_path") or ""
            ).strip()
            evidence_sha256 = str(
                coverage.get("verified_nontrading_evidence_sha256") or ""
            ).strip().lower()
            pit_path = str(coverage.get("pit_membership_path") or "").strip()
            pit_sha256 = str(
                coverage.get("pit_membership_sha256") or ""
            ).strip().lower()
            if not evidence_path:
                blockers.append("coverage_nontrading_evidence_path_missing")
            if len(evidence_sha256) != 64 or any(
                ch not in "0123456789abcdef" for ch in evidence_sha256
            ):
                blockers.append("coverage_nontrading_evidence_sha256_invalid")
            if not pit_path:
                blockers.append("coverage_pit_membership_path_missing")
            if len(pit_sha256) != 64 or any(
                ch not in "0123456789abcdef" for ch in pit_sha256
            ):
                blockers.append("coverage_pit_membership_sha256_invalid")
        if not verified_terminal_delisting.issubset(inactive):
            blockers.append(
                "coverage_terminal_delisting_not_subset_of_inactive"
            )
        if verified_terminal_delisting:
            evidence_path = str(
                coverage.get("verified_terminal_delisting_evidence_path")
                or ""
            ).strip()
            evidence_sha256 = str(
                coverage.get("verified_terminal_delisting_evidence_sha256")
                or ""
            ).strip().lower()
            payload_sha256 = str(
                coverage.get("verified_terminal_delisting_payload_sha256")
                or ""
            ).strip().lower()
            inferred_dates = coverage.get(
                "verified_terminal_delisting_inferred_dates", {}
            )
            if not evidence_path:
                blockers.append(
                    "coverage_terminal_delisting_evidence_path_missing"
                )
            for name, digest in (
                ("evidence", evidence_sha256),
                ("payload", payload_sha256),
            ):
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in digest
                ):
                    blockers.append(
                        f"coverage_terminal_delisting_{name}_sha256_invalid"
                    )
            if not isinstance(inferred_dates, Mapping) or set(
                _normalize_symbol_list(inferred_dates)
            ) != verified_terminal_delisting:
                blockers.append(
                    "coverage_terminal_delisting_inferred_dates_mismatch"
                )
    return blockers


@dataclass(frozen=True)
class ParquetSnapshot:
    snapshot_id: str
    latest_complete_trade_date: str
    latest_trade_date: str
    table_root: Path
    serving_root: Path
    manifest_path: Path
    latest_pointer_path: Path


class MarketDataReader:
    """Unified local market data reader.

    CN runtime paths use the strict Parquet canonical pointer plus symbol-serving
    files.  CSV remains available elsewhere for exports and legacy maintenance,
    but this reader never falls back to CSV when the Parquet snapshot is missing
    or unhealthy.
    """

    def __init__(
        self,
        *,
        market: str = "CN",
        data_root: str | Path | None = None,
        mode_policy: str = "strict",
    ) -> None:
        self.market = str(market or "").strip().upper()
        self.data_root = Path(data_root or "data")
        self.mode_policy = str(mode_policy or "strict").strip().lower() or "strict"
        self.parquet_market_root = self.data_root / "parquet" / self.market.lower()
        self.latest_pointer_path = self.parquet_market_root / "_latest.json"
        self.catalog_path = self.parquet_market_root / "_catalog.json"
        self.issues: list[DataQualityIssue] = []
        self._latest_payload: dict[str, Any] | None = None
        self._snapshot_gate_cache: dict[str, Any] | None = None
        self._serving_symbols_cache: tuple[tuple[str, str], list[str]] | None = None
        self._components_payload: dict[str, Any] | None = None
        self._catalog_payload: dict[str, Any] | None = None

    def _resolve_data_path(self, raw_path: Any, fallback: Path) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            return fallback
        path = Path(text)
        if path.is_absolute():
            return path
        candidates = [
            Path.cwd() / path,
            self.data_root / path,
        ]
        if self.data_root.name == "data":
            candidates.append(self.data_root.parent / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_catalog_table_path(
        self,
        *,
        catalog: Mapping[str, Any],
        table_meta: Mapping[str, Any],
        logical_table: str,
    ) -> Path:
        raw_path = table_meta.get("path") or table_meta.get("table_root")
        schema_version = str(catalog.get("schema_version") or "").strip()
        if schema_version != "strict-parquet-catalog.v1":
            return self._resolve_data_path(
                raw_path,
                self.parquet_market_root / logical_table,
            )

        error_prefix = f"strict catalog table path invalid for {logical_table}"
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise MarketDataUnavailableError(f"{error_prefix}: path missing")

        relative_path = Path(raw_path.strip())
        if relative_path.is_absolute():
            raise MarketDataUnavailableError(
                f"{error_prefix}: absolute path rejected"
            )
        if not relative_path.parts:
            raise MarketDataUnavailableError(f"{error_prefix}: path missing")
        if ".." in relative_path.parts:
            raise MarketDataUnavailableError(
                f"{error_prefix}: parent traversal rejected"
            )

        candidate = self.parquet_market_root / relative_path
        current = self.parquet_market_root
        for part in relative_path.parts:
            if part == ".":
                continue
            current = current / part
            if current.is_symlink():
                raise MarketDataUnavailableError(
                    f"{error_prefix}: symlink rejected"
                )

        try:
            market_root = self.parquet_market_root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MarketDataUnavailableError(
                f"{error_prefix}: path missing or unreadable"
            ) from exc
        try:
            resolved.relative_to(market_root)
        except ValueError as exc:
            raise MarketDataUnavailableError(
                f"{error_prefix}: path escape rejected"
            ) from exc
        return resolved

    @staticmethod
    def _strict_catalog_expected_hash(
        table_meta: Mapping[str, Any],
        *,
        logical_table: str,
    ) -> str:
        declared = [
            str(value).strip()
            for value in (
                table_meta.get("sha256"),
                table_meta.get("parquet_sha256"),
            )
            if value not in (None, "")
        ]
        prefix = f"strict catalog table hash invalid for {logical_table}"
        if not declared:
            raise MarketDataUnavailableError(
                f"strict catalog table hash missing for {logical_table}"
            )
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in declared
        ):
            raise MarketDataUnavailableError(prefix)
        if len(set(declared)) != 1:
            raise MarketDataUnavailableError(
                f"strict catalog table hash conflict for {logical_table}"
            )
        return declared[0]

    def _read_strict_catalog_parquet(
        self,
        path: Path,
        *,
        table_meta: Mapping[str, Any],
        logical_table: str,
    ) -> pd.DataFrame:
        expected_sha = self._strict_catalog_expected_hash(
            table_meta,
            logical_table=logical_table,
        )
        prefix = f"strict catalog table read invalid for {logical_table}"
        descriptor: int | None = None
        try:
            before = os.lstat(path)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise MarketDataUnavailableError(
                    f"{prefix}: regular file required"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            opened_signature = _file_signature(opened)
            if _file_signature(before) != opened_signature:
                raise MarketDataUnavailableError(
                    f"{prefix}: file identity changed before read"
                )
            digest = hashlib.sha256()
            size_read = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size_read += len(chunk)
                if size_read != opened.st_size:
                    raise MarketDataUnavailableError(
                        f"{prefix}: file size changed during hash"
                    )
                declared_size = table_meta.get("size_bytes")
                if declared_size not in (None, ""):
                    if isinstance(declared_size, bool):
                        raise MarketDataUnavailableError(
                            f"strict catalog table size invalid for {logical_table}"
                        )
                    try:
                        expected_size = int(declared_size)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise MarketDataUnavailableError(
                            f"strict catalog table size invalid for {logical_table}"
                        ) from exc
                    try:
                        size_is_exact_integer = (
                            float(declared_size) == float(expected_size)
                        )
                    except (TypeError, ValueError, OverflowError):
                        size_is_exact_integer = False
                    if expected_size < 0 or not size_is_exact_integer:
                        raise MarketDataUnavailableError(
                            f"strict catalog table size invalid for {logical_table}"
                        )
                    if expected_size != size_read:
                        raise MarketDataUnavailableError(
                            f"strict catalog table size mismatch for {logical_table}"
                        )
                if digest.hexdigest() != expected_sha:
                    raise MarketDataUnavailableError(
                        f"strict catalog table hash mismatch for {logical_table}"
                    )
                handle.seek(0)
                try:
                    frame = pd.read_parquet(handle)
                except Exception as exc:
                    raise MarketDataUnavailableError(
                        f"{prefix}: parquet unreadable"
                    ) from exc
            after_opened = os.fstat(descriptor)
            try:
                after_path = os.lstat(path)
            except OSError as exc:
                raise MarketDataUnavailableError(
                    f"{prefix}: path replaced during read"
                ) from exc
            if (
                _file_signature(after_opened) != opened_signature
                or _file_signature(after_path) != opened_signature
            ):
                raise MarketDataUnavailableError(
                    f"{prefix}: file changed or replaced during read"
                )
            return frame
        except MarketDataUnavailableError:
            raise
        except OSError as exc:
            raise MarketDataUnavailableError(
                f"{prefix}: file open failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _filter_catalog_table_frame(
        frame: pd.DataFrame,
        *,
        date_column: str,
        as_of: str,
        date_range: tuple[str, str] | None,
        columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        result = frame.copy()
        target = _normalize_trade_date(as_of)
        start = _normalize_trade_date(date_range[0]) if date_range else target
        end = _normalize_trade_date(date_range[1]) if date_range else target
        requires_date = bool(target or start or end)
        if requires_date and date_column not in result.columns:
            raise MarketDataUnavailableError(
                f"strict catalog date column missing: {date_column}"
            )
        if date_column in result.columns:
            normalized = result[date_column].map(_normalize_trade_date)
            result[date_column] = normalized
            mask = normalized.str.len().eq(8)
            if start:
                mask &= normalized >= start
            if end:
                mask &= normalized <= end
            result = result.loc[mask].copy()
        if "symbol" not in result.columns and "ts_code" in result.columns:
            result["symbol"] = result["ts_code"].map(_normalize_symbol)
        if "ts_code" not in result.columns and "symbol" in result.columns:
            result["ts_code"] = result["symbol"].map(_normalize_symbol)
        if columns is not None:
            wanted = [str(column) for column in columns if str(column)]
            if "symbol" in result.columns and "symbol" not in wanted:
                wanted.append("symbol")
            available = [column for column in wanted if column in result.columns]
            result = result.loc[:, available].copy()
        return result.reset_index(drop=True)

    def _load_latest_payload(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._latest_payload is not None and not refresh:
            return dict(self._latest_payload)
        if not self.latest_pointer_path.exists():
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer missing: {self.latest_pointer_path}"
            )
        try:
            payload = json.loads(self.latest_pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer unreadable: {self.latest_pointer_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MarketDataUnavailableError(
                f"strict Parquet snapshot pointer invalid: {self.latest_pointer_path}"
            )
        self._latest_payload = dict(payload)
        return dict(payload)

    def _snapshot_from_payload(self, payload: Mapping[str, Any]) -> ParquetSnapshot:
        table_root = self._resolve_data_path(
            payload.get("table_root"),
            self.parquet_market_root / "bars",
        )
        serving_root = self._resolve_data_path(
            payload.get("derived_serving_root"),
            self.data_root / "parquet_serving" / self.market.lower() / "bars",
        )
        manifest_path = self._resolve_data_path(
            payload.get("manifest_path") or payload.get("clean_manifest_path"),
            self.parquet_market_root / "_snapshots" / f"{payload.get('snapshot_id', '')}.json",
        )
        return ParquetSnapshot(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            latest_complete_trade_date=_normalize_trade_date(
                payload.get("latest_complete_trade_date") or payload.get("latest_trade_date")
            ),
            latest_trade_date=_normalize_trade_date(
                payload.get("latest_trade_date") or payload.get("latest_complete_trade_date")
            ),
            table_root=table_root,
            serving_root=serving_root,
            manifest_path=manifest_path,
            latest_pointer_path=self.latest_pointer_path,
        )

    def clean_snapshot_gate(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._snapshot_gate_cache is not None and not refresh:
            return dict(self._snapshot_gate_cache)
        try:
            payload = self._load_latest_payload(refresh=refresh)
        except MarketDataUnavailableError as exc:
            return {
                "status": "blocked",
                "healthy": False,
                "blockers": [str(exc)],
                "latest_pointer_path": str(self.latest_pointer_path),
                "mode_policy": self.mode_policy,
            }

        snapshot = self._snapshot_from_payload(payload)
        blockers: list[str] = []
        if str(payload.get("status") or "").upper() != "OK":
            blockers.append(f"latest pointer status is {payload.get('status')!r}")
        blockers.extend(str(item) for item in list(payload.get("blockers", []) or []) if str(item).strip())
        if not snapshot.snapshot_id:
            blockers.append("snapshot_id missing")
        if not snapshot.latest_complete_trade_date:
            blockers.append("latest_complete_trade_date missing")
        if not snapshot.table_root.exists():
            blockers.append(f"canonical bars table_root missing: {snapshot.table_root}")
        elif not any(snapshot.table_root.rglob("*.parquet")):
            blockers.append(f"canonical bars table_root has no parquet files: {snapshot.table_root}")
        if not snapshot.serving_root.exists():
            blockers.append(f"serving bars root missing: {snapshot.serving_root}")
        elif not any(snapshot.serving_root.glob("symbol=*/bars.parquet")):
            blockers.append(f"serving bars root has no symbol parquet files: {snapshot.serving_root}")
        if not snapshot.manifest_path.exists():
            blockers.append(f"manifest missing: {snapshot.manifest_path}")

        coverage = (
            dict(payload.get("coverage") or {})
            if isinstance(payload.get("coverage"), dict)
            else {}
        )
        coverage_provenance_blockers: list[str] = []
        snapshot_manifest: dict[str, Any] | None = None
        if snapshot.manifest_path.exists():
            try:
                raw_snapshot_manifest = json.loads(
                    snapshot.manifest_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                blockers.append(
                    f"manifest unreadable: {snapshot.manifest_path}: {exc}"
                )
            else:
                if isinstance(raw_snapshot_manifest, dict):
                    snapshot_manifest = raw_snapshot_manifest
                else:
                    blockers.append(f"manifest invalid: {snapshot.manifest_path}")

        if snapshot_manifest is not None:
            manifest_has_coverage = isinstance(snapshot_manifest.get("coverage"), dict)
            manifest_coverage = (
                dict(snapshot_manifest.get("coverage") or {})
                if manifest_has_coverage
                else {}
            )
            # Legacy snapshots may expose informational row/symbol statistics only
            # in the pointer.  Once either side publishes a bound coverage object,
            # or the pointer claims completeness, the two records must match.
            if manifest_has_coverage or coverage.get("complete") is True:
                pointer_coverage_sha256 = _coverage_fingerprint(coverage)
                manifest_coverage_sha256 = _coverage_fingerprint(manifest_coverage)
                if pointer_coverage_sha256 != manifest_coverage_sha256:
                    blockers.append(
                        "coverage_pointer_manifest_mismatch:"
                        f"{pointer_coverage_sha256}!={manifest_coverage_sha256}"
                    )
            blockers.extend(
                _complete_coverage_blockers(
                    coverage,
                    latest_complete_trade_date=snapshot.latest_complete_trade_date,
                )
            )
            if snapshot_manifest.get(
                "historical_scope_hash_backfilled"
            ) is True:
                coverage_provenance_blockers.append(
                    "coverage_scope_hash_backfilled_from_historical_target"
                )

        if str(coverage.get("coverage_schema_version") or "") == "cn-full-a-coverage.v3":
            verified_nontrading = list(
                coverage.get(
                    "verified_nontrading_bak_daily_zero_symbols", []
                )
                or []
            )
            if verified_nontrading:
                resolved_evidence_path: Path | None = None
                for path_key, sha_key, blocker_prefix in (
                    (
                        "verified_nontrading_evidence_path",
                        "verified_nontrading_evidence_sha256",
                        "coverage_nontrading_evidence",
                    ),
                    (
                        "pit_membership_path",
                        "pit_membership_sha256",
                        "coverage_pit_membership",
                    ),
                ):
                    raw_path = str(coverage.get(path_key) or "").strip()
                    expected_sha256 = str(
                        coverage.get(sha_key) or ""
                    ).strip().lower()
                    if not raw_path or not expected_sha256:
                        continue
                    resolved_path = self._resolve_data_path(
                        raw_path,
                        Path(raw_path),
                    )
                    if not resolved_path.exists():
                        blockers.append(f"{blocker_prefix}_missing:{resolved_path}")
                        continue
                    digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
                    if digest != expected_sha256:
                        blockers.append(
                            f"{blocker_prefix}_sha256_mismatch:"
                            f"{digest}!={expected_sha256}"
                        )
                    elif path_key == "verified_nontrading_evidence_path":
                        resolved_evidence_path = resolved_path
                if resolved_evidence_path is not None:
                    try:
                        evidence_payload = json.loads(
                            resolved_evidence_path.read_text(encoding="utf-8")
                        )
                    except Exception as exc:
                        blockers.append(
                            "coverage_nontrading_evidence_unreadable:"
                            f"{resolved_evidence_path}:{exc}"
                        )
                    else:
                        if not isinstance(evidence_payload, dict):
                            blockers.append(
                                "coverage_nontrading_evidence_invalid_payload"
                            )
                        else:
                            semantic_blockers = (
                                validate_bak_daily_nontrading_evidence(
                                    evidence_payload,
                                    trade_date=str(
                                        coverage.get("coverage_trade_date") or ""
                                    ),
                                    primary_missing_symbols=evidence_payload.get(
                                        "primary_missing_symbols", []
                                    )
                                    or [],
                                    pit_membership_sha256=str(
                                        coverage.get("pit_membership_sha256") or ""
                                    ),
                                )
                            )
                            blockers.extend(
                                "coverage_nontrading_evidence_semantic:"
                                f"{item}"
                                for item in semantic_blockers
                            )
                            evidence_symbols = set(
                                _normalize_symbol_list(
                                    evidence_payload.get(
                                        "verified_symbols", []
                                    )
                                    or []
                                )
                            )
                            if evidence_symbols != set(
                                _normalize_symbol_list(verified_nontrading)
                            ):
                                blockers.append(
                                    "coverage_nontrading_evidence_symbols_mismatch"
                                )

        verified_terminal_delisting = _normalize_symbol_list(
            coverage.get("verified_terminal_delisting_symbols", []) or []
        )
        if verified_terminal_delisting:
            raw_evidence_path = str(
                coverage.get("verified_terminal_delisting_evidence_path")
                or ""
            ).strip()
            raw_pit_path = str(
                coverage.get("pit_membership_path") or ""
            ).strip()
            expected_evidence_sha256 = str(
                coverage.get("verified_terminal_delisting_evidence_sha256")
                or ""
            ).strip().lower()
            expected_payload_sha256 = str(
                coverage.get("verified_terminal_delisting_payload_sha256")
                or ""
            ).strip().lower()
            expected_pit_sha256 = str(
                coverage.get("pit_membership_sha256") or ""
            ).strip().lower()
            resolved_terminal_path = (
                self._resolve_data_path(
                    raw_evidence_path,
                    Path(raw_evidence_path),
                )
                if raw_evidence_path
                else None
            )
            resolved_pit_path = (
                self._resolve_data_path(
                    raw_pit_path,
                    Path(raw_pit_path),
                )
                if raw_pit_path
                else None
            )
            if resolved_terminal_path is None:
                blockers.append(
                    "coverage_terminal_delisting_evidence_path_missing"
                )
            elif not resolved_terminal_path.exists():
                blockers.append(
                    "coverage_terminal_delisting_evidence_missing:"
                    f"{resolved_terminal_path}"
                )
            elif (
                hashlib.sha256(resolved_terminal_path.read_bytes()).hexdigest()
                != expected_evidence_sha256
            ):
                blockers.append(
                    "coverage_terminal_delisting_evidence_sha256_mismatch"
                )
            if resolved_pit_path is None:
                blockers.append(
                    "coverage_terminal_delisting_pit_path_missing"
                )
            elif not resolved_pit_path.exists():
                blockers.append(
                    f"coverage_terminal_delisting_pit_missing:{resolved_pit_path}"
                )
            elif hashlib.sha256(resolved_pit_path.read_bytes()).hexdigest() != (
                expected_pit_sha256
            ):
                blockers.append(
                    "coverage_terminal_delisting_pit_sha256_mismatch"
                )
            if (
                resolved_terminal_path is not None
                and resolved_pit_path is not None
                and resolved_terminal_path.exists()
                and resolved_pit_path.exists()
            ):
                try:
                    terminal_payload = json.loads(
                        resolved_terminal_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:
                    blockers.append(
                        "coverage_terminal_delisting_evidence_unreadable:"
                        f"{exc}"
                    )
                else:
                    if not isinstance(terminal_payload, dict):
                        blockers.append(
                            "coverage_terminal_delisting_evidence_invalid_payload"
                        )
                    else:
                        terminal_blockers = validate_terminal_delisting_evidence(
                            terminal_payload,
                            target_trade_date=str(
                                coverage.get("coverage_trade_date") or ""
                            ),
                            candidate_symbols=verified_terminal_delisting,
                            pit_membership_path=raw_pit_path,
                            pit_membership_sha256=expected_pit_sha256,
                        )
                        blockers.extend(
                            "coverage_terminal_delisting_evidence_semantic:"
                            f"{item}"
                            for item in terminal_blockers
                        )
                        if str(
                            terminal_payload.get("payload_sha256") or ""
                        ).lower() != expected_payload_sha256:
                            blockers.append(
                                "coverage_terminal_delisting_payload_sha256_mismatch"
                            )
                        if dict(
                            terminal_payload.get("inferred_delist_dates", {})
                            or {}
                        ) != dict(
                            coverage.get(
                                "verified_terminal_delisting_inferred_dates",
                                {},
                            )
                            or {}
                        ):
                            blockers.append(
                                "coverage_terminal_delisting_inferred_dates_mismatch"
                            )

        gate_payload = {
            "status": "ok" if not blockers else "blocked",
            "healthy": not blockers,
            "blockers": blockers,
            "snapshot_id": snapshot.snapshot_id,
            "latest_complete_trade_date": snapshot.latest_complete_trade_date,
            "latest_trade_date": snapshot.latest_trade_date,
            "table_root": str(snapshot.table_root),
            "serving_root": str(snapshot.serving_root),
            "manifest_path": str(snapshot.manifest_path),
            "latest_pointer_path": str(snapshot.latest_pointer_path),
            "mode_policy": self.mode_policy,
            "coverage": coverage,
            "coverage_provenance_blockers": coverage_provenance_blockers,
        }
        self._snapshot_gate_cache = dict(gate_payload)
        return dict(gate_payload)

    def _require_snapshot(self) -> ParquetSnapshot:
        gate = self.clean_snapshot_gate()
        if not gate.get("healthy"):
            blockers = "; ".join(str(item) for item in gate.get("blockers", []) if str(item).strip())
            raise MarketDataUnavailableError(blockers or "strict Parquet snapshot is not healthy")
        return self._snapshot_from_payload(self._load_latest_payload())

    def snapshot(self) -> dict[str, Any]:
        gate = self.clean_snapshot_gate()
        payload = {
            "backend": "parquet",
            "storage_layer": "canonical+serving",
            "mode_policy": self.mode_policy,
            "resolution_strategy": "strict_parquet_serving",
            "fallback_used": False,
        }
        payload.update(gate)
        return payload

    def physical_directories_for_full_a(self) -> list[Path]:
        snapshot = self._require_snapshot()
        return [snapshot.serving_root]

    def _serving_symbols(self, snapshot: ParquetSnapshot) -> list[str]:
        cache_key = (snapshot.snapshot_id, str(snapshot.serving_root))
        if self._serving_symbols_cache is not None:
            cached_key, cached_symbols = self._serving_symbols_cache
            if cached_key == cache_key:
                return list(cached_symbols)
        symbols = []
        for path in sorted(snapshot.serving_root.glob("symbol=*/bars.parquet")):
            symbol = path.parent.name.split("symbol=", 1)[-1].strip().upper()
            if symbol:
                symbols.append(symbol)
        self._serving_symbols_cache = (cache_key, list(symbols))
        return list(symbols)

    def _load_components(self) -> dict[str, Any]:
        if self._components_payload is not None:
            return dict(self._components_payload)
        candidates = [
            self.data_root / "cn_universe" / "cn_index_components.json",
            Path.cwd() / "data" / "cn_universe" / "cn_index_components.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                self._components_payload = dict(payload)
                return dict(self._components_payload)
        self._components_payload = {}
        return {}

    def list_symbols(
        self,
        universe_key: str = "full_a",
        category: str | None = None,
        as_of: str | None = None,
    ) -> list[str]:
        snapshot = self._require_snapshot()
        serving_symbols = self._serving_symbols(snapshot)
        key = str(category or universe_key or "full_a").strip().lower()
        if key in {"", "all", "full", "full_a", "all_a", "full_market"}:
            if self.market == "CN" and as_of:
                from quant_investor.config import config
                from quant_investor.market.pit_universe import PITUniverseStore

                if bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
                    pit_symbols = PITUniverseStore.from_config().listed_symbols(as_of)
                    if pit_symbols:
                        serving_set = set(serving_symbols)
                        return [symbol for symbol in pit_symbols if symbol in serving_set]
                    if bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)):
                        return []
            return serving_symbols

        components = self._load_components()
        component_symbols = [
            _normalize_symbol(symbol)
            for symbol in list(components.get(key, []) or [])
            if _normalize_symbol(symbol)
        ]
        if component_symbols:
            serving_set = set(serving_symbols)
            symbols = [symbol for symbol in _dedupe(component_symbols) if symbol in serving_set]
            if self.market == "CN" and as_of:
                from quant_investor.config import config
                from quant_investor.market.pit_universe import filter_symbols_by_pit_status, PITUniverseStore

                if bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
                    records = PITUniverseStore.from_config().records_by_symbol()
                    filtered = filter_symbols_by_pit_status(
                        symbols,
                        as_of=as_of,
                        records=records,
                        required=bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)),
                    )
                    return filtered.symbols
            return symbols
        if self.market != "CN":
            return serving_symbols
        return []

    def resolve_symbol_path(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        for_write: bool = False,
    ) -> Path | None:
        snapshot = self._require_snapshot()
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return None
        path = snapshot.serving_root / f"symbol={normalized}" / "bars.parquet"
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return path if path.exists() else None

    def _metadata(self, snapshot: ParquetSnapshot, **overrides: Any) -> dict[str, Any]:
        metadata = {
            "backend": "parquet",
            "storage_layer": "serving",
            "mode_policy": self.mode_policy,
            "snapshot_id": snapshot.snapshot_id,
            "latest_complete_trade_date": snapshot.latest_complete_trade_date,
            "latest_trade_date": snapshot.latest_trade_date,
            "fallback_used": False,
        }
        metadata.update(overrides)
        return metadata

    def _read_symbol_parquet(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return frame
        if "trade_date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        elif "date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        elif "Date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["Date"].map(_normalize_trade_date)
            frame = frame.sort_values("trade_date").reset_index(drop=True)
        if "symbol" not in frame.columns and "ts_code" in frame.columns:
            frame = frame.copy()
            frame["symbol"] = frame["ts_code"].map(_normalize_symbol)
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.copy()
            frame["ts_code"] = frame["symbol"].map(_normalize_symbol)
        return frame

    def _filter_frame(
        self,
        frame: pd.DataFrame,
        *,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        result = frame.copy()
        if not result.empty and "trade_date" in result.columns:
            normalized = result["trade_date"].map(_normalize_trade_date)
            mask = normalized.str.len().eq(8)
            start = _normalize_trade_date(start_date)
            end = _normalize_trade_date(end_date)
            if start:
                mask &= normalized >= start
            if end:
                mask &= normalized <= end
            result = result.loc[mask].copy()
            result["trade_date"] = result["trade_date"].map(_normalize_trade_date)
        if columns is not None:
            wanted = [str(column) for column in columns if str(column)]
            available = [column for column in wanted if column in result.columns]
            result = result.loc[:, available].copy()
        return result.reset_index(drop=True)

    def _missing_symbol_result(
        self,
        *,
        snapshot: ParquetSnapshot,
        symbol: str,
        universe_key: str,
        category: str | None = None,
        resolver_trace: Mapping[str, Any] | None = None,
        path: str = "",
        message: str = "symbol not resolved to an existing Parquet serving file",
        issue_type: str = "missing_file",
    ) -> MarketDataReadResult:
        normalized = _normalize_symbol(symbol)
        trace = dict(resolver_trace or self.snapshot())
        issue = DataQualityIssue(
            path=path,
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            issue_type=issue_type,
            severity="error",
            message=message,
            resolver_strategy=str(trace.get("resolution_strategy") or "strict_parquet_serving"),
            metadata={"snapshot": trace},
        )
        self.issues.append(issue)
        return MarketDataReadResult(
            path=path,
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=trace,
            issues=[issue],
            metadata=self._metadata(snapshot, resolved=False),
        )

    def peek_symbol_latest_date(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
    ) -> str:
        path = self.resolve_symbol_path(symbol, universe_key=universe_key, category=category)
        if path is None:
            return ""
        try:
            frame = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=["date"])
            except Exception:
                try:
                    frame = pd.read_parquet(path, columns=["Date"])
                except Exception:
                    return ""
        if frame.empty:
            return ""
        date_column = (
            "trade_date"
            if "trade_date" in frame.columns
            else "date"
            if "date" in frame.columns
            else "Date"
            if "Date" in frame.columns
            else ""
        )
        if not date_column:
            return ""
        return max((_normalize_trade_date(value) for value in frame[date_column]), default="")

    def read_symbol_frame(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> MarketDataReadResult:
        snapshot = self._require_snapshot()
        normalized = _normalize_symbol(symbol)
        path = self.resolve_symbol_path(
            normalized,
            universe_key=universe_key,
            category=category,
        )
        resolver_trace = self.snapshot()
        if path is None:
            return self._missing_symbol_result(
                snapshot=snapshot,
                symbol=normalized,
                universe_key=universe_key,
                category=category,
                resolver_trace=resolver_trace,
            )
        try:
            frame = self._filter_frame(
                self._read_symbol_parquet(path),
                start_date=start_date,
                end_date=end_date,
                columns=columns,
            )
        except Exception as exc:
            issue = DataQualityIssue(
                path=str(path),
                symbol=normalized,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                issue_type="read_error",
                severity="error",
                message=str(exc),
                resolver_strategy="strict_parquet_serving",
                metadata={"snapshot": resolver_trace},
            )
            self.issues.append(issue)
            return MarketDataReadResult(
                path=str(path),
                symbol=normalized,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=resolver_trace,
                issues=[issue],
                metadata=self._metadata(snapshot, resolved=True, row_count=0),
            )
        return MarketDataReadResult(
            frame=frame,
            path=str(path),
            symbol=normalized,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=resolver_trace,
            issues=[],
            metadata=self._metadata(snapshot, resolved=True, row_count=int(len(frame))),
        )

    def read_symbol_frames(
        self,
        symbols: Iterable[str],
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> dict[str, MarketDataReadResult]:
        snapshot = self._require_snapshot()
        normalized_symbols = _normalize_symbol_list(symbols)
        if not normalized_symbols:
            return {}

        resolver_trace = self.snapshot()
        resolver_trace["resolution_strategy"] = "strict_parquet_canonical_batch"
        try:
            batch_frame = self._read_dataset(
                snapshot.table_root,
                date_column="trade_date",
                date_range=(
                    start_date or "",
                    end_date or "",
                ) if start_date or end_date else None,
                columns=columns,
                symbol_filter=normalized_symbols,
                derive_symbol_column=False,
            )
        except Exception as exc:
            return {
                symbol: self._missing_symbol_result(
                    snapshot=snapshot,
                    symbol=symbol,
                    universe_key=universe_key,
                    category=category,
                    resolver_trace=resolver_trace,
                    path=str(snapshot.table_root),
                    message=f"canonical Parquet batch read failed: {exc}",
                    issue_type="read_error",
                )
                for symbol in normalized_symbols
            }

        results: dict[str, MarketDataReadResult] = {}
        symbol_column = (
            "symbol"
            if "symbol" in batch_frame.columns
            else "ts_code"
            if "ts_code" in batch_frame.columns
            else ""
        )
        frames_by_symbol: dict[str, pd.DataFrame] = {}
        if symbol_column and not batch_frame.empty:
            for raw_symbol, group in batch_frame.groupby(symbol_column, sort=False):
                normalized_group_symbol = _normalize_symbol(raw_symbol)
                if normalized_group_symbol:
                    frames_by_symbol[normalized_group_symbol] = group
        serving_symbols: set[str] | None = None
        for symbol in normalized_symbols:
            symbol_frame = frames_by_symbol.get(symbol)
            if symbol_frame is None:
                symbol_frame = pd.DataFrame()
            if symbol_frame.empty:
                if serving_symbols is None:
                    serving_symbols = set(self._serving_symbols(snapshot))
                if symbol not in serving_symbols:
                    results[symbol] = self._missing_symbol_result(
                        snapshot=snapshot,
                        symbol=symbol,
                        universe_key=universe_key,
                        category=category,
                        resolver_trace=resolver_trace,
                        path=str(snapshot.table_root),
                        message="symbol not found in canonical Parquet batch dataset",
                    )
                    continue
            if not symbol_frame.empty and "trade_date" in symbol_frame.columns:
                trade_dates = symbol_frame["trade_date"]
                if not trade_dates.is_monotonic_increasing:
                    symbol_frame = symbol_frame.sort_values("trade_date", kind="stable")
            if columns is not None:
                wanted = [str(column) for column in columns if str(column)]
                available = [column for column in wanted if column in symbol_frame.columns]
                if list(symbol_frame.columns) != available:
                    symbol_frame = symbol_frame.loc[:, available]
            symbol_frame = symbol_frame.reset_index(drop=True)
            results[symbol] = MarketDataReadResult(
                frame=symbol_frame,
                path=str(snapshot.table_root),
                symbol=symbol,
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=resolver_trace,
                issues=[],
                metadata=self._metadata(
                    snapshot,
                    storage_layer="canonical_batch",
                    resolution_strategy="strict_parquet_canonical_batch",
                    resolved=True,
                    batch_read=True,
                    row_count=int(len(symbol_frame)),
                ),
            )
        return results

    def read_path(
        self,
        path: str | Path,
        *,
        symbol: str = "",
        category: str = "",
        universe_key: str = "",
        start_date: str = "",
        end_date: str = "",
        columns: Sequence[str] | None = None,
    ) -> MarketDataReadResult:
        snapshot = self._require_snapshot()
        parquet_path = Path(path)
        if parquet_path.suffix.lower() != ".parquet":
            issue = DataQualityIssue(
                path=str(parquet_path),
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                issue_type="unsupported_runtime_format",
                severity="error",
                message="strict market data reader does not read CSV runtime paths",
                resolver_strategy="strict_parquet_serving",
                metadata={"fallback_used": False},
            )
            self.issues.append(issue)
            return MarketDataReadResult(
                path=str(parquet_path),
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=self.snapshot(),
                issues=[issue],
                metadata=self._metadata(snapshot, resolved=False),
            )
        frame = self._filter_frame(
            self._read_symbol_parquet(parquet_path),
            start_date=start_date,
            end_date=end_date,
            columns=columns,
        )
        return MarketDataReadResult(
            frame=frame,
            path=str(parquet_path),
            symbol=str(symbol or ""),
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_trace=self.snapshot(),
            issues=[],
            metadata=self._metadata(snapshot, resolved=True, row_count=int(len(frame))),
        )

    def latest_trade_date(self, universe_key: str = "full_a", category: str | None = None) -> str:
        snapshot = self._require_snapshot()
        return snapshot.latest_complete_trade_date or snapshot.latest_trade_date

    def _read_dataset(
        self,
        path: Path,
        *,
        date_column: str = "trade_date",
        as_of: str = "",
        date_range: tuple[str, str] | None = None,
        columns: Sequence[str] | None = None,
        symbol_filter: Sequence[str] | None = None,
        derive_symbol_column: bool = True,
    ) -> pd.DataFrame:
        read_columns = list(columns or [])
        if read_columns and date_column not in read_columns:
            read_columns.append(date_column)
        if read_columns and "ts_code" not in read_columns and "symbol" not in read_columns:
            read_columns.append("ts_code")
        normalized_symbols = _normalize_symbol_list(symbol_filter or [])
        if normalized_symbols and read_columns and "ts_code" not in read_columns and "symbol" not in read_columns:
            read_columns.append("ts_code")

        dataset_symbol_filter_applied = False
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            filter_expr = None
            normalized_as_of = _normalize_trade_date(as_of)
            if normalized_as_of:
                filter_expr = ds.field(date_column) == normalized_as_of
            elif date_range is not None:
                start = _normalize_trade_date(date_range[0])
                end = _normalize_trade_date(date_range[1])
                if start and end:
                    filter_expr = (ds.field(date_column) >= start) & (ds.field(date_column) <= end)
                elif start:
                    filter_expr = ds.field(date_column) >= start
                elif end:
                    filter_expr = ds.field(date_column) <= end
            if normalized_symbols:
                schema_names = set(getattr(dataset.schema, "names", []) or [])
                symbol_field = "ts_code" if "ts_code" in schema_names else "symbol" if "symbol" in schema_names else ""
                if symbol_field:
                    symbol_expr = ds.field(symbol_field).isin(normalized_symbols)
                    filter_expr = symbol_expr if filter_expr is None else filter_expr & symbol_expr
                    dataset_symbol_filter_applied = True
            table = dataset.to_table(
                columns=_dedupe(read_columns) if read_columns else None,
                filter=filter_expr,
            )
            frame = table.to_pandas()
        except Exception:
            files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
            frames = [pd.read_parquet(file) for file in files if file.exists()]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            start_date = date_range[0] if date_range else as_of
            end_date = date_range[1] if date_range else as_of
            frame = self._filter_frame(
                frame,
                start_date=start_date,
                end_date=end_date,
                columns=None,
            )

        if not frame.empty and date_column in frame.columns:
            frame = frame.copy()
            frame[date_column] = frame[date_column].map(_normalize_trade_date)
        if (
            derive_symbol_column
            and not frame.empty
            and "symbol" not in frame.columns
            and "ts_code" in frame.columns
        ):
            frame = frame.copy()
            frame["symbol"] = frame["ts_code"].map(_normalize_symbol)
        if normalized_symbols and "symbol" in frame.columns and not dataset_symbol_filter_applied:
            frame = frame[frame["symbol"].isin(set(normalized_symbols))].copy()
        if columns is not None:
            wanted = [str(column) for column in columns if str(column)]
            for derived in ("symbol",):
                if derived in frame.columns and derived not in wanted:
                    wanted.append(derived)
            available = [column for column in wanted if column in frame.columns]
            frame = frame.loc[:, available].copy()
        return frame.reset_index(drop=True)

    def read_cross_section(
        self,
        trade_date: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        snapshot = self._require_snapshot()
        frame = self._read_dataset(
            snapshot.table_root,
            date_column="trade_date",
            as_of=trade_date,
            columns=columns,
        )
        key = str(category or universe_key or "full_a").strip().lower()
        if key not in {"", "all", "full", "full_a", "all_a", "full_market"} and "symbol" in frame.columns:
            symbols = set(self.list_symbols(key))
            frame = frame[frame["symbol"].isin(symbols)].copy()
        return frame.reset_index(drop=True)

    def _load_catalog(self) -> dict[str, Any]:
        if self._catalog_payload is not None:
            return dict(self._catalog_payload)
        if not self.catalog_path.exists():
            self._catalog_payload = {}
            return {}
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except Exception:
            self._catalog_payload = {}
            return {}
        self._catalog_payload = dict(payload) if isinstance(payload, dict) else {}
        return dict(self._catalog_payload)

    def read_table(
        self,
        logical_table: str,
        *,
        as_of: str = "",
        date_range: tuple[str, str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._require_snapshot()
        key = str(logical_table or "").strip()
        catalog = self._load_catalog()
        tables = dict(catalog.get("tables", {}) or {})
        table_meta = tables.get(key)
        if not isinstance(table_meta, dict):
            raise MarketDataUnavailableError(f"Parquet logical table not found in catalog: {key}")
        path = self._resolve_catalog_table_path(
            catalog=catalog,
            table_meta=table_meta,
            logical_table=key,
        )
        date_column = str(table_meta.get("date_column") or "trade_date")
        if str(catalog.get("schema_version") or "").strip() == (
            "strict-parquet-catalog.v1"
        ):
            frame = self._read_strict_catalog_parquet(
                path,
                table_meta=table_meta,
                logical_table=key,
            )
            return self._filter_catalog_table_frame(
                frame,
                date_column=date_column,
                as_of=as_of,
                date_range=date_range,
                columns=columns,
            )
        return self._read_dataset(
            path,
            date_column=date_column,
            as_of=as_of,
            date_range=date_range,
            columns=columns,
        )


__all__ = [
    "MarketDataReadResult",
    "MarketDataReader",
    "MarketDataUnavailableError",
    "coverage_fingerprint",
]
