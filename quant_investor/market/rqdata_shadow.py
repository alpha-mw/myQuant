"""Immutable, non-authorizing RQData shadow evidence contracts.

The builders in this module are pure and perform no network or filesystem I/O.
They bind an exact RQData request to raw and normalized frames, then measure a
same-sample comparison without deciding procurement or production promotion.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.market.fundamental_provider_contract import (
    canonical_json_sha256,
    frame_fingerprint,
)
from quant_investor.market.rqdata_adapter import normalize_rqdata_symbol

RQDATA_SHADOW_REQUEST_SCHEMA = "myquant-rqdata-shadow-request.v1"
RQDATA_SHADOW_MANIFEST_SCHEMA = "myquant-rqdata-shadow-manifest.v1"
RQDATA_RECONCILIATION_SCHEMA = "myquant-provider-reconciliation.v1"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DAILY_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
    "prev_close",
    "limit_up",
    "limit_down",
    "num_trades",
)
_RECONCILIATION_FIELDS = ("close", "vol", "amount")


class RQDataShadowContractError(ValueError):
    """Raised when shadow evidence is incomplete, ambiguous, or tampered."""


def _strict_date(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise RQDataShadowContractError(f"{label} must be exact YYYYMMDD") from exc
    return parsed.strftime("%Y%m%d")


def _strict_utc(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text.endswith("Z"):
        raise RQDataShadowContractError(f"{label} must be UTC with Z suffix")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise RQDataShadowContractError(f"{label} is invalid") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RQDataShadowContractError(f"{label} must be UTC")
    return text


def _artifact_hash(payload: Mapping[str, Any], *, field: str) -> str:
    projection = {key: value for key, value in payload.items() if key != field}
    return canonical_json_sha256(projection)


def _frame_with_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    named = [name for name in frame.index.names if name is not None]
    return frame.reset_index() if named else frame.copy()


def build_rqdata_shadow_request(
    *,
    order_book_ids: Sequence[str],
    start_date: Any,
    end_date: Any,
) -> dict[str, Any]:
    """Build a deterministic RQData request description with no credentials."""

    if isinstance(order_book_ids, (str, bytes)):
        raise TypeError("order_book_ids must be a sequence, not text")
    normalized_symbols: list[str] = []
    for value in order_book_ids:
        provider_symbol = str(value or "").strip().upper()
        normalize_rqdata_symbol(provider_symbol)
        normalized_symbols.append(provider_symbol)
    ordered_symbols = sorted(set(normalized_symbols))
    if not ordered_symbols:
        raise RQDataShadowContractError("order_book_ids must not be empty")
    if len(ordered_symbols) != len(normalized_symbols):
        raise RQDataShadowContractError("order_book_ids must be unique")
    normalized_start = _strict_date(start_date, label="start_date")
    normalized_end = _strict_date(end_date, label="end_date")
    if normalized_start > normalized_end:
        raise RQDataShadowContractError("start_date must not follow end_date")

    request: dict[str, Any] = {
        "schema_version": RQDATA_SHADOW_REQUEST_SCHEMA,
        "provider": "rqdata",
        "dataset": "get_price",
        "market": "cn",
        "frequency": "1d",
        "adjust_type": "none",
        "skip_suspended": False,
        "order_book_ids": ordered_symbols,
        "start_date": normalized_start,
        "end_date": normalized_end,
        "fields": list(_DAILY_FIELDS),
        "research_only": True,
        "provider_call_authorized": False,
        "canonical_write_authorized": False,
        "promotion_authorized": False,
    }
    request["request_sha256"] = _artifact_hash(request, field="request_sha256")
    return request


def validate_rqdata_shadow_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Replay an exact request and reject unknown or changed fields."""

    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    rebuilt = build_rqdata_shadow_request(
        order_book_ids=list(request.get("order_book_ids") or []),
        start_date=request.get("start_date"),
        end_date=request.get("end_date"),
    )
    if dict(request) != rebuilt:
        raise RQDataShadowContractError("RQData shadow request does not replay exactly")
    return rebuilt


def _validate_normalized_candidate(
    canonical_frame: pd.DataFrame,
    *,
    request: Mapping[str, Any],
) -> pd.DataFrame:
    required = {
        "ts_code",
        "trade_date",
        "close",
        "vol",
        "amount",
        "provider",
        "provider_dataset",
        "provider_symbol",
        "adjustment_type",
    }
    missing = sorted(required - set(canonical_frame.columns))
    if missing:
        raise RQDataShadowContractError(f"canonical frame missing columns: {missing}")
    if canonical_frame.empty:
        raise RQDataShadowContractError("canonical frame is empty")
    if canonical_frame.duplicated(subset=["ts_code", "trade_date"]).any():
        raise RQDataShadowContractError("canonical frame contains duplicate keys")
    if set(canonical_frame["provider"].astype(str)) != {"rqdata"}:
        raise RQDataShadowContractError("canonical frame provider mismatch")
    if set(canonical_frame["provider_dataset"].astype(str)) != {"get_price"}:
        raise RQDataShadowContractError("canonical frame dataset mismatch")
    if set(canonical_frame["adjustment_type"].astype(str)) != {"none"}:
        raise RQDataShadowContractError("canonical frame adjustment mismatch")
    requested = set(request["order_book_ids"])
    observed = set(canonical_frame["provider_symbol"].astype(str))
    if not observed.issubset(requested):
        raise RQDataShadowContractError("canonical frame contains unrequested symbols")
    dates = canonical_frame["trade_date"].astype(str)
    if dates.lt(str(request["start_date"])).any() or dates.gt(str(request["end_date"])).any():
        raise RQDataShadowContractError("canonical frame contains out-of-range dates")
    return canonical_frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_rqdata_shadow_manifest(
    *,
    request: Mapping[str, Any],
    raw_frame: pd.DataFrame,
    canonical_frame: pd.DataFrame,
    acquired_at_utc: Any,
    provider_client_version: str,
    code_commit: str,
    run_id: str,
) -> dict[str, Any]:
    """Seal one completed shadow capture without granting any authority."""

    validated_request = validate_rqdata_shadow_request(request)
    if not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
        raise RQDataShadowContractError("raw frame must be a non-empty DataFrame")
    canonical = _validate_normalized_candidate(canonical_frame, request=validated_request)
    if len(raw_frame) != len(canonical):
        raise RQDataShadowContractError("raw and canonical row counts differ")
    acquired = _strict_utc(acquired_at_utc, label="acquired_at_utc")
    client_version = str(provider_client_version or "").strip()
    if not client_version:
        raise RQDataShadowContractError("provider_client_version is required")
    commit = str(code_commit or "").strip().lower()
    if _COMMIT.fullmatch(commit) is None:
        raise RQDataShadowContractError("code_commit must be a full lowercase git SHA")
    normalized_run_id = str(run_id or "").strip()
    if _RUN_ID.fullmatch(normalized_run_id) is None:
        raise RQDataShadowContractError("run_id is invalid")

    keys = sorted(
        f"{symbol}@{trade_date}"
        for symbol, trade_date in canonical[["ts_code", "trade_date"]].itertuples(
            index=False, name=None
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": RQDATA_SHADOW_MANIFEST_SCHEMA,
        "provider": "rqdata",
        "dataset": "get_price",
        "role": "primary_candidate",
        "request_sha256": validated_request["request_sha256"],
        "raw_frame_sha256": frame_fingerprint(_frame_with_identity_columns(raw_frame)),
        "canonical_frame_sha256": frame_fingerprint(canonical),
        "canonical_keyset_sha256": canonical_json_sha256(keys),
        "raw_row_count": len(raw_frame),
        "canonical_row_count": len(canonical),
        "requested_symbol_count": len(validated_request["order_book_ids"]),
        "symbol_count": int(canonical["ts_code"].nunique()),
        "missing_provider_symbols": sorted(
            set(validated_request["order_book_ids"]) - set(canonical["provider_symbol"].astype(str))
        ),
        "requested_symbol_coverage_ratio": (
            canonical["provider_symbol"].nunique() / len(validated_request["order_book_ids"])
        ),
        "first_trade_date": str(canonical["trade_date"].min()),
        "last_trade_date": str(canonical["trade_date"].max()),
        "acquired_at_utc": acquired,
        "provider_client_version": client_version,
        "code_commit": commit,
        "run_id": normalized_run_id,
        "research_only": True,
        "canonical_write_authorized": False,
        "promotion_authorized": False,
    }
    manifest["manifest_sha256"] = _artifact_hash(manifest, field="manifest_sha256")
    return manifest


def validate_rqdata_shadow_manifest(
    manifest: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    raw_frame: pd.DataFrame,
    canonical_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild a manifest from its bound inputs and require exact equality."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    rebuilt = build_rqdata_shadow_manifest(
        request=request,
        raw_frame=raw_frame,
        canonical_frame=canonical_frame,
        acquired_at_utc=manifest.get("acquired_at_utc"),
        provider_client_version=str(manifest.get("provider_client_version") or ""),
        code_commit=str(manifest.get("code_commit") or ""),
        run_id=str(manifest.get("run_id") or ""),
    )
    if dict(manifest) != rebuilt:
        raise RQDataShadowContractError("RQData shadow manifest does not replay exactly")
    return rebuilt


def _finite_numeric(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise RQDataShadowContractError(f"{label} contains non-numeric values") from exc
    for value in numeric.dropna().tolist():
        if not math.isfinite(float(value)):
            raise RQDataShadowContractError(f"{label} contains non-finite values")
    return numeric


def measure_same_sample_reconciliation(
    *,
    rqdata_frame: pd.DataFrame,
    tushare_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Measure canonical overlap and value differences without setting a gate."""

    required = {"ts_code", "trade_date", *_RECONCILIATION_FIELDS}
    for label, frame in (("rqdata", rqdata_frame), ("tushare", tushare_frame)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{label}_frame must be a pandas DataFrame")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RQDataShadowContractError(f"{label} frame missing columns: {missing}")
        if frame.empty:
            raise RQDataShadowContractError(f"{label} frame is empty")
        if frame.duplicated(subset=["ts_code", "trade_date"]).any():
            raise RQDataShadowContractError(f"{label} frame contains duplicate keys")

    left = rqdata_frame[[*sorted(required)]].copy()
    right = tushare_frame[[*sorted(required)]].copy()
    for frame in (left, right):
        frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
        frame["trade_date"] = frame["trade_date"].astype(str)
        if frame["ts_code"].eq("").any() or frame["trade_date"].eq("").any():
            raise RQDataShadowContractError("reconciliation frame contains empty keys")
    rq_keys = set(zip(left["ts_code"], left["trade_date"]))
    ts_keys = set(zip(right["ts_code"], right["trade_date"]))
    common_keys = rq_keys & ts_keys
    merged = left.merge(
        right,
        on=["ts_code", "trade_date"],
        how="inner",
        suffixes=("_rqdata", "_tushare"),
        validate="one_to_one",
    )

    measurements: dict[str, Any] = {}
    for field in _RECONCILIATION_FIELDS:
        rq_values = _finite_numeric(merged[f"{field}_rqdata"], label=f"rqdata.{field}")
        ts_values = _finite_numeric(merged[f"{field}_tushare"], label=f"tushare.{field}")
        usable = rq_values.notna() & ts_values.notna()
        paired_rq = rq_values.loc[usable].astype(float)
        paired_ts = ts_values.loc[usable].astype(float)
        pearson: float | None = None
        spearman: float | None = None
        if len(paired_rq) >= 2 and paired_rq.nunique() > 1 and paired_ts.nunique() > 1:
            pearson = float(paired_rq.corr(paired_ts, method="pearson"))
            spearman = float(paired_rq.corr(paired_ts, method="spearman"))
            pearson = pearson if math.isfinite(pearson) else None
            spearman = spearman if math.isfinite(spearman) else None
        differences = (paired_rq - paired_ts).abs()
        measurements[field] = {
            "paired_count": int(usable.sum()),
            "missing_mismatch_count": int((rq_values.isna() ^ ts_values.isna()).sum()),
            "exact_match_count": int((paired_rq == paired_ts).sum()),
            "max_absolute_difference": (
                float(differences.max()) if not differences.empty else None
            ),
            "pearson": pearson,
            "spearman": spearman,
        }

    result: dict[str, Any] = {
        "schema_version": RQDATA_RECONCILIATION_SCHEMA,
        "rqdata_key_count": len(rq_keys),
        "tushare_key_count": len(ts_keys),
        "common_key_count": len(common_keys),
        "rqdata_only_key_count": len(rq_keys - ts_keys),
        "tushare_only_key_count": len(ts_keys - rq_keys),
        "common_keyset_sha256": canonical_json_sha256(
            sorted(f"{symbol}@{trade_date}" for symbol, trade_date in common_keys)
        ),
        "measurements": measurements,
        "assessment_state": "MEASURED_NOT_EVALUATED",
        "research_only": True,
        "promotion_authorized": False,
    }
    result["reconciliation_sha256"] = _artifact_hash(result, field="reconciliation_sha256")
    return result
