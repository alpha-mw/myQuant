"""Read-only native holdings identity shared by close validation and Decision context."""

from collections.abc import Mapping
from decimal import Decimal
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .store import (
    StrategyRecordConflict,
    StrategyRecordStoreError,
    NEW_RECORD_MAX_FILE_BYTES,
    _canonical_relative_path,
    _read_regular,
    _record_json,
)


def load_holdings_identity(root: Path, record: Mapping[str, Any]) -> tuple[pd.DataFrame, Decimal]:
    """Owning close identity checks, using the Store's bounded stable file reader."""
    relative = _canonical_relative_path(record["ledger_path"], label="holdings ledger")
    ledger_path = root / relative
    if ledger_path.name != "ledger_after_manual_switch.parquet":
        raise StrategyRecordStoreError("daily-close requires exact Parquet ledger")
    raw, _ = _read_regular(
        ledger_path, max_bytes=NEW_RECORD_MAX_FILE_BYTES, label="committed holdings"
    )
    if hashlib.sha256(raw).hexdigest() != record["ledger_sha256"]:
        raise StrategyRecordConflict("committed holdings SHA differs")
    ledger = pd.read_parquet(io.BytesIO(raw))
    manual = _record_json(
        root,
        path_value=record["manual_manifest_path"],
        expected_sha=record["manual_manifest_sha256"],
        label="committed manual manifest",
    )
    columns = ["symbol", "shares", "avg_cost", "cost_basis"]
    if not set(columns) <= set(ledger) or ledger[columns].isna().any().any():
        raise StrategyRecordStoreError("committed holdings identity unavailable")
    numeric = ledger[["shares", "avg_cost", "cost_basis"]].to_numpy().ravel()
    cash = Decimal(str(manual["cash_after"]))
    if not all(math.isfinite(float(value)) for value in numeric) or not cash.is_finite():
        raise StrategyRecordStoreError("committed holdings identity nonfinite")
    identity = ledger[columns].copy()
    for column in ("shares", "avg_cost", "cost_basis"):
        identity[column] = identity[column].map(lambda value: Decimal(str(value)))
    return identity.sort_values("symbol").reset_index(drop=True), cash
