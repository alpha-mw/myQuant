"""Compile and run one offline current-cutoff V17 v3 model-only shadow closure.

This command never acquires provider data.  It consumes exact captured evidence
and strict canonical Parquet, builds the two staged source locators, then calls
the isolated v17 v3 runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from quant_investor.factors.price_volume import (
    compute_price_volume_factor,
    prepare_price_volume_frames,
)
from quant_investor.v17_v3_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v3_contract.identities import (
    require_opaque_id,
    require_sha256,
    require_utc_cutoff,
)
from quant_investor.v17_v3_contract.resources import load_packaged_json
from quant_investor.v17_v3_runtime.algorithms import (
    fuse_branches,
    validate_branch_output,
)
from quant_investor.v17_v3_runtime.artifacts import (
    RuntimeArtifact,
    load_typed_artifact,
    runtime_artifact,
    seal_typed_artifact,
    write_typed_exact_once,
)
from quant_investor.v17_v3_runtime.authority import (
    PROTOCOL_VERSION,
    authority_envelope,
)
from quant_investor.v17_v3_runtime.service import analyze, build_initial_pool
from quant_investor.v17_v3_runtime.storage import (
    FORMAL_RESULTS_ROOT,
    PRIVATE_RUNS_ROOT,
    PRIVATE_SOURCES_ROOT,
    SecureStore,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
FACTOR_BASELINE_MODE = "PROVISIONAL_RESEARCH"
PORTFOLIO_BASIS = "MODEL_ONLY_NO_PRIVATE_HOLDINGS"
PRESELECT_PHASE = "SHADOW_CURRENT_PRESELECT"
PORTFOLIO_PHASE = "SHADOW_CURRENT_MODEL_PORTFOLIO"
PRESELECT_FACTOR_NAMES = (
    "pv_amihud_liquidity_20d",
    "pv_momentum_120d",
    "pv_price_efficiency_60d",
)
QUANT_FACTOR_NAMES = (
    "pv_blend_volstab19x2_mom90_amihud5_w80",
    "pv_downside_volatility_15d",
    "pv_short_reversal_25d",
)
REQUIRED_MARKET_COLUMNS = (
    "ts_code",
    "trade_date",
    "adj_close",
    "close",
    "vol",
    "amount",
)
REQUIRED_MEMBERSHIP_COLUMNS = (
    "symbol",
    "name",
    "source_list_status",
    "list_date",
    "delist_date",
    "effective_from",
    "effective_to",
    "observed_at",
)
REQUIRED_FUNDAMENTAL_COLUMNS = (
    "symbol",
    "research_eligible",
    "membership_conflict",
    "membership_is_pit",
    "availability",
)
TOKEN_KEY_RE = re.compile(
    r"(?:^|_)(?:token|api[_-]?key|authorization|secret|password)(?:$|_)",
    re.IGNORECASE,
)


class CurrentShadowBuildError(RuntimeError):
    """The current source closure could not be proven."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="compile and run an offline v17 v3 current model-only shadow"
    )
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase1-run-root", required=True)
    parser.add_argument("--market-pointer", required=True)
    parser.add_argument("--expected-market-pointer-sha256", required=True)
    parser.add_argument("--factor-readiness", required=True)
    parser.add_argument("--expected-factor-readiness-sha256", required=True)
    parser.add_argument("--bak-basic-acquisition-manifest", required=True)
    parser.add_argument("--expected-bak-basic-manifest-sha256", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _absolute(path: str, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise CurrentShadowBuildError(f"{label}_path_invalid")
    return value


def _read_exact(path: Path, expected_sha256: str, *, label: str) -> bytes:
    expected = require_sha256(expected_sha256, label=f"{label} SHA-256")
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
            raise CurrentShadowBuildError(f"{label}_file_invalid")
        raw = path.read_bytes()
        after = os.lstat(path)
    except OSError as exc:
        raise CurrentShadowBuildError(f"{label}_unreadable") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CurrentShadowBuildError(f"{label}_changed_during_read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise CurrentShadowBuildError(f"{label}_sha256_mismatch")
    return raw


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = load_canonical_resource(raw, label=label)
    except ValueError:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurrentShadowBuildError(f"{label}_json_invalid") from exc
    if type(value) is not dict:
        raise CurrentShadowBuildError(f"{label}_json_invalid")
    return value


def _contains_credential_material(value: Any, *, parent_key: str = "") -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if TOKEN_KEY_RE.search(key_text) and not (
                item is None or item is False or item == "" or item == "REDACTED"
            ):
                return True
            if _contains_credential_material(item, parent_key=key_text):
                return True
    elif isinstance(value, list):
        return any(_contains_credential_material(item, parent_key=parent_key) for item in value)
    return False


def _cutoff(value: str) -> tuple[str, datetime]:
    canonical = require_utc_cutoff(value, label="cutoff")
    parsed = datetime.fromisoformat(canonical.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if parsed > now:
        raise CurrentShadowBuildError("cutoff_is_in_the_future")
    return canonical, parsed


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    raw = stream.getvalue()
    if not (raw.startswith(b"PAR1") and raw.endswith(b"PAR1")):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    return raw


def _decimal_text(value: Any, *, places: int = 12) -> str:
    number = Decimal(str(value))
    if not number.is_finite():
        raise CurrentShadowBuildError("source_dataset_nonfinite_value")
    quantum = Decimal("1").scaleb(-places)
    normalized = number.quantize(quantum)
    text = format(normalized, "f").rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _raw_reference(
    *,
    role: str,
    relative_path: PurePosixPath,
    raw: bytes,
    strategy_id: str,
    cutoff: str,
) -> dict[str, str]:
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "artifact_id": f"raw-{role.replace('_', '-')}",
        "artifact_version": "myquant.v17.v3.raw-source.v1",
        "byte_sha256": digest,
        "cutoff": cutoff,
        "relative_path": str(relative_path),
        "semantic_sha256": digest,
        "strategy_id": strategy_id,
    }


def _write_raw(
    store: SecureStore,
    *,
    run_id: str,
    role: str,
    suffix: str,
    raw: bytes,
    strategy_id: str,
    cutoff: str,
) -> dict[str, str]:
    path = PRIVATE_SOURCES_ROOT / run_id / "raw" / f"{role}.{suffix}"
    store.write_exact_once(path, raw)
    return _raw_reference(
        role=role,
        relative_path=path,
        raw=raw,
        strategy_id=strategy_id,
        cutoff=cutoff,
    )


def _write_typed(
    store: SecureStore,
    *,
    relative_path: PurePosixPath,
    payload: Mapping[str, Any],
) -> RuntimeArtifact:
    artifact = runtime_artifact(
        relative_path=relative_path,
        document=seal_typed_artifact(payload),
    )
    write_typed_exact_once(store, artifact)
    return artifact


def _manifest(
    store: SecureStore,
    *,
    run_id: str,
    name: str,
    strategy_id: str,
    cutoff: str,
    phase: str,
    sources: Sequence[Mapping[str, Any]],
    parent: RuntimeArtifact | None = None,
    raw_profile: str | None = None,
) -> RuntimeArtifact:
    document: dict[str, Any] = {
        "version": "myquant.v17.v3.source-manifest.v1",
        "protocol_version": PROTOCOL_VERSION,
        "manifest_id": f"{run_id}-{name}",
        "strategy_id": strategy_id,
        "cutoff": cutoff,
        "created_at": cutoff,
        "phase": phase,
        "closure_kind": "RAW" if parent is None else "DERIVED_CLOSURE",
        "sources": sorted(
            (dict(row) for row in sources),
            key=lambda row: str(row["role"]),
        ),
        "authority": authority_envelope(),
    }
    if parent is not None:
        document["parent_raw_manifest_ref"] = parent.reference
    if raw_profile is not None:
        document["raw_profile"] = raw_profile
    return _write_typed(
        store,
        relative_path=PRIVATE_SOURCES_ROOT / run_id / "manifests" / f"{name}.json",
        payload=document,
    )


def _locator(
    store: SecureStore,
    *,
    run_id: str,
    name: str,
    strategy_id: str,
    cutoff: str,
    manifest: RuntimeArtifact,
    preselection: RuntimeArtifact | None,
) -> RuntimeArtifact:
    return _write_typed(
        store,
        relative_path=PRIVATE_SOURCES_ROOT / run_id / "locators" / f"{name}.json",
        payload={
            "version": "myquant.v17.v3.source-locator.v1",
            "protocol_version": PROTOCOL_VERSION,
            "locator_id": f"{run_id}-{name}",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "source_manifest_ref": manifest.reference,
            "preselection_locator_ref": (None if preselection is None else preselection.reference),
            "authority": authority_envelope(),
        },
    )


def _find_calendar_path(phase1_inventory: Mapping[str, Any]) -> tuple[Path, str]:
    acquisition = phase1_inventory.get("acquisition")
    rows = acquisition.get("files") if isinstance(acquisition, Mapping) else None
    if not isinstance(rows, list):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    matches: list[tuple[Path, str]] = []
    for row in rows:
        if (
            isinstance(row, Mapping)
            and isinstance(row.get("query_id"), str)
            and str(row["query_id"]).startswith("trade_cal_cn_")
            and row.get("kind") == "parquet"
        ):
            matches.append(
                (Path(str(row["path"])), str(row["expected_sha256"]))
            )
    if len(matches) != 1:
        raise CurrentShadowBuildError("source_dataset_calendar_missing")
    return matches[0]


def _last_completed_session(calendar: pd.DataFrame, cutoff: datetime) -> str:
    required = {"cal_date", "is_open"}
    if not required.issubset(calendar.columns):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    if calendar["cal_date"].duplicated().any():
        raise CurrentShadowBuildError("source_dataset_duplicate_key")
    local = cutoff.astimezone(SHANGHAI)
    completed_date = local.date()
    if local.time() < time(15, 0):
        completed_date = completed_date.fromordinal(completed_date.toordinal() - 1)
    sessions = pd.to_datetime(
        calendar.loc[calendar["is_open"].eq(1), "cal_date"],
        format="%Y%m%d",
        errors="coerce",
    ).dt.date
    candidates = [value for value in sessions.dropna().tolist() if value <= completed_date]
    if not candidates:
        raise CurrentShadowBuildError("source_dataset_calendar_missing")
    return max(candidates).strftime("%Y%m%d")


def _market_partitions(table_root: Path, latest_session: str) -> list[Path]:
    latest = pd.Timestamp(latest_session)
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in table_root.glob("year=*/month=*/part.parquet"):
        try:
            year = int(path.parent.parent.name.split("=", 1)[1])
            month = int(path.parent.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        stamp = pd.Timestamp(year=year, month=month, day=1)
        if stamp <= latest:
            candidates.append((stamp, path))
    candidates.sort(reverse=True)
    selected: list[Path] = []
    observed_dates: set[str] = set()
    for _, path in candidates:
        selected.append(path)
        dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"].astype(str)
        observed_dates.update(dates[dates <= latest_session].tolist())
        if len(observed_dates) >= 140:
            break
    if len(observed_dates) < 121:
        raise CurrentShadowBuildError("source_dataset_insufficient_history")
    return sorted(selected)


def _load_market(
    table_root: Path,
    *,
    latest_session: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if table_root.is_symlink() or not table_root.is_dir():
        raise CurrentShadowBuildError("source_dataset_table_root_invalid")
    parts = _market_partitions(table_root, latest_session)
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for path in parts:
        raw = path.read_bytes()
        inventory.append(
            {
                "relative_path": str(path.relative_to(table_root)),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
        frame = pd.read_parquet(path)
        if not set(REQUIRED_MARKET_COLUMNS).issubset(frame.columns):
            raise CurrentShadowBuildError("source_dataset_schema_invalid")
        frames.append(frame.loc[:, list(REQUIRED_MARKET_COLUMNS)])
    market = pd.concat(frames, ignore_index=True)
    market["trade_date"] = market["trade_date"].astype(str)
    market = market.loc[market["trade_date"] <= latest_session].copy()
    if market.empty:
        raise CurrentShadowBuildError("source_dataset_stale")
    if market[["ts_code", "trade_date"]].duplicated().any():
        raise CurrentShadowBuildError("source_dataset_duplicate_key")
    if market["trade_date"].max() != latest_session:
        raise CurrentShadowBuildError("source_dataset_stale")
    for column in ("adj_close", "close", "vol", "amount"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return market, inventory


def _active_membership(
    membership: pd.DataFrame,
    *,
    latest_session: str,
) -> pd.DataFrame:
    if not set(REQUIRED_MEMBERSHIP_COLUMNS).issubset(membership.columns):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    if membership["symbol"].duplicated().any():
        raise CurrentShadowBuildError("source_dataset_duplicate_key")
    if not membership["observed_at"].map(lambda value: isinstance(value, str)).all():
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    active = membership.loc[
        membership["source_list_status"].eq("L")
        & membership["list_date"].astype(str).le(latest_session)
        & (
            membership["delist_date"].fillna("").astype(str).eq("")
            | membership["delist_date"].astype(str).gt(latest_session)
        )
        & membership["effective_from"].astype(str).le(latest_session)
        & (
            membership["effective_to"].fillna("").astype(str).eq("")
            | membership["effective_to"].astype(str).ge(latest_session)
        )
    ].copy()
    if len(active) < 5000:
        raise CurrentShadowBuildError("source_dataset_membership_incomplete")
    return active.sort_values("symbol").reset_index(drop=True)


def _factor_inputs(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    latest_session: str,
    baseline_policy: Mapping[str, Any],
    preselector_policy_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, pd.Series],
    dict[str, Mapping[str, Any]],
]:
    frames = {
        str(symbol): group.tail(140).copy()
        for symbol, group in market.groupby("ts_code", sort=True)
    }
    prepared = prepare_price_volume_frames(
        frames,
        include_amihud_base=True,
        lookback_rows=140,
    )
    cache: dict[str, Any] = {
        "active_price_volume_names": [
            "pv_amihud_illiquidity_20d",
            *QUANT_FACTOR_NAMES,
            "pv_momentum_120d",
            "pv_price_efficiency_60d",
        ]
    }
    factor_values: dict[str, pd.Series] = {
        "pv_momentum_120d": compute_price_volume_factor(
            "pv_momentum_120d",
            frames,
            prepared_frames=prepared,
            factor_cache=cache,
        ),
        "pv_amihud_liquidity_20d": -compute_price_volume_factor(
            "pv_amihud_illiquidity_20d",
            frames,
            prepared_frames=prepared,
            factor_cache=cache,
        ),
        "pv_price_efficiency_60d": compute_price_volume_factor(
            "pv_price_efficiency_60d",
            frames,
            prepared_frames=prepared,
            factor_cache=cache,
        ),
    }
    for name in QUANT_FACTOR_NAMES:
        factor_values[name] = compute_price_volume_factor(
            name,
            frames,
            prepared_frames=prepared,
            factor_cache=cache,
        )
    scored_by_symbol = {
        str(row["symbol"]): row
        for row in scored.to_dict("records")
        if type(row.get("symbol")) is str
    }
    observations: list[dict[str, Any]] = []
    observation_by_symbol: dict[str, Mapping[str, Any]] = {}
    for symbol in membership["symbol"].astype(str):
        frame = frames.get(symbol)
        history_count = 0 if frame is None else int(frame["adj_close"].notna().sum())
        latest_row = (
            None
            if frame is None or frame.empty
            else frame.loc[frame["trade_date"].astype(str).eq(latest_session)]
        )
        latest_ready = latest_row is not None and not latest_row.empty
        values: list[dict[str, str]] = []
        finite = True
        for factor_id in PRESELECT_FACTOR_NAMES:
            value = factor_values[factor_id].get(symbol, math.nan)
            if not math.isfinite(float(value)):
                finite = False
                continue
            values.append(
                {
                    "factor_id": factor_id,
                    "value": _decimal_text(value),
                }
            )
        score_row = scored_by_symbol.get(symbol, {})
        research_eligible = (
            score_row.get("research_eligible") is True
            and score_row.get("membership_conflict") is False
            and score_row.get("membership_is_pit") is True
        )
        liquid = bool(
            latest_ready
            and float(pd.to_numeric(latest_row["amount"], errors="coerce").iloc[-1]) > 0
        )
        tradable = bool(
            latest_ready and float(pd.to_numeric(latest_row["close"], errors="coerce").iloc[-1]) > 0
        )
        row = {
            "symbol": symbol,
            "history_count": history_count,
            "data_ready": finite and len(values) == 3 and history_count >= 121,
            "liquid": liquid,
            "research_eligible": research_eligible,
            "tradable": tradable,
            "factor_values": sorted(values, key=lambda item: item["factor_id"]),
        }
        observations.append(row)
        observation_by_symbol[symbol] = row
    preselector_factors = baseline_policy["preselector_factors"]
    factor_contract = sorted(
        (
            {
                "definition_hash": row["definition_sha256"],
                "family": row["family_id"],
                "lineage": row["lineage_id"],
                "lookback": row["lookback_open_days"],
                "minimum_coverage": "0.90",
                "name": row["factor_id"],
                "warmup": row["lookback_open_days"],
                "weight": row["weight"],
            }
            for row in preselector_factors
        ),
        key=lambda row: row["name"],
    )
    quant_inventory = sorted(
        (
            {
                "definition_hash": row["definition_sha256"],
                "family": row["family_id"],
                "lineage": row["lineage_id"],
                "name": row["factor_id"],
            }
            for row in baseline_policy["quant_factors"]
        ),
        key=lambda row: row["name"],
    )
    payload = {
        "factor_contract": factor_contract,
        "observations": observations,
        "policy_sha256": preselector_policy_sha256,
        "quant_branch_inventory": quant_inventory,
    }
    return [payload], factor_values, observation_by_symbol


def _branch_artifact(
    store: SecureStore,
    *,
    run_id: str,
    branch: str,
    strategy_id: str,
    cutoff: str,
    preselection_locator: RuntimeArtifact,
    initial_pool: RuntimeArtifact,
    records: Sequence[Mapping[str, Any]],
) -> RuntimeArtifact:
    pool = list(initial_pool.document["selected_symbols"])
    policy = load_packaged_json(f"resources/{branch}_branch_policy.v1.json")
    return _write_typed(
        store,
        relative_path=PRIVATE_SOURCES_ROOT / run_id / "derived" / f"{branch}_branch.json",
        payload={
            "version": "myquant.v17.v3.branch-output.v1",
            "protocol_version": PROTOCOL_VERSION,
            "output_id": f"{run_id}-{branch}-branch",
            "run_id": run_id,
            "branch": branch,
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "state": "BRANCHES_COMPLETE",
            "source_locator_ref": preselection_locator.reference,
            "initial_pool_ref": initial_pool.reference,
            "initial_pool_count": len(pool),
            "initial_pool_symbol_order_sha256": hashlib.sha256(canonical_bytes(pool)).hexdigest(),
            "policy_sha256": policy["semantic_sha256"],
            "ordered_domain": pool,
            "records": list(records),
            "authority": authority_envelope(),
        },
    )


def _branch_records(
    pool: Sequence[str],
    factor_values: Mapping[str, pd.Series],
    scored: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quant_frame = pd.DataFrame(
        {name: factor_values[name].reindex(pool) for name in QUANT_FACTOR_NAMES},
        index=list(pool),
    )
    quant_score = quant_frame.rank(pct=True, method="average").mean(axis=1)
    fundamental_by_symbol = scored.set_index("symbol", drop=False)
    quant: list[dict[str, Any]] = []
    fundamental: list[dict[str, Any]] = []
    for symbol in pool:
        q_value = quant_score.get(symbol, math.nan)
        if math.isfinite(float(q_value)):
            quant.append(
                {
                    "symbol": symbol,
                    "status": "READY",
                    "score": _decimal_text(q_value),
                    "reason": None,
                }
            )
        else:
            quant.append(
                {
                    "symbol": symbol,
                    "status": "UNAVAILABLE",
                    "score": None,
                    "reason": "quant_factor_unavailable",
                }
            )
        if symbol in fundamental_by_symbol.index:
            row = fundamental_by_symbol.loc[symbol]
            value = row.get("total_score")
            available = (
                row.get("status") == "AVAILABLE"
                and value is not None
                and math.isfinite(float(value))
            )
        else:
            value = None
            available = False
        fundamental.append(
            {
                "symbol": symbol,
                "status": "READY" if available else "UNAVAILABLE",
                "score": _decimal_text(value) if available else None,
                "reason": None if available else "fundamental_score_unavailable",
            }
        )
    return quant, fundamental


def _expected_top24(
    *,
    pool: Sequence[str],
    quant_branch: RuntimeArtifact,
    fundamental_branch: RuntimeArtifact,
    preselection_locator: RuntimeArtifact,
    initial_pool: RuntimeArtifact,
    cutoff: str,
) -> tuple[str, ...]:
    order_sha = hashlib.sha256(canonical_bytes(list(pool))).hexdigest()
    bindings = {
        "source_locator_path": str(preselection_locator.relative_path),
        "source_locator_byte_sha256": preselection_locator.byte_sha256,
        "cutoff": cutoff,
        "pool_byte_sha256": initial_pool.byte_sha256,
        "pool_semantic_sha256": str(initial_pool.document["semantic_sha256"]),
        "pool_count": str(len(pool)),
        "pool_symbol_order_sha256": order_sha,
    }

    def normalized(artifact: RuntimeArtifact) -> dict[str, Any]:
        return {
            "branch": artifact.document["branch"],
            "ordered_domain": artifact.document["ordered_domain"],
            "bindings": bindings,
            "records": artifact.document["records"],
        }

    quant = validate_branch_output(
        normalized(quant_branch),
        ordered_pool=pool,
        expected_bindings=bindings,
    )
    fundamental = validate_branch_output(
        normalized(fundamental_branch),
        ordered_pool=pool,
        expected_bindings=bindings,
    )
    fusion = fuse_branches(
        quant,
        fundamental,
        ordered_pool=pool,
        quant_weight=0.5,
        quant_bindings=bindings,
        fundamental_bindings=bindings,
        top_n=24,
    )
    selected = tuple(fusion.selected_symbols)
    if fusion.status != "READY" or len(selected) != 24:
        raise CurrentShadowBuildError("source_dataset_ready_coverage_below_24")
    return selected


def _initial_pool_artifact(
    store: SecureStore,
    outcome: Any,
) -> RuntimeArtifact:
    raw = store.read(outcome.relative_path, outcome.byte_sha256)
    document = load_typed_artifact(
        raw,
        label="initial pool",
        expected_version="myquant.v17.v3.initial-pool-output.v1",
    )
    return runtime_artifact(
        relative_path=outcome.relative_path,
        document=document,
    )


def _compile(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = _absolute(args.workspace_root, label="workspace_root")
    repo_root = _absolute(args.repo_root, label="repo_root")
    phase1_root = _absolute(args.phase1_run_root, label="phase1_run_root")
    market_pointer_path = _absolute(args.market_pointer, label="market_pointer")
    readiness_path = _absolute(args.factor_readiness, label="factor_readiness")
    bak_manifest_path = _absolute(
        args.bak_basic_acquisition_manifest,
        label="bak_basic_acquisition_manifest",
    )
    cutoff, cutoff_instant = _cutoff(args.cutoff)
    strategy_id = require_opaque_id(args.strategy_id, label="strategy_id")
    run_id = require_opaque_id(args.run_id, label="run_id")

    if workspace_root.exists():
        if workspace_root.is_symlink() or not workspace_root.is_dir():
            raise CurrentShadowBuildError("workspace_root_invalid")
        if stat.S_IMODE(workspace_root.stat().st_mode) != 0o700:
            raise CurrentShadowBuildError("workspace_root_mode_invalid")
        formal_root = workspace_root / FORMAL_RESULTS_ROOT
        if formal_root.exists() and any(formal_root.rglob("*")):
            raise CurrentShadowBuildError("active_v3_state_present")
    else:
        workspace_root.mkdir(mode=0o700, parents=True)
        workspace_root.chmod(0o700)

    pointer_raw = _read_exact(
        market_pointer_path,
        args.expected_market_pointer_sha256,
        label="market_pointer",
    )
    readiness_raw = _read_exact(
        readiness_path,
        args.expected_factor_readiness_sha256,
        label="factor_readiness",
    )
    bak_raw = _read_exact(
        bak_manifest_path,
        args.expected_bak_basic_manifest_sha256,
        label="bak_basic_manifest",
    )
    pointer = _json_object(pointer_raw, label="market pointer")
    readiness_source = _json_object(readiness_raw, label="factor readiness")
    bak_manifest = _json_object(bak_raw, label="bak_basic acquisition manifest")
    if _contains_credential_material(bak_manifest):
        raise CurrentShadowBuildError("credential_material_detected")
    if "token_persisted" in bak_manifest and bak_manifest.get("token_persisted") is not False:
        raise CurrentShadowBuildError("credential_material_detected")

    inventory_raw = (phase1_root / "input_inventory.json").read_bytes()
    phase1_inventory = _json_object(inventory_raw, label="phase1 input inventory")
    calendar_path, calendar_sha = _find_calendar_path(phase1_inventory)
    calendar_raw = _read_exact(
        calendar_path,
        calendar_sha,
        label="calendar_source",
    )
    calendar = pd.read_parquet(BytesIO(calendar_raw))
    latest_session = _last_completed_session(calendar, cutoff_instant)
    if str(pointer.get("latest_complete_trade_date")) != latest_session:
        raise CurrentShadowBuildError("source_dataset_stale")

    table_root_value = pointer.get("table_root")
    if type(table_root_value) is not str or ".." in Path(table_root_value).parts:
        raise CurrentShadowBuildError("source_dataset_table_root_invalid")
    table_root = (
        Path(table_root_value)
        if Path(table_root_value).is_absolute()
        else repo_root / table_root_value
    )
    market, shard_inventory = _load_market(
        table_root,
        latest_session=latest_session,
    )

    canonical_inventory = phase1_inventory.get("canonical")
    if not isinstance(canonical_inventory, Mapping):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    membership_meta = canonical_inventory.get("pit_membership")
    if not isinstance(membership_meta, Mapping):
        raise CurrentShadowBuildError("source_dataset_membership_missing")
    membership_path = _absolute(
        str(membership_meta.get("path")),
        label="membership",
    )
    membership_raw = _read_exact(
        membership_path,
        str(membership_meta.get("sha256")),
        label="membership",
    )
    membership = _active_membership(
        pd.read_parquet(BytesIO(membership_raw)),
        latest_session=latest_session,
    )

    fundamental_path = phase1_root / "fundamental_snapshot.parquet"
    scored_path = phase1_root / "scored_full_a.parquet"
    if not fundamental_path.is_file() or not scored_path.is_file():
        raise CurrentShadowBuildError("source_dataset_fundamental_missing")
    fundamental = pd.read_parquet(fundamental_path)
    scored = pd.read_parquet(scored_path)
    if not set(REQUIRED_FUNDAMENTAL_COLUMNS).issubset(fundamental.columns):
        raise CurrentShadowBuildError("source_dataset_schema_invalid")
    if fundamental["symbol"].duplicated().any() or scored["symbol"].duplicated().any():
        raise CurrentShadowBuildError("source_dataset_duplicate_key")
    availability = pd.to_datetime(fundamental["availability"], utc=True, errors="coerce")
    if availability.isna().any() or availability.max().to_pydatetime() > cutoff_instant:
        raise CurrentShadowBuildError("source_dataset_cutoff_drift")

    baseline_policy = load_packaged_json("resources/provisional_factor_baseline_policy.v1.json")
    preselector_policy = load_packaged_json("resources/preselector_policy.v1.json")
    allocation_policy = load_packaged_json("resources/portfolio_allocation_policy.v1.json")
    if (
        tuple(row["factor_id"] for row in baseline_policy["preselector_factors"])
        != PRESELECT_FACTOR_NAMES
        or tuple(row["factor_id"] for row in baseline_policy["quant_factors"]) != QUANT_FACTOR_NAMES
    ):
        raise CurrentShadowBuildError("factor_policy_inventory_mismatch")
    payload_rows, factor_values, observation_by_symbol = _factor_inputs(
        market,
        membership,
        scored,
        latest_session=latest_session,
        baseline_policy=baseline_policy,
        preselector_policy_sha256=str(preselector_policy["semantic_sha256"]),
    )
    preselection_payload = payload_rows[0]

    store = SecureStore(workspace_root)
    store.initialize()
    calendar_sessions = (
        pd.to_datetime(
            calendar.loc[calendar["is_open"].eq(1), "cal_date"],
            format="%Y%m%d",
        )
        .dt.strftime("%Y-%m-%d")
        .sort_values()
    )
    calendar_frame = pd.DataFrame({"trade_date": calendar_sessions})
    membership_frame = membership.copy()
    market_frame = market.copy()
    fundamental_frame = fundamental.copy()
    raw_refs = {
        "cn_open_day_calendar": _write_raw(
            store,
            run_id=run_id,
            role="cn_open_day_calendar",
            suffix="parquet",
            raw=_parquet_bytes(calendar_frame),
            strategy_id=strategy_id,
            cutoff=cutoff,
        ),
        "market_bars": _write_raw(
            store,
            run_id=run_id,
            role="market_bars",
            suffix="parquet",
            raw=_parquet_bytes(market_frame),
            strategy_id=strategy_id,
            cutoff=cutoff,
        ),
        "pit_fundamentals": _write_raw(
            store,
            run_id=run_id,
            role="pit_fundamentals",
            suffix="parquet",
            raw=_parquet_bytes(fundamental_frame),
            strategy_id=strategy_id,
            cutoff=cutoff,
        ),
        "universe_membership": _write_raw(
            store,
            run_id=run_id,
            role="universe_membership",
            suffix="parquet",
            raw=_parquet_bytes(membership_frame),
            strategy_id=strategy_id,
            cutoff=cutoff,
        ),
    }

    source_as_of = str(readiness_source.get("as_of"))
    readiness_ready = readiness_source.get("factor_governance_ready") is True
    readiness_artifact = _write_typed(
        store,
        relative_path=(PRIVATE_SOURCES_ROOT / run_id / "raw" / "factor_governance_readiness.json"),
        payload={
            "version": "myquant.v17.v3.factor-governance-readiness.v1",
            "protocol_version": PROTOCOL_VERSION,
            "readiness_id": f"{run_id}-factor-readiness",
            "role": "factor_governance_readiness",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "available_at": cutoff,
            "source_schema_version": "factor-governance-readiness.v4",
            "source_byte_sha256": args.expected_factor_readiness_sha256,
            "source_as_of": source_as_of,
            "readiness_status": ("FACTOR_V4_READY" if readiness_ready else "FACTOR_V4_NOT_READY"),
            "factor_governance_ready": readiness_ready,
            "production_factor_count": int(readiness_source.get("production_factor_count", 0)),
            "production_family_count": int(readiness_source.get("production_family_count", 0)),
            "healthy_factor_count": int(readiness_source.get("healthy_factor_count", 0)),
            "activation_receipt_valid": bool(
                (
                    readiness_source.get("activation_receipt")
                    if isinstance(readiness_source.get("activation_receipt"), Mapping)
                    else {}
                ).get("valid", False)
            ),
            "blockers": sorted(
                {
                    str(blocker)
                    for blocker in readiness_source.get("blockers", ())
                    if type(blocker) is str and blocker
                }
            ),
            "authority": authority_envelope(),
        },
    )
    raw_refs["factor_governance_readiness"] = readiness_artifact.reference
    if readiness_ready:
        raise CurrentShadowBuildError("provisional_baseline_requires_factor_v4_not_ready")

    baseline_artifact = _write_typed(
        store,
        relative_path=(
            PRIVATE_SOURCES_ROOT / run_id / "derived" / "provisional_factor_baseline.json"
        ),
        payload={
            "version": "myquant.v17.v3.provisional-factor-baseline.v1",
            "protocol_version": PROTOCOL_VERSION,
            "baseline_id": f"{run_id}-provisional-factor-baseline",
            "role": "provisional_factor_baseline",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "factor_governance_readiness_ref": readiness_artifact.reference,
            "factor_baseline_mode": FACTOR_BASELINE_MODE,
            "policy_sha256": baseline_policy["semantic_sha256"],
            "preselector_factors": baseline_policy["preselector_factors"],
            "quant_factors": baseline_policy["quant_factors"],
            "authority": authority_envelope(),
        },
    )
    quant_inputs = _write_typed(
        store,
        relative_path=(
            PRIVATE_SOURCES_ROOT / run_id / "derived" / "quant_preselection_inputs.json"
        ),
        payload={
            "version": "myquant.v17.v3.quant-preselection-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": f"{run_id}-quant-preselection-inputs",
            "run_id": run_id,
            "role": "quant_preselection_inputs",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "factor_baseline_ref": baseline_artifact.reference,
            "factor_baseline_mode": FACTOR_BASELINE_MODE,
            "payload": preselection_payload,
            "authority": authority_envelope(),
        },
    )
    raw_manifest = _manifest(
        store,
        run_id=run_id,
        name="raw-shadow-current",
        strategy_id=strategy_id,
        cutoff=cutoff,
        phase="RAW",
        raw_profile="SHADOW_CURRENT",
        sources=[{"role": role, "artifact_ref": reference} for role, reference in raw_refs.items()],
    )
    preselect_manifest = _manifest(
        store,
        run_id=run_id,
        name="shadow-current-preselect",
        strategy_id=strategy_id,
        cutoff=cutoff,
        phase=PRESELECT_PHASE,
        parent=raw_manifest,
        sources=[
            {
                "role": "provisional_factor_baseline",
                "artifact_ref": baseline_artifact.reference,
            },
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": quant_inputs.reference,
            },
        ],
    )
    preselect_locator = _locator(
        store,
        run_id=run_id,
        name="shadow-current-preselect",
        strategy_id=strategy_id,
        cutoff=cutoff,
        manifest=preselect_manifest,
        preselection=None,
    )
    initial_outcome = build_initial_pool(
        workspace_root=workspace_root,
        locator_path=str(preselect_locator.relative_path),
        expected_locator_sha256=preselect_locator.byte_sha256,
    )
    initial_pool = _initial_pool_artifact(store, initial_outcome)
    pool = list(initial_pool.document["selected_symbols"])
    if not 24 <= len(pool) <= 500:
        raise CurrentShadowBuildError("source_dataset_ready_coverage_below_24")

    quant_records, fundamental_records = _branch_records(
        pool,
        factor_values,
        scored,
    )
    quant_branch = _branch_artifact(
        store,
        run_id=run_id,
        branch="quant",
        strategy_id=strategy_id,
        cutoff=cutoff,
        preselection_locator=preselect_locator,
        initial_pool=initial_pool,
        records=quant_records,
    )
    fundamental_branch = _branch_artifact(
        store,
        run_id=run_id,
        branch="fundamental",
        strategy_id=strategy_id,
        cutoff=cutoff,
        preselection_locator=preselect_locator,
        initial_pool=initial_pool,
        records=fundamental_records,
    )
    top24 = _expected_top24(
        pool=pool,
        quant_branch=quant_branch,
        fundamental_branch=fundamental_branch,
        preselection_locator=preselect_locator,
        initial_pool=initial_pool,
        cutoff=cutoff,
    )
    gross = Decimal(str(allocation_policy["gross_weight"]))
    cap = Decimal(str(allocation_policy["max_weight_per_symbol"]))
    per_name = min(
        cap,
        (gross / Decimal(len(top24))).quantize(
            Decimal("0.00000001"),
            rounding=ROUND_DOWN,
        ),
    )
    base_target = format(per_name, "f")
    deep_inputs = _write_typed(
        store,
        relative_path=(PRIVATE_SOURCES_ROOT / run_id / "derived" / "deep_research_inputs.json"),
        payload={
            "version": "myquant.v17.v3.deep-research-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": f"{run_id}-deep-inputs",
            "run_id": run_id,
            "role": "deep_research_inputs",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "payload": [
                {
                    "symbol": symbol,
                    "lane": "SELECTION_POOL",
                    "held": False,
                    "available": False,
                    "signal": None,
                    "veto_buy": True,
                    "base_target": base_target,
                    "current_target": "0",
                    "evidence_refs": [],
                }
                for symbol in sorted(top24)
            ],
            "authority": authority_envelope(),
        },
    )
    permissions = _write_typed(
        store,
        relative_path=PRIVATE_SOURCES_ROOT / run_id / "derived" / "permissions.json",
        payload={
            "version": "myquant.v17.v3.pretrade-permissions.v1",
            "protocol_version": PROTOCOL_VERSION,
            "permissions_id": f"{run_id}-permissions",
            "run_id": run_id,
            "role": "permissions",
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "created_at": cutoff,
            "portfolio_basis": PORTFOLIO_BASIS,
            "canonical_calendar_ref": raw_refs["cn_open_day_calendar"],
            "decision_session": datetime.strptime(
                latest_session,
                "%Y%m%d",
            ).strftime("%Y-%m-%d"),
            "holdings_snapshot_as_of_session": None,
            "holdings_snapshot_age_sessions": None,
            "holdings_snapshot_ref": None,
            "payload": [
                {
                    "symbol": symbol,
                    "lane": "SELECTION_POOL",
                    "held": False,
                    "can_buy": bool(
                        observation_by_symbol[symbol]["tradable"]
                        and observation_by_symbol[symbol]["liquid"]
                    ),
                    "current_target": "0",
                }
                for symbol in sorted(top24)
            ],
            "authority": authority_envelope(),
        },
    )
    analyze_manifest = _manifest(
        store,
        run_id=run_id,
        name="shadow-current-model-portfolio",
        strategy_id=strategy_id,
        cutoff=cutoff,
        phase=PORTFOLIO_PHASE,
        parent=raw_manifest,
        sources=[
            {
                "role": "deep_research_inputs",
                "artifact_ref": deep_inputs.reference,
            },
            {
                "role": "fundamental_branch_output",
                "artifact_ref": fundamental_branch.reference,
            },
            {
                "role": "initial_pool_output",
                "artifact_ref": initial_pool.reference,
            },
            {"role": "permissions", "artifact_ref": permissions.reference},
            {
                "role": "provisional_factor_baseline",
                "artifact_ref": baseline_artifact.reference,
            },
            {
                "role": "quant_branch_output",
                "artifact_ref": quant_branch.reference,
            },
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": quant_inputs.reference,
            },
        ],
    )
    analyze_locator = _locator(
        store,
        run_id=run_id,
        name="shadow-current-model-portfolio",
        strategy_id=strategy_id,
        cutoff=cutoff,
        manifest=analyze_manifest,
        preselection=preselect_locator,
    )
    outcome = analyze(
        workspace_root=workspace_root,
        mode="shadow",
        locator_path=str(analyze_locator.relative_path),
        expected_locator_sha256=analyze_locator.byte_sha256,
    )
    result = outcome.result
    names = dict(zip(membership["symbol"], membership["name"], strict=True))
    deep_statuses = {symbol: getattr(decision, "status", "") for symbol, decision in result.deep}
    private_summary = {
        "version": "myquant.v17.v3.current-shadow-run-summary.v1",
        "status": result.terminal.state,
        "strategy_id": strategy_id,
        "run_id": run_id,
        "cutoff": cutoff,
        "decision_session": latest_session,
        "factor_baseline_mode": FACTOR_BASELINE_MODE,
        "portfolio_basis": PORTFOLIO_BASIS,
        "calibration": result.calibration_label,
        "preselection_count": len(pool),
        "fusion_count": len(top24),
        "deep_evaluation_count": len(result.deep),
        "gross_weight": (
            result.artifacts[-2].document.get("gross_weight")
            if len(result.artifacts) >= 2
            else None
        ),
        "cash_weight": (
            result.artifacts[-2].document.get("cash_weight") if len(result.artifacts) >= 2 else None
        ),
        "top24": [
            {
                "symbol": symbol,
                "name": str(names.get(symbol) or "UNKNOWN_NAME"),
                "deep_status": deep_statuses.get(symbol),
                "final_target": _decimal_text(result.final_targets.get(symbol, 0), places=8),
            }
            for symbol in top24
        ],
        "overlay_stages": (
            result.terminal_artifact.document.get("overlay_stages")
            if result.terminal_artifact is not None
            else []
        ),
        "source_bindings": {
            "market_pointer_sha256": args.expected_market_pointer_sha256,
            "factor_readiness_sha256": args.expected_factor_readiness_sha256,
            "bak_basic_acquisition_manifest_sha256": (args.expected_bak_basic_manifest_sha256),
            "bak_basic_admission_status": bak_manifest.get("admission_status"),
            "market_shards": shard_inventory,
            "compiler_build_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preselection_locator_sha256": preselect_locator.byte_sha256,
            "analyze_locator_sha256": analyze_locator.byte_sha256,
        },
        "blockers": list(result.terminal.blockers),
        "authority": authority_envelope(),
    }
    summary_path = PRIVATE_RUNS_ROOT / run_id / "run_summary.json"
    store.write_exact_once(summary_path, canonical_resource_bytes(private_summary))
    public = result.to_public_wire()
    public.update(
        {
            "decision_session": latest_session,
            "private_run_summary": str(summary_path),
            "factor_baseline_mode": FACTOR_BASELINE_MODE,
            "portfolio_basis": PORTFOLIO_BASIS,
        }
    )
    return public


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        public = _compile(args)
        print(json.dumps(public, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0 if public.get("status") == "SHADOW_COMPLETE" else 2
    except (CurrentShadowBuildError, OSError, TypeError, ValueError) as exc:
        blocker = (
            exc.blocker
            if isinstance(exc, CurrentShadowBuildError)
            else f"compiler_contract_error:{type(exc).__name__}"
        )
        payload = {
            "version": "myquant.v17.v3.current-shadow-build-error.v1",
            "status": f"DATA_BLOCKED:{blocker}",
            "blocker": blocker,
            **authority_envelope(),
        }
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
