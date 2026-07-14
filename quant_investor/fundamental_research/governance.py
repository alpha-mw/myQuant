"""Recomputable activation evidence derived only from private hash-chain ledgers."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator
import pandas as pd

from .ledger import HashChainLedger, LedgerConflictError
from .replay import validate_control_chain_replay
from .models import (
    ApplicationEventV1,
    ApplicationState,
    FundamentalResearchRequestV1,
    JobEventV1,
    JobState,
    StrictModel,
)
from .storage import (
    PRIVATE_MODE,
    _assert_contained,
    canonical_json_bytes,
    load_json_model,
    sha256_bytes,
)

JOB_LEDGER = "state/jobs.v1.jsonl"
APPLICATION_LEDGER = "state/applications.v1.jsonl"
LONGITUDINAL_LEDGER = "state/longitudinal.v1.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDINGS_MAX_AGE = timedelta(days=7)
ELIGIBLE_HOLDINGS_STATUSES = frozenset(
    {
        "no_action_carry_forward",
        "filled_local_manual",
        "filled_local_manual_paper_rebalance",
        "rejected_no_fill_carry_forward",
    }
)
Sha256 = Annotated[str, StringConstraints(pattern=r"^(?:|[0-9a-f]{64})$")]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]


class LongitudinalObservationV1(StrictModel):
    """A hash-bound comparison produced after both portfolios are materialized."""

    schema_version: Literal["fundamental-research-longitudinal.v1"] = (
        "fundamental-research-longitudinal.v1"
    )
    event_id: Identifier
    request_id: Identifier
    dossier_id: Identifier
    observation_type: Literal["target_weight", "nav_attribution"]
    run_key: Identifier
    trading_date: date
    application_trading_date: date
    occurred_at: datetime
    actual_artifact_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    counterfactual_artifact_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    actual_artifact_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    counterfactual_artifact_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    actual_value: float = Field(ge=-10.0, le=10.0)
    counterfactual_value: float = Field(ge=-10.0, le=10.0)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_paths_and_date(self) -> "LongitudinalObservationV1":
        if (
            Path(self.actual_artifact_path).is_absolute()
            or Path(self.counterfactual_artifact_path).is_absolute()
        ):
            raise ValueError("longitudinal artifact paths must be relative")
        if self.trading_date > self.occurred_at.date():
            raise ValueError("trading_date cannot follow occurred_at")
        if self.application_trading_date > self.trading_date:
            raise ValueError("application_trading_date cannot follow observation date")
        if (
            self.observation_type == "target_weight"
            and self.application_trading_date != self.trading_date
        ):
            raise ValueError("target-weight observation must use the application date")
        if (
            self.observation_type == "nav_attribution"
            and self.application_trading_date >= self.trading_date
        ):
            raise ValueError("NAV attribution must follow the application date")
        return self


class LongitudinalOutcomeArtifactV1(StrictModel):
    schema_version: Literal["fundamental-research-outcome.v1"] = "fundamental-research-outcome.v1"
    observation_type: Literal["target_weight", "nav_attribution"]
    variant: Literal["actual", "counterfactual"]
    run_key: Identifier
    trading_date: date
    value: float = Field(ge=-10.0, le=10.0)
    produced_at: datetime
    source_generation: Identifier
    source_kind: Literal["analysis_run_manifest", "canonical_nav_snapshot"]
    source_artifact_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    source_artifact_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("produced_at")
    @classmethod
    def validate_produced_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("produced_at must be timezone-aware")
        return value

    @field_validator("source_artifact_path")
    @classmethod
    def require_relative_source(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("longitudinal source artifact path must be relative")
        return value


class LongitudinalSourceArtifactV1(StrictModel):
    """Normalized, immutable projection of the authoritative run/NAV output."""

    schema_version: Literal["fundamental-research-longitudinal-source.v1"] = (
        "fundamental-research-longitudinal-source.v1"
    )
    source_kind: Literal["analysis_run_manifest", "canonical_nav_snapshot"]
    variant: Literal["actual", "counterfactual"]
    run_key: Identifier
    trading_date: date
    generation: Identifier
    produced_at: datetime
    value: float = Field(ge=-10.0, le=10.0)
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
    dossier_variant: Literal["with_dossier", "without_dossier"]
    canonical_artifact_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    canonical_artifact_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    parent_observation_path: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=512)
    ] = ""
    parent_observation_sha256: Sha256 = ""

    @field_validator("produced_at")
    @classmethod
    def require_aware_produced_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source produced_at must be timezone-aware")
        return value

    @field_validator("canonical_artifact_path", "parent_observation_path")
    @classmethod
    def require_relative_canonical_source(cls, value: str) -> str:
        if not value:
            return ""
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("canonical source artifact path must be relative")
        return Path(value).as_posix()


class CanonicalLongitudinalMetricV1(StrictModel):
    run_key: Identifier
    trading_date: date
    application_trading_date: date
    generation: Identifier
    variant: Literal["actual", "counterfactual"]
    dossier_variant: Literal["with_dossier", "without_dossier"]
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
    value: float = Field(ge=-10.0, le=10.0)


class CanonicalNavSnapshotV1(StrictModel):
    schema_version: Literal["portfolio-nav-snapshot.v1"] = "portfolio-nav-snapshot.v1"
    produced_at: datetime
    longitudinal_metric: CanonicalLongitudinalMetricV1
    analysis_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    target_weights: dict[str, float]
    symbol_returns: dict[str, float]
    target_weight_observation_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    snapshot_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("produced_at")
    @classmethod
    def nav_produced_at_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("NAV produced_at must be timezone-aware")
        return value


class HoldingsScopeSnapshotV1(StrictModel):
    schema_version: Literal["fundamental-research-holdings-scope.v1"] = (
        "fundamental-research-holdings-scope.v1"
    )
    generated_at: datetime
    symbols: list[str]
    manual_manifest_repo_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    manual_ledger_repo_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    manual_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    manual_ledger_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value

    @field_validator("symbols")
    @classmethod
    def canonical_symbols(cls, value: list[str]) -> list[str]:
        canonical = sorted({str(item).strip().upper() for item in value if str(item).strip()})
        if not canonical or value != canonical:
            raise ValueError("symbols must be a non-empty sorted unique set")
        return value

    @field_validator("manual_manifest_repo_path", "manual_ledger_repo_path")
    @classmethod
    def canonical_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("holdings source paths must be repository-relative")
        return path.as_posix()


class ActivationGateEvidenceV2(StrictModel):
    schema_version: Literal["fundamental-research-activation-evidence.v2"] = (
        "fundamental-research-activation-evidence.v2"
    )
    generated_at: datetime
    job_ledger_head_sha256: Sha256 = ""
    application_ledger_head_sha256: Sha256 = ""
    longitudinal_ledger_head_sha256: Sha256 = ""
    holdings_snapshot_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    holdings_snapshot_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    validated_request_ids: list[Identifier] = Field(default_factory=list)
    validated_symbols: list[str] = Field(default_factory=list)
    validated_company_names: list[str] = Field(default_factory=list)
    validated_industries: list[str] = Field(default_factory=list)
    holdings_symbols: list[str] = Field(default_factory=list)
    shadow_trading_dates: list[date] = Field(default_factory=list)
    limited_trading_dates: list[date] = Field(default_factory=list)
    target_weight_counterfactual_dates: list[date] = Field(default_factory=list)
    nav_attribution_dates: list[date] = Field(default_factory=list)
    recent_received_request_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    recent_validation_success_count: int = Field(ge=0, le=20)
    critical_error_codes: list[str] = Field(default_factory=list)

    @field_validator("generated_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_canonical_sets(self) -> "ActivationGateEvidenceV2":
        fields = (
            "validated_request_ids",
            "validated_symbols",
            "validated_company_names",
            "validated_industries",
            "holdings_symbols",
            "shadow_trading_dates",
            "limited_trading_dates",
            "target_weight_counterfactual_dates",
            "nav_attribution_dates",
            "critical_error_codes",
        )
        for name in fields:
            value = getattr(self, name)
            if value != sorted(set(value)):
                raise ValueError(f"{name} must be a sorted unique set")
        if self.recent_validation_success_count > len(self.recent_received_request_ids):
            raise ValueError("recent success count exceeds sample size")
        return self

    @property
    def recent_validation_success_rate(self) -> float | None:
        if not self.recent_received_request_ids:
            return None
        return self.recent_validation_success_count / len(self.recent_received_request_ids)

    @property
    def holdings_coverage_passed(self) -> bool:
        return bool(self.holdings_symbols) and set(self.holdings_symbols) <= set(
            self.validated_symbols
        )


def _records(root: Path, relative_path: str) -> list[dict[str, object]]:
    path = root / relative_path
    if not path.exists():
        return []
    return HashChainLedger(root, path).read_records()


def _head(records: list[dict[str, object]]) -> str:
    return str(records[-1].get("event_sha256") or "") if records else ""


def _repo_artifact(relative: str) -> Path:
    root = REPO_ROOT.resolve(strict=True)
    candidate = REPO_ROOT / relative
    if candidate.is_symlink():
        raise ValueError("canonical holdings paths cannot be symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("canonical holdings path escapes repository") from exc
    if not resolved.is_file():
        raise ValueError("canonical holdings path is not a file")
    return resolved


def _stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("canonical holdings source changed during read")
    return payload


def verify_holdings_scope_snapshot(snapshot: HoldingsScopeSnapshotV1, *, as_of: datetime) -> None:
    """Re-read the canonical manifest and Parquet; duplicated hashes are insufficient."""

    manifest_path = _repo_artifact(snapshot.manual_manifest_repo_path)
    ledger_path = _repo_artifact(snapshot.manual_ledger_repo_path)
    if manifest_path.name != "manual_execution_manifest.json":
        raise ValueError("canonical holdings manifest filename mismatch")
    if ledger_path.name != "ledger_after_manual_switch.parquet":
        raise ValueError("canonical holdings ledger filename mismatch")
    manifest_bytes = _stable_bytes(manifest_path)
    ledger_bytes = _stable_bytes(ledger_path)
    if sha256_bytes(manifest_bytes) != snapshot.manual_manifest_sha256:
        raise ValueError("canonical holdings manifest hash mismatch")
    if sha256_bytes(ledger_bytes) != snapshot.manual_ledger_sha256:
        raise ValueError("canonical holdings ledger hash mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical holdings manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "cn_aggressive_manual_execution.v2"
    ):
        raise ValueError("canonical holdings manifest schema mismatch")
    status = str(manifest.get("status", "")).strip().casefold()
    if status not in ELIGIBLE_HOLDINGS_STATUSES:
        raise ValueError("canonical holdings manifest status is not eligible")
    raw_time = next(
        (
            str(manifest.get(key, "")).strip()
            for key in ("recorded_at", "record_timestamp", "timestamp")
            if str(manifest.get(key, "")).strip()
        ),
        "",
    )
    try:
        recorded_at = datetime.fromisoformat(raw_time)
    except ValueError as exc:
        raise ValueError("canonical holdings manifest timestamp is invalid") from exc
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("canonical holdings manifest timestamp must be timezone-aware")
    if snapshot.generated_at > as_of:
        raise ValueError("holdings snapshot is newer than gate cutoff")
    age = as_of - recorded_at
    if age < timedelta(0) or age > HOLDINGS_MAX_AGE:
        raise ValueError("canonical holdings manifest is future-dated or stale")
    declared = str(
        manifest.get("ledger_after_manual_switch_parquet") or manifest.get("next_ledger_path") or ""
    ).strip()
    declared_path = (manifest_path.parent / declared).resolve(strict=True)
    if declared_path != ledger_path:
        raise ValueError("canonical holdings manifest ledger path mismatch")
    declared_hash = (
        str(
            manifest.get("ledger_after_manual_switch_parquet_sha256")
            or manifest.get("next_ledger_sha256")
            or manifest.get("ledger_sha256")
            or ""
        )
        .strip()
        .lower()
    )
    if declared_hash and declared_hash != snapshot.manual_ledger_sha256:
        raise ValueError("canonical holdings manifest ledger hash mismatch")
    frame = pd.read_parquet(ledger_path)
    if sha256_bytes(_stable_bytes(ledger_path)) != snapshot.manual_ledger_sha256:
        raise ValueError("canonical holdings ledger changed during parse")
    symbol_column = next(
        (item for item in ("symbol", "ts_code", "code") if item in frame.columns), ""
    )
    shares_column = next(
        (item for item in ("shares", "quantity", "position") if item in frame.columns), ""
    )
    if not symbol_column:
        raise ValueError("canonical holdings ledger symbol column is missing")
    symbols: set[str] = set()
    for row in frame.to_dict(orient="records"):
        if shares_column:
            try:
                if float(row.get(shares_column, 0.0) or 0.0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        symbol = str(row.get(symbol_column, "") or "").strip().upper()
        if symbol:
            symbols.add(symbol)
    if sorted(symbols) != snapshot.symbols:
        raise ValueError("canonical holdings symbols mismatch")


def _validate_canonical_longitudinal_source(
    *, root: Path, source: LongitudinalSourceArtifactV1
) -> dict[str, object]:
    if source.source_kind == "analysis_run_manifest":
        path = _repo_artifact(source.canonical_artifact_path)
    else:
        _, path = _assert_contained(
            root,
            root / source.canonical_artifact_path,
            allow_missing_leaf=False,
            create_parents=False,
        )
    if path.stat().st_mode & 0o777 != PRIVATE_MODE:
        raise ValueError("canonical longitudinal source permissions are not 0600")
    payload_bytes = path.read_bytes()
    if sha256_bytes(payload_bytes) != source.canonical_artifact_sha256:
        raise ValueError("canonical longitudinal source hash mismatch")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical longitudinal source is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("canonical longitudinal source must be an object")
    if source.source_kind == "analysis_run_manifest":
        if not path.name.startswith("analysis_run_manifest") or not path.name.endswith(".v1.json"):
            raise ValueError("analysis source filename mismatch")
        if payload.get("schema_version") != "analysis-run-manifest.v1":
            raise ValueError("analysis source schema mismatch")
        supplied = str(payload.get("manifest_sha256", "")).lower()
        unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        if supplied != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("analysis source self-hash mismatch")
        meta = payload.get("analysis_meta")
        if not isinstance(meta, dict):
            raise ValueError("analysis source metadata is missing")
        if payload.get("market") != "CN" or str(
            payload.get("analysis_meta_sha256", "")
        ).lower() != sha256_bytes(canonical_json_bytes(meta)):
            raise ValueError("analysis source metadata lineage mismatch")
        if meta.get("fundamental_research_variant") != source.dossier_variant:
            raise ValueError("analysis source dossier variant mismatch")
        portfolio = meta.get("portfolio_decision")
        if not isinstance(portfolio, dict):
            raise ValueError("analysis source portfolio decision is missing")
        weights = portfolio.get("target_weights")
        if not isinstance(weights, dict):
            raise ValueError("analysis source target weights are missing")
        try:
            canonical_value = float(weights.get(source.symbol, 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis source target weight is invalid") from exc
        snapshot = meta.get("data_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        global_context = meta.get("global_context")
        global_context = global_context if isinstance(global_context, dict) else {}
        global_metadata = global_context.get("metadata")
        global_metadata = global_metadata if isinstance(global_metadata, dict) else {}
        nested_snapshot = global_metadata.get("data_snapshot")
        nested_snapshot = nested_snapshot if isinstance(nested_snapshot, dict) else {}
        canonical_generation = str(
            snapshot.get("snapshot_id") or nested_snapshot.get("snapshot_id") or ""
        )
        canonical_trade_date = str(
            snapshot.get("local_latest_trade_date")
            or snapshot.get("latest_complete_trade_date")
            or global_context.get("latest_trade_date")
            or ""
        )
        digits = "".join(ch for ch in canonical_trade_date if ch.isdigit())[:8]
        canonical_date = date.fromisoformat(f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}")
        universe = str(global_context.get("universe_key") or meta.get("universe") or "full_a")
        metric = {
            "run_key": f"CN:{universe}:{digits}",
            "trading_date": canonical_date.isoformat(),
            "generation": canonical_generation,
            "variant": source.variant,
            "value": canonical_value,
            "symbol": source.symbol,
            "dossier_variant": source.dossier_variant,
        }
        produced_raw = payload.get("generated_at")
    else:
        if path.name != "canonical_nav_snapshot.v1.json":
            raise ValueError("NAV source filename mismatch")
        if payload.get("schema_version") != "portfolio-nav-snapshot.v1":
            raise ValueError("NAV source schema mismatch")
        supplied = str(payload.get("snapshot_sha256", "")).lower()
        unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
        if supplied != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("NAV source self-hash mismatch")
        metric = payload.get("longitudinal_metric")
        produced_raw = payload.get("produced_at")
    if not isinstance(metric, dict):
        raise ValueError("canonical longitudinal metric is missing")
    try:
        canonical_produced_at = datetime.fromisoformat(str(produced_raw))
        canonical_date = date.fromisoformat(str(metric.get("trading_date")))
        canonical_value = float(str(metric.get("value")))
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical longitudinal metric is invalid") from exc
    if canonical_produced_at.tzinfo is None or canonical_produced_at.utcoffset() is None:
        raise ValueError("canonical longitudinal timestamp must be timezone-aware")
    expected = (
        str(metric.get("run_key", "")),
        canonical_date,
        str(metric.get("generation", "")),
        str(metric.get("variant", "")),
        canonical_value,
        canonical_produced_at,
        str(metric.get("symbol", "")),
        str(metric.get("dossier_variant", "")),
    )
    actual = (
        source.run_key,
        source.trading_date,
        source.generation,
        source.variant,
        source.value,
        source.produced_at,
        source.symbol,
        source.dossier_variant,
    )
    if expected != actual:
        raise ValueError("canonical longitudinal metric lineage mismatch")
    return payload


def _validate_longitudinal_observation(
    *,
    root: Path,
    observation: LongitudinalObservationV1,
    applications: list[ApplicationEventV1],
) -> None:
    matching = [
        event
        for event in applications
        if event.request_id == observation.request_id
        and event.dossier_id == observation.dossier_id
        and event.run_key == observation.run_key
        and event.run_cutoff.date() == observation.application_trading_date
        and event.occurred_at <= observation.occurred_at
    ]
    if observation.observation_type == "target_weight":
        eligible_states = {
            ApplicationState.SHADOW_EVALUATED,
            ApplicationState.LIMITED_APPLIED,
            ApplicationState.PRODUCTION_APPLIED,
        }
    else:
        eligible_states = {
            ApplicationState.LIMITED_APPLIED,
            ApplicationState.PRODUCTION_APPLIED,
        }
    matching = [event for event in matching if event.state in eligible_states]
    if not matching:
        raise ValueError("longitudinal observation has no matching application event")
    expected_kind = (
        "analysis_run_manifest"
        if observation.observation_type == "target_weight"
        else "canonical_nav_snapshot"
    )
    outcomes: list[LongitudinalOutcomeArtifactV1] = []
    sources: list[LongitudinalSourceArtifactV1] = []
    canonical_payloads: list[dict[str, object]] = []
    parent_observations: list[LongitudinalObservationV1] = []
    for variant, relative, expected_sha in (
        ("actual", observation.actual_artifact_path, observation.actual_artifact_sha256),
        (
            "counterfactual",
            observation.counterfactual_artifact_path,
            observation.counterfactual_artifact_sha256,
        ),
    ):
        _, path = _assert_contained(
            root, root / relative, allow_missing_leaf=False, create_parents=False
        )
        if path.stat().st_mode & 0o777 != PRIVATE_MODE:
            raise ValueError("longitudinal outcome artifact permissions are not 0600")
        if sha256_bytes(path.read_bytes()) != expected_sha:
            raise ValueError("longitudinal outcome artifact hash mismatch")
        outcome = load_json_model(root, path, LongitudinalOutcomeArtifactV1)
        if (
            outcome.variant != variant
            or outcome.observation_type != observation.observation_type
            or outcome.run_key != observation.run_key
            or outcome.trading_date != observation.trading_date
            or outcome.source_kind != expected_kind
            or outcome.produced_at > observation.occurred_at
        ):
            raise ValueError("longitudinal outcome lineage mismatch")
        _, source_path = _assert_contained(
            root,
            root / outcome.source_artifact_path,
            allow_missing_leaf=False,
            create_parents=False,
        )
        if source_path.stat().st_mode & 0o777 != PRIVATE_MODE:
            raise ValueError("longitudinal source artifact permissions are not 0600")
        if sha256_bytes(source_path.read_bytes()) != outcome.source_artifact_sha256:
            raise ValueError("longitudinal source artifact hash mismatch")
        source = load_json_model(root, source_path, LongitudinalSourceArtifactV1)
        if (
            source.source_kind != outcome.source_kind
            or source.variant != variant
            or source.run_key != observation.run_key
            or source.trading_date != observation.trading_date
            or source.generation != outcome.source_generation
            or source.produced_at != outcome.produced_at
            or source.value != outcome.value
        ):
            raise ValueError("longitudinal authoritative source lineage mismatch")
        canonical_payload = _validate_canonical_longitudinal_source(root=root, source=source)
        sources.append(source)
        canonical_payloads.append(canonical_payload)
        if observation.observation_type == "nav_attribution":
            if not source.parent_observation_path or not source.parent_observation_sha256:
                raise ValueError("NAV source parent target-weight observation is missing")
            _, parent_path = _assert_contained(
                root,
                root / source.parent_observation_path,
                allow_missing_leaf=False,
                create_parents=False,
            )
            if sha256_bytes(parent_path.read_bytes()) != source.parent_observation_sha256:
                raise ValueError("NAV source parent observation hash mismatch")
            parent = load_json_model(root, parent_path, LongitudinalObservationV1)
            if (
                parent.observation_type != "target_weight"
                or parent.request_id != observation.request_id
                or parent.dossier_id != observation.dossier_id
                or parent.run_key != observation.run_key
                or parent.trading_date != observation.application_trading_date
            ):
                raise ValueError("NAV source parent observation lineage mismatch")
            parent_outcome_relative = (
                parent.actual_artifact_path
                if variant == "actual"
                else parent.counterfactual_artifact_path
            )
            parent_outcome = load_json_model(
                root,
                root / parent_outcome_relative,
                LongitudinalOutcomeArtifactV1,
            )
            parent_source = load_json_model(
                root,
                root / parent_outcome.source_artifact_path,
                LongitudinalSourceArtifactV1,
            )
            if (
                canonical_payload.get("analysis_manifest_sha256")
                != parent_source.canonical_artifact_sha256
                or canonical_payload.get("target_weight_observation_sha256")
                != source.parent_observation_sha256
            ):
                raise ValueError("NAV source analysis/target lineage mismatch")
            parent_observations.append(parent)
        outcomes.append(outcome)
    if outcomes[0].source_generation != outcomes[1].source_generation:
        raise ValueError("longitudinal source generation mismatch")
    if (
        outcomes[0].value != observation.actual_value
        or outcomes[1].value != observation.counterfactual_value
    ):
        raise ValueError("longitudinal outcome value mismatch")
    if observation.observation_type == "target_weight":
        application_state = matching[-1].state
        expected_actual_variant = (
            "without_dossier"
            if application_state == ApplicationState.SHADOW_EVALUATED
            else "with_dossier"
        )
        expected_counterfactual_variant = (
            "with_dossier" if expected_actual_variant == "without_dossier" else "without_dossier"
        )
        if (
            sources[0].dossier_variant != expected_actual_variant
            or sources[1].dossier_variant != expected_counterfactual_variant
        ):
            raise ValueError("actual/counterfactual dossier variant mismatch")
        actual_run_id = str(canonical_payloads[0].get("run_id", ""))
        counter_meta = canonical_payloads[1].get("analysis_meta")
        counter_meta = counter_meta if isinstance(counter_meta, dict) else {}
        if (
            counter_meta.get("fundamental_research_source_run_id") != actual_run_id
            or counter_meta.get("fundamental_research_source_manifest_sha256")
            != sources[0].canonical_artifact_sha256
        ):
            raise ValueError("companion analysis source mismatch")
        replay = counter_meta.get("fundamental_research_control_chain")
        replay = replay if isinstance(replay, dict) else {}
        try:
            validate_control_chain_replay(replay)
        except ValueError as exc:
            raise ValueError("companion deterministic control-chain replay mismatch") from exc
        if replay.get("variant") != sources[1].dossier_variant or replay.get(
            "portfolio_decision"
        ) != counter_meta.get("portfolio_decision"):
            raise ValueError("companion deterministic control-chain replay mismatch")
    else:
        if len(parent_observations) != 2 or parent_observations[0] != parent_observations[1]:
            raise ValueError("NAV parent target-weight observation mismatch")
        parent = parent_observations[0]
        ledger_parents = [
            LongitudinalObservationV1.model_validate(record.get("event") or {})
            for record in _records(root, LONGITUDINAL_LEDGER)
        ]
        if parent not in ledger_parents:
            raise ValueError("NAV parent target-weight observation is not in ledger")
        _validate_longitudinal_observation(root=root, observation=parent, applications=applications)


def build_activation_gate_evidence(
    *,
    root: str | Path,
    holdings_snapshot_path: str | Path,
    generated_at: datetime,
) -> ActivationGateEvidenceV2:
    """Rebuild all activation counts from immutable ledger events at a PIT cutoff."""

    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    root_path = Path(root)
    snapshot_path = Path(holdings_snapshot_path)
    snapshot = load_json_model(root_path, snapshot_path, HoldingsScopeSnapshotV1)
    verify_holdings_scope_snapshot(snapshot, as_of=generated_at)
    snapshot_sha = sha256_bytes(snapshot_path.read_bytes())
    job_records = _records(root_path, JOB_LEDGER)
    application_records = _records(root_path, APPLICATION_LEDGER)
    longitudinal_records = _records(root_path, LONGITUDINAL_LEDGER)

    job_pairs = [
        (record, JobEventV1.model_validate(record.get("event") or {})) for record in job_records
    ]
    job_pairs = [pair for pair in job_pairs if pair[1].occurred_at <= generated_at]
    job_events = [pair[1] for pair in job_pairs]
    latest: dict[str, JobEventV1] = {}
    received_order: list[str] = []
    for event in job_events:
        latest[event.request_id] = event
        if event.state == JobState.RECEIVED:
            received_order.append(event.request_id)
    validated_ids = sorted(
        request_id for request_id, event in latest.items() if event.state == JobState.VALIDATED
    )

    requests: dict[str, tuple[FundamentalResearchRequestV1, Path]] = {}
    for request_path in root_path.glob("CN/*/*/requests/*.request.v1.json"):
        try:
            request = load_json_model(root_path, request_path, FundamentalResearchRequestV1)
        except Exception:
            continue
        requests[request.request_id] = (request, request_path)
    validated_requests = [requests[item][0] for item in validated_ids if item in requests]
    critical_errors = []
    if len(validated_requests) != len(validated_ids):
        critical_errors.append("validated_request_artifact_missing")
    for request_id in validated_ids:
        indexed = requests.get(request_id)
        if indexed is None:
            continue
        request, request_path = indexed
        event = latest[request_id]
        hashes = (
            event.request_sha256,
            event.response_sha256,
            event.dossier_sha256,
            event.overlay_sha256,
        )
        if not all(hashes):
            critical_errors.append(f"validated_artifact_hash_binding_missing:{request_id}")
            continue
        run_dir = request_path.parent.parent
        artifact_paths = (
            request_path,
            run_dir / "responses" / f"{request.symbol}.response.v1.json",
            run_dir / "dossiers" / f"{request.symbol}.dossier.v1.json",
            run_dir / "overlays" / f"{request.symbol}.overlay.v1.json",
        )
        try:
            actual_hashes = tuple(sha256_bytes(path.read_bytes()) for path in artifact_paths)
        except OSError:
            critical_errors.append(f"validated_artifact_missing:{request_id}")
            continue
        if actual_hashes != hashes:
            critical_errors.append(f"validated_artifact_sha256_mismatch:{request_id}")

    application_pairs = [
        (record, ApplicationEventV1.model_validate(record.get("event") or {}))
        for record in application_records
    ]
    application_pairs = [pair for pair in application_pairs if pair[1].occurred_at <= generated_at]
    applications = [pair[1] for pair in application_pairs]
    longitudinal_pairs = [
        (record, LongitudinalObservationV1.model_validate(record.get("event") or {}))
        for record in longitudinal_records
    ]
    longitudinal_pairs = [
        pair for pair in longitudinal_pairs if pair[1].occurred_at <= generated_at
    ]
    longitudinal: list[LongitudinalObservationV1] = []
    for _, observation in longitudinal_pairs:
        try:
            _validate_longitudinal_observation(
                root=root_path, observation=observation, applications=applications
            )
        except Exception as exc:
            critical_errors.append(
                "longitudinal_artifact_invalid:" f"{observation.event_id}:{type(exc).__name__}"
            )
        else:
            longitudinal.append(observation)

    recent_received = list(dict.fromkeys(reversed(received_order)))[:20]
    recent_successes = sum(
        latest.get(request_id) is not None and latest[request_id].state == JobState.VALIDATED
        for request_id in recent_received
    )
    return ActivationGateEvidenceV2(
        generated_at=generated_at,
        job_ledger_head_sha256=_head([pair[0] for pair in job_pairs]),
        application_ledger_head_sha256=_head([pair[0] for pair in application_pairs]),
        longitudinal_ledger_head_sha256=_head([pair[0] for pair in longitudinal_pairs]),
        holdings_snapshot_path=str(snapshot_path.relative_to(root_path)),
        holdings_snapshot_sha256=snapshot_sha,
        validated_request_ids=validated_ids,
        validated_symbols=sorted({item.symbol for item in validated_requests}),
        validated_company_names=sorted({item.company_name for item in validated_requests}),
        validated_industries=sorted(
            {
                item.local_context.industry
                for item in validated_requests
                if item.local_context.industry_status == "confirmed"
            }
        ),
        holdings_symbols=list(snapshot.symbols),
        shadow_trading_dates=sorted(
            {event.run_cutoff.date() for event in applications if event.mode == "shadow"}
        ),
        limited_trading_dates=sorted(
            {event.run_cutoff.date() for event in applications if event.mode == "limited"}
        ),
        target_weight_counterfactual_dates=sorted(
            {
                event.trading_date
                for event in longitudinal
                if event.observation_type == "target_weight"
            }
        ),
        nav_attribution_dates=sorted(
            {
                event.trading_date
                for event in longitudinal
                if event.observation_type == "nav_attribution"
            }
        ),
        recent_received_request_ids=recent_received,
        recent_validation_success_count=recent_successes,
        critical_error_codes=sorted(set(critical_errors)),
    )


def verify_recomputed_evidence(
    *, root: str | Path, evidence: ActivationGateEvidenceV2
) -> list[str]:
    rebuilt = build_activation_gate_evidence(
        root=root,
        holdings_snapshot_path=Path(root) / evidence.holdings_snapshot_path,
        generated_at=evidence.generated_at,
    )
    return [] if rebuilt == evidence else ["gate_evidence_recomputation_mismatch"]


def activation_readiness(evidence: ActivationGateEvidenceV2) -> dict[str, object]:
    """Explain mode eligibility without trusting any caller-supplied pass flag."""

    common = list(evidence.critical_error_codes)
    if not evidence.holdings_coverage_passed:
        common.append("holdings_coverage_incomplete")
    rate = evidence.recent_validation_success_rate
    if len(evidence.recent_received_request_ids) >= 10 and rate is not None and rate < 0.80:
        common.append("recent_validation_success_below_80pct")
    limited = list(common)
    thresholds = {
        "shadow_trading_days": (len(evidence.shadow_trading_dates), 10),
        "validated_dossiers": (len(evidence.validated_request_ids), 30),
        "distinct_companies": (len(evidence.validated_company_names), 10),
        "distinct_industries": (len(evidence.validated_industries), 3),
        "target_weight_counterfactual_days": (
            len(evidence.target_weight_counterfactual_dates),
            10,
        ),
    }
    limited.extend(
        f"{name}_below_{minimum}"
        for name, (actual, minimum) in thresholds.items()
        if actual < minimum
    )
    production = list(common)
    production_thresholds = {
        "shadow_trading_days": (len(evidence.shadow_trading_dates), 20),
        "validated_dossiers": (len(evidence.validated_request_ids), 60),
        "distinct_companies": (len(evidence.validated_company_names), 20),
        "distinct_industries": (len(evidence.validated_industries), 3),
        "limited_trading_days": (len(evidence.limited_trading_dates), 10),
        "target_weight_counterfactual_days": (
            len(evidence.target_weight_counterfactual_dates),
            20,
        ),
        "nav_attribution_days": (len(evidence.nav_attribution_dates), 10),
    }
    production.extend(
        f"{name}_below_{minimum}"
        for name, (actual, minimum) in production_thresholds.items()
        if actual < minimum
    )
    return {
        "eligible_modes": [
            "shadow",
            *([] if limited else ["limited"]),
            *([] if production else ["production"]),
        ],
        "limited_blockers": sorted(set(limited)),
        "production_blockers": sorted(set(production)),
        "recent_validation_success_rate": rate,
    }


def append_longitudinal_observation(
    *, root: str | Path, observation_path: str | Path
) -> dict[str, object]:
    """Validate two private outcome artifacts and append one idempotent event."""

    root_path = Path(root)
    observation = load_json_model(root_path, Path(observation_path), LongitudinalObservationV1)
    application_records = _records(root_path, APPLICATION_LEDGER)
    applications = [
        ApplicationEventV1.model_validate(record.get("event") or {})
        for record in application_records
    ]
    _validate_longitudinal_observation(
        root=root_path, observation=observation, applications=applications
    )

    ledger = HashChainLedger(root_path, root_path / LONGITUDINAL_LEDGER)
    records = ledger.read_records()
    expected_head = _head(records)
    for record in records:
        existing = LongitudinalObservationV1.model_validate(record.get("event") or {})
        if existing.event_id == observation.event_id:
            if existing != observation:
                raise ValueError("longitudinal event id collision")
            return {
                "appended": False,
                "idempotent_replay": True,
                "event_id": observation.event_id,
                "ledger_head_sha256": str(record.get("event_sha256") or ""),
            }
        logical_key = (
            existing.observation_type,
            existing.request_id,
            existing.dossier_id,
            existing.run_key,
            existing.trading_date,
        )
        incoming_key = (
            observation.observation_type,
            observation.request_id,
            observation.dossier_id,
            observation.run_key,
            observation.trading_date,
        )
        if logical_key == incoming_key:
            raise ValueError("longitudinal logical observation already exists")
    try:
        event_sha = ledger.append(observation, expected_head=expected_head)
    except LedgerConflictError as exc:
        raise ValueError("longitudinal ledger concurrent update") from exc
    return {
        "appended": True,
        "idempotent_replay": False,
        "event_id": observation.event_id,
        "ledger_head_sha256": event_sha,
    }


__all__ = [
    "ActivationGateEvidenceV2",
    "CanonicalLongitudinalMetricV1",
    "CanonicalNavSnapshotV1",
    "HoldingsScopeSnapshotV1",
    "LongitudinalObservationV1",
    "LongitudinalOutcomeArtifactV1",
    "LongitudinalSourceArtifactV1",
    "LONGITUDINAL_LEDGER",
    "build_activation_gate_evidence",
    "activation_readiness",
    "append_longitudinal_observation",
    "verify_recomputed_evidence",
    "verify_holdings_scope_snapshot",
]
