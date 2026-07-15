"""Fail-closed runtime consumption of validated fundamental research overlays."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping
from zoneinfo import ZoneInfo

from pydantic import Field, StringConstraints, field_validator, model_validator

from .governance import (
    ActivationGateEvidenceV2,
    HoldingsScopeSnapshotV1,
    activation_readiness,
    build_activation_gate_evidence,
    verify_recomputed_evidence,
)
from .ledger import HashChainLedger, LedgerConflictError
from .models import (
    ApplicationEventV1,
    ApplicationState,
    FundamentalOverlayV1,
    FundamentalResearchDossierV1,
    FundamentalResearchRequestV1,
    FundamentalResearchResponseV1,
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

DEFAULT_ROOT = "results/fundamental_research"
DEFAULT_MODE = "shadow"
APPLICATION_LEDGER = "state/applications.v1.jsonl"
JOB_LEDGER = "state/jobs.v1.jsonl"


class ActivationManifestV1(StrictModel):
    """Hash-pinned, human-approved gate for a score-affecting runtime mode."""

    schema_version: Literal["fundamental-research-activation.v1"] = (
        "fundamental-research-activation.v1"
    )
    strategy_version: Literal["v14-fundamental-research"] = "v14-fundamental-research"
    mode: Literal["limited", "production"]
    phase: Literal["limited_phase_1", "limited_phase_2", "production"]
    effective_from: datetime
    approved_by: Literal["maxwell"]
    approved_at: datetime
    approval_id: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9_.:-]+$",
        ),
    ]
    separate_confirmation: Literal[True]
    shadow_gates_passed: Literal[True]
    gate_evidence_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]
    gate_evidence_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    holdings_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    holdings_ledger_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    shadow_trading_days: int = Field(ge=0)
    validated_dossiers: int = Field(ge=0)
    distinct_companies: int = Field(ge=0)
    distinct_industries: int = Field(ge=0)
    holdings_coverage_passed: Literal[True]
    limited_trading_days: int = Field(ge=0)
    target_weight_counterfactual_days: int = Field(ge=0)
    nav_attribution_days: int = Field(ge=0)
    critical_error_count: Literal[0] = 0
    max_abs_delta: float = Field(gt=0.0, le=0.1)
    status: Literal["active"] = "active"

    @field_validator("effective_from", "approved_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("activation timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_mode_cap(self) -> "ActivationManifestV1":
        if self.mode == "limited" and self.phase not in {
            "limited_phase_1",
            "limited_phase_2",
        }:
            raise ValueError("limited activation requires a limited phase")
        if self.mode == "production" and self.phase != "production":
            raise ValueError("production activation requires phase=production")
        if self.phase == "limited_phase_1" and (
            self.limited_trading_days > 4 or self.max_abs_delta > 0.03
        ):
            raise ValueError("limited phase 1 requires days 0..4 and cap <= 0.03")
        if self.phase == "limited_phase_2" and (
            self.limited_trading_days < 5 or self.max_abs_delta > 0.05
        ):
            raise ValueError("limited phase 2 requires at least 5 days and cap <= 0.05")
        if self.mode == "limited" and (
            self.shadow_trading_days < 10
            or self.validated_dossiers < 30
            or self.distinct_companies < 10
            or self.distinct_industries < 3
            or self.target_weight_counterfactual_days < 10
        ):
            raise ValueError("limited activation evidence gates are not satisfied")
        if self.mode == "production" and (
            self.shadow_trading_days < 20
            or self.validated_dossiers < 60
            or self.distinct_companies < 20
            or self.distinct_industries < 3
            or self.limited_trading_days < 10
            or self.target_weight_counterfactual_days < 20
            or self.nav_attribution_days < 10
        ):
            raise ValueError("production activation evidence gates are not satisfied")
        return self


@dataclass(frozen=True)
class RuntimeOverlayDecision:
    requested_mode: str
    effective_mode: str
    applied: bool = False
    suppress_generic_fundamental_overlay: bool = False
    adjusted_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _run_cutoff(value: Any) -> datetime:
    """Normalize a date-only DAG cutoff to the CN 15:00 decision boundary.

    Only a request with a strictly earlier decision cutoff and a validation
    event before this instant can be consumed.
    """

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime cutoff datetime must be timezone-aware")
        return value
    raw = str(value or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError("runtime cutoff must be YYYYMMDD or a timezone-aware datetime")
    parsed = datetime.strptime(raw, "%Y%m%d").date()
    return datetime.combine(parsed, time(hour=15), tzinfo=ZoneInfo("Asia/Shanghai"))


def _runtime_mode(env: Mapping[str, str]) -> str:
    mode = str(env.get("FUNDAMENTAL_RESEARCH_OVERLAY_MODE", DEFAULT_MODE)).strip().lower()
    return mode if mode in {"off", "shadow", "limited", "production"} else "off"


def _resolve_root(env: Mapping[str, str]) -> Path:
    return Path(env.get("FUNDAMENTAL_RESEARCH_ROOT", DEFAULT_ROOT)).absolute()


def _activation(
    *, root: Path, mode: str, cutoff: datetime, env: Mapping[str, str]
) -> tuple[ActivationManifestV1 | None, str, list[str]]:
    if mode not in {"limited", "production"}:
        return None, "", []
    path_raw = str(env.get("FUNDAMENTAL_RESEARCH_ACTIVATION_PATH", "")).strip()
    expected = str(env.get("FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256", "")).strip().lower()
    blockers: list[str] = []
    if not path_raw:
        blockers.append("activation_path_missing")
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        blockers.append("activation_expected_sha256_missing_or_invalid")
    if blockers:
        return None, "", blockers
    try:
        path = Path(path_raw)
        if not path.is_absolute():
            path = root / path
        manifest = load_json_model(root, path, ActivationManifestV1)
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            return None, actual, ["activation_sha256_mismatch"]
    except Exception as exc:  # untrusted local activation input
        return None, "", [f"activation_invalid:{type(exc).__name__}"]
    if manifest.mode != mode:
        blockers.append("activation_mode_mismatch")
    if cutoff < manifest.effective_from:
        blockers.append("activation_not_yet_effective")
    if manifest.approved_at > manifest.effective_from:
        blockers.append("activation_approval_after_effective_from")
    evidence_path = Path(manifest.gate_evidence_path)
    if evidence_path.is_absolute():
        blockers.append("gate_evidence_path_must_be_relative")
    else:
        try:
            _, evidence_target = _assert_contained(
                root,
                root / evidence_path,
                allow_missing_leaf=False,
                create_parents=False,
            )
            if evidence_target.stat().st_mode & 0o777 != PRIVATE_MODE:
                raise ValueError("gate evidence permissions are not 0600")
            if sha256_bytes(evidence_target.read_bytes()) != manifest.gate_evidence_sha256:
                raise ValueError("gate evidence hash mismatch")
            evidence = load_json_model(root, evidence_target, ActivationGateEvidenceV2)
            snapshot = load_json_model(
                root,
                root / evidence.holdings_snapshot_path,
                HoldingsScopeSnapshotV1,
            )
            if (
                snapshot.manual_manifest_sha256 != manifest.holdings_manifest_sha256
                or snapshot.manual_ledger_sha256 != manifest.holdings_ledger_sha256
            ):
                raise ValueError("activation holdings lineage mismatch")
            blockers.extend(verify_recomputed_evidence(root=root, evidence=evidence))
            current_evidence = build_activation_gate_evidence(
                root=root,
                holdings_snapshot_path=root / evidence.holdings_snapshot_path,
                generated_at=cutoff,
            )
            readiness = activation_readiness(current_evidence)
            raw_current_blockers = readiness.get(f"{mode}_blockers", [])
            current_blockers = (
                [str(item) for item in raw_current_blockers]
                if isinstance(raw_current_blockers, list)
                else ["activation_readiness_invalid"]
            )
            blockers.extend(
                (
                    f"gate_critical_error:{code}"
                    if code in current_evidence.critical_error_codes
                    else code
                )
                for code in current_blockers
            )
            evidence_counts = (
                len(evidence.shadow_trading_dates),
                len(evidence.validated_request_ids),
                len(evidence.validated_company_names),
                len(evidence.validated_industries),
                evidence.holdings_coverage_passed,
                len(evidence.limited_trading_dates),
                len(evidence.target_weight_counterfactual_dates),
                len(evidence.nav_attribution_dates),
                len(evidence.critical_error_codes),
            )
            manifest_counts = (
                manifest.shadow_trading_days,
                manifest.validated_dossiers,
                manifest.distinct_companies,
                manifest.distinct_industries,
                manifest.holdings_coverage_passed,
                manifest.limited_trading_days,
                manifest.target_weight_counterfactual_days,
                manifest.nav_attribution_days,
                manifest.critical_error_count,
            )
            if evidence_counts != manifest_counts:
                raise ValueError("gate evidence counts mismatch")
            if evidence.generated_at > manifest.approved_at:
                raise ValueError("gate evidence postdates approval")
        except Exception as exc:
            blockers.append(f"gate_evidence_invalid:{type(exc).__name__}")
    return (manifest if not blockers else None), actual, blockers


def _latest_job_states(root: Path, *, cutoff: datetime) -> dict[str, str]:
    return {
        request_id: event.state.value
        for request_id, event in _latest_job_events(root, cutoff=cutoff).items()
    }


def _latest_job_events(root: Path, *, cutoff: datetime) -> dict[str, JobEventV1]:
    ledger_path = root / JOB_LEDGER
    if not ledger_path.exists():
        return {}
    records = HashChainLedger(root, ledger_path).read_records()
    result: dict[str, JobEventV1] = {}
    for record in records:
        try:
            event = JobEventV1.model_validate(record.get("event") or {})
        except Exception:
            continue
        if event.occurred_at < cutoff:
            result[event.request_id] = event
    return result


def _application_keys(root: Path) -> set[tuple[str, str, str]]:
    return set(_application_records(root))


def _application_records(
    root: Path,
) -> dict[tuple[str, str, str], tuple[ApplicationEventV1, str]]:
    ledger_path = root / APPLICATION_LEDGER
    if not ledger_path.exists():
        return {}
    records = HashChainLedger(root, ledger_path).read_records()
    result: dict[tuple[str, str, str], tuple[ApplicationEventV1, str]] = {}
    for record in records:
        event = ApplicationEventV1.model_validate(record.get("event") or {})
        key = (event.request_id, event.dossier_id, event.run_key)
        if key in result:
            raise ValueError("duplicate application key in ledger")
        result[key] = (event, str(record.get("event_sha256") or ""))
    return result


def _artifact_candidates(
    *,
    root: Path,
    symbol: str,
    cutoff: datetime,
    run_key: str,
    current_data_generation: str,
) -> tuple[
    list[
        tuple[
            FundamentalResearchRequestV1,
            FundamentalOverlayV1,
            FundamentalResearchDossierV1,
        ]
    ],
    list[str],
]:
    if not root.exists():
        return [], []
    job_events = _latest_job_events(root, cutoff=cutoff)
    candidates: list[
        tuple[
            FundamentalResearchRequestV1,
            FundamentalOverlayV1,
            FundamentalResearchDossierV1,
        ]
    ] = []
    blockers: set[str] = set()
    for overlay_path in sorted(root.glob("CN/*/*/overlays/*.overlay.v1.json")):
        try:
            overlay = load_json_model(root, overlay_path, FundamentalOverlayV1)
            if overlay.symbol != symbol or not overlay.eligible:
                continue
            validation_event = job_events.get(overlay.request_id)
            if validation_event is None or validation_event.state != JobState.VALIDATED:
                blockers.add("job_not_validated_at_run_cutoff")
                continue
            if not all(
                (
                    validation_event.request_sha256,
                    validation_event.response_sha256,
                    validation_event.dossier_sha256,
                    validation_event.overlay_sha256,
                )
            ):
                blockers.add("validated_artifact_hash_binding_missing")
                continue
            if sha256_bytes(overlay_path.read_bytes()) != validation_event.overlay_sha256:
                blockers.add("validated_overlay_sha256_mismatch")
                continue
            request_paths = list(
                overlay_path.parent.parent.joinpath("requests").glob("*.request.v1.json")
            )
            request = None
            for path in request_paths:
                if not path.is_file():
                    continue
                try:
                    item = load_json_model(root, path, FundamentalResearchRequestV1)
                except Exception:
                    continue
                if item.request_id == overlay.request_id:
                    request = item
                    break
            if request is None:
                blockers.add("request_artifact_missing_or_invalid")
                continue
            if sha256_bytes(path.read_bytes()) != validation_event.request_sha256:
                blockers.add("validated_request_sha256_mismatch")
                continue
            dossier_path = overlay_path.parent.parent / "dossiers" / f"{symbol}.dossier.v1.json"
            response_path = overlay_path.parent.parent / "responses" / f"{symbol}.response.v1.json"
            dossier = load_json_model(root, dossier_path, FundamentalResearchDossierV1)
            response = load_json_model(root, response_path, FundamentalResearchResponseV1)
            if sha256_bytes(dossier_path.read_bytes()) != validation_event.dossier_sha256:
                blockers.add("validated_dossier_sha256_mismatch")
                continue
            if sha256_bytes(response_path.read_bytes()) != validation_event.response_sha256:
                blockers.add("validated_response_sha256_mismatch")
                continue
            if (
                dossier.request_id != request.request_id
                or dossier.dossier_id != overlay.dossier_id
                or response.request_id != request.request_id
                or response.dossier != dossier
            ):
                blockers.add("validated_artifact_lineage_mismatch")
                continue
            if request.symbol != symbol or request.market != "CN":
                blockers.add("request_identity_mismatch")
                continue
            if (
                request.decision_cutoff >= cutoff
                or request.created_at >= cutoff
                or request.expires_at < cutoff
            ):
                blockers.add("request_not_valid_at_run_cutoff")
                continue
            if request.data_generation != current_data_generation:
                blockers.add("fundamental_data_generation_mismatch")
                continue
            if abs(request.base_score - overlay.base_score) > 1e-12:
                blockers.add("request_overlay_base_score_mismatch")
                continue
            expected_adjusted = max(-1.0, min(1.0, overlay.base_score + overlay.computed_delta))
            if abs(expected_adjusted - overlay.adjusted_score) > 1e-12:
                blockers.add("overlay_adjusted_score_invalid")
                continue
            candidates.append((request, overlay, dossier))
        except Exception:
            # One malformed or hostile artifact cannot affect another symbol.
            continue
    return (
        sorted(
            candidates,
            key=lambda item: (
                item[0].decision_cutoff,
                item[0].created_at,
                item[0].request_id,
            ),
            reverse=True,
        ),
        sorted(blockers),
    )


def _dossier_summary(dossier: FundamentalResearchDossierV1) -> dict[str, Any]:
    claims = {claim.claim_id: claim for claim in dossier.claims}

    def statements(ids: list[str]) -> list[str]:
        return [claims[item].statement for item in ids if item in claims]

    return {
        "dimensions": {
            item.dimension.value: {
                "signal": item.signal.value,
                "unknowns": list(item.unknowns),
            }
            for item in dossier.dimensions
        },
        "bull_case": statements(dossier.bull_case),
        "bear_case": statements(dossier.bear_case),
        "key_risks": statements(dossier.key_risks),
        "catalysts": statements(dossier.catalysts),
        "unknowns": list(dossier.unknowns),
    }


def _append_application(*, root: Path, event: ApplicationEventV1) -> tuple[bool, str]:
    ledger = HashChainLedger(root, root / APPLICATION_LEDGER)
    for _ in range(3):
        try:
            return True, ledger.append(event, expected_head=ledger.head())
        except LedgerConflictError:
            if (event.request_id, event.dossier_id, event.run_key) in _application_keys(root):
                return False, ""
    return False, ""


def consume_overlay(
    *,
    symbol: str,
    base_score: float,
    run_cutoff: Any,
    run_key: str,
    current_data_generation: str,
    market: str = "CN",
    env: Mapping[str, str] | None = None,
    occurred_at: datetime | None = None,
) -> RuntimeOverlayDecision:
    """Consume at most one validated prior-run overlay for ``symbol``.

    Score-affecting modes are fail-closed unless a separately confirmed,
    hash-bound activation manifest is valid. Shadow records only the
    counterfactual; off performs no artifact mutation.
    """

    runtime_env = os.environ if env is None else env
    requested_mode = _runtime_mode(runtime_env)
    audit: dict[str, Any] = {
        "schema_version": "fundamental-research-runtime-audit.v1",
        "requested_mode": requested_mode,
        "effective_mode": requested_mode,
        "symbol": symbol,
        "market": str(market).upper(),
        "run_key": run_key,
        "current_data_generation": current_data_generation,
        "deterministic_base_score": float(base_score),
        "applied": False,
        "counterfactual": False,
        "generic_fundamental_overlay_suppressed": False,
        "blockers": [],
    }
    if requested_mode == "off":
        return RuntimeOverlayDecision("off", "off", metadata=audit)
    if str(market).upper() != "CN":
        audit["effective_mode"] = "off"
        audit["blockers"] = ["market_not_supported"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    try:
        cutoff = _run_cutoff(run_cutoff)
    except ValueError as exc:
        audit["effective_mode"] = "off"
        audit["blockers"] = [str(exc)]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    if not str(run_key or "").strip():
        audit["effective_mode"] = "off"
        audit["blockers"] = ["run_key_missing"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    if not str(current_data_generation or "").strip():
        audit["effective_mode"] = "off"
        audit["blockers"] = ["current_data_generation_missing"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)

    root = _resolve_root(runtime_env)
    activation, activation_sha, blockers = _activation(
        root=root, mode=requested_mode, cutoff=cutoff, env=runtime_env
    )
    effective_mode = requested_mode
    if blockers:
        fallback_blockers = {"recent_validation_success_below_80pct"}
        fallback_to_shadow = bool(blockers) and set(blockers) <= fallback_blockers
        effective_mode = "shadow" if fallback_to_shadow else "off"
        audit.update(
            {
                "effective_mode": effective_mode,
                "activation_sha256": activation_sha,
                "blockers": blockers,
                "activation_blockers": list(blockers),
                "automatic_shadow_fallback": fallback_to_shadow,
            }
        )
        if not fallback_to_shadow:
            return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
        activation = None

    try:
        candidates, candidate_blockers = _artifact_candidates(
            root=root,
            symbol=symbol,
            cutoff=cutoff,
            run_key=run_key,
            current_data_generation=current_data_generation,
        )
    except Exception as exc:
        audit["effective_mode"] = "off"
        audit["blockers"] = [f"artifact_state_invalid:{type(exc).__name__}"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    if not candidates:
        audit["blockers"] = candidate_blockers or ["no_eligible_prior_run_overlay"]
        return RuntimeOverlayDecision(requested_mode, effective_mode, metadata=audit)
    request, overlay, dossier = candidates[0]
    dossier_summary = _dossier_summary(dossier)
    if abs(float(base_score) - overlay.base_score) > 1e-12:
        audit.update(
            {
                "request_id": request.request_id,
                "dossier_id": overlay.dossier_id,
                "blockers": ["runtime_base_score_mismatch"],
            }
        )
        return RuntimeOverlayDecision(requested_mode, effective_mode, metadata=audit)

    cap = 0.1 if activation is None else activation.max_abs_delta
    delta = max(-cap, min(cap, overlay.computed_delta))
    adjusted = max(-1.0, min(1.0, float(base_score) + delta))
    state = {
        "shadow": ApplicationState.SHADOW_EVALUATED,
        "limited": ApplicationState.LIMITED_APPLIED,
        "production": ApplicationState.PRODUCTION_APPLIED,
    }[effective_mode]
    application_key = (request.request_id, overlay.dossier_id, run_key)

    def replay_existing(
        existing: tuple[ApplicationEventV1, str],
    ) -> RuntimeOverlayDecision:
        previous, previous_sha = existing
        if (
            previous.mode != effective_mode
            or previous.state != state
            or abs(previous.base_score - float(base_score)) > 1e-12
            or abs(previous.computed_delta - delta) > 1e-12
            or abs(previous.adjusted_score - adjusted) > 1e-12
            or previous.run_cutoff != cutoff
        ):
            audit["effective_mode"] = "off"
            audit["blockers"] = ["application_replay_mismatch"]
            return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
        applies = effective_mode in {"limited", "production"}
        audit.update(
            {
                "request_id": request.request_id,
                "dossier_id": overlay.dossier_id,
                "request_decision_cutoff": request.decision_cutoff.isoformat(),
                "application_event_sha256": previous_sha,
                "activation_sha256": activation_sha,
                "dossier_summary": dossier_summary,
                "computed_delta": delta,
                "counterfactual_adjusted_score": adjusted,
                "counterfactual": effective_mode == "shadow",
                "applied": applies,
                "generic_fundamental_overlay_suppressed": applies,
                "idempotent_replay": True,
                "blockers": [],
            }
        )
        return RuntimeOverlayDecision(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            applied=applies,
            suppress_generic_fundamental_overlay=applies,
            adjusted_score=adjusted if applies else None,
            metadata=audit,
        )

    try:
        existing_application = _application_records(root).get(application_key)
    except Exception as exc:
        audit["effective_mode"] = "off"
        audit["blockers"] = [f"application_ledger_invalid:{type(exc).__name__}"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    if existing_application is not None:
        return replay_existing(existing_application)

    now = occurred_at or datetime.now(timezone.utc)
    application_identity = {
        "request_id": request.request_id,
        "dossier_id": overlay.dossier_id,
        "run_key": run_key,
    }
    event = ApplicationEventV1(
        event_id=f"app:{sha256_bytes(canonical_json_bytes(application_identity))}",
        request_id=request.request_id,
        dossier_id=overlay.dossier_id,
        run_key=run_key,
        run_cutoff=cutoff,
        state=state,
        occurred_at=now,
        mode=effective_mode,
        base_score=float(base_score),
        computed_delta=delta,
        adjusted_score=adjusted,
    )
    try:
        appended, event_sha = _append_application(root=root, event=event)
    except Exception as exc:
        audit["effective_mode"] = "off"
        audit["blockers"] = [f"application_ledger_invalid:{type(exc).__name__}"]
        return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
    audit.update(
        {
            "effective_mode": effective_mode,
            "request_id": request.request_id,
            "dossier_id": overlay.dossier_id,
            "request_decision_cutoff": request.decision_cutoff.isoformat(),
            "application_event_sha256": event_sha,
            "activation_sha256": activation_sha,
            "dossier_summary": dossier_summary,
            "computed_delta": delta,
            "counterfactual_adjusted_score": adjusted,
            "counterfactual": effective_mode == "shadow" and appended,
            "applied": effective_mode in {"limited", "production"} and appended,
            "generic_fundamental_overlay_suppressed": effective_mode in {"limited", "production"}
            and appended,
            "blockers": [] if appended else ["overlay_already_consumed"],
        }
    )
    if not appended:
        try:
            concurrent_application = _application_records(root).get(application_key)
        except Exception as exc:
            audit["effective_mode"] = "off"
            audit["blockers"] = [f"application_ledger_invalid:{type(exc).__name__}"]
            return RuntimeOverlayDecision(requested_mode, "off", metadata=audit)
        if concurrent_application is not None:
            return replay_existing(concurrent_application)
        return RuntimeOverlayDecision(requested_mode, effective_mode, metadata=audit)
    applies = effective_mode in {"limited", "production"}
    return RuntimeOverlayDecision(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        applied=applies,
        suppress_generic_fundamental_overlay=applies,
        adjusted_score=adjusted if applies else None,
        metadata=audit,
    )
