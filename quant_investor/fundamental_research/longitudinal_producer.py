"""Deterministic producers for target-weight and realized NAV comparisons."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import governance as governance_module
from .governance import (
    APPLICATION_LEDGER,
    CanonicalLongitudinalMetricV1,
    CanonicalNavSnapshotV1,
    LongitudinalObservationV1,
    LongitudinalOutcomeArtifactV1,
    LongitudinalSourceArtifactV1,
    append_longitudinal_observation,
)
from .ledger import HashChainLedger
from .models import ApplicationEventV1, ApplicationState, FundamentalResearchRequestV1
from .storage import (
    PRIVATE_MODE,
    atomic_write_json_model,
    canonical_json_bytes,
    load_json_model,
    sha256_bytes,
)


class LongitudinalProducerError(ValueError):
    """A canonical producer input failed lineage or PIT validation."""


def _repo_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    root = governance_module.REPO_ROOT.resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise LongitudinalProducerError(
            "canonical analysis manifest must be inside the repository"
        ) from exc


def _load_analysis_manifest(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    target = Path(path).resolve(strict=True)
    _repo_relative(target)
    if target.is_symlink() or target.stat().st_mode & 0o777 != PRIVATE_MODE:
        raise LongitudinalProducerError(
            "analysis manifest must be a regular mode-0600 repository file"
        )
    payload_bytes = target.read_bytes()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LongitudinalProducerError("analysis manifest is invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "analysis-run-manifest.v1":
        raise LongitudinalProducerError("analysis manifest schema mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if str(payload.get("manifest_sha256", "")).lower() != sha256_bytes(
        canonical_json_bytes(unsigned)
    ):
        raise LongitudinalProducerError("analysis manifest self-hash mismatch")
    meta = payload.get("analysis_meta")
    if not isinstance(meta, dict) or str(
        payload.get("analysis_meta_sha256", "")
    ).lower() != sha256_bytes(canonical_json_bytes(meta)):
        raise LongitudinalProducerError("analysis manifest metadata hash mismatch")
    return payload, target, sha256_bytes(payload_bytes)


def _manifest_facts(payload: dict[str, Any], *, symbol: str) -> dict[str, Any]:
    meta = dict(payload.get("analysis_meta") or {})
    variant = str(meta.get("fundamental_research_variant") or "")
    if variant not in {"with_dossier", "without_dossier"}:
        raise LongitudinalProducerError("analysis manifest dossier variant is missing")
    portfolio = meta.get("portfolio_decision")
    if not isinstance(portfolio, dict) or not isinstance(portfolio.get("target_weights"), dict):
        raise LongitudinalProducerError("analysis manifest target weights are missing")
    weights = {
        str(key).strip().upper(): float(value)
        for key, value in dict(portfolio["target_weights"]).items()
        if str(key).strip()
    }
    snapshot = meta.get("data_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    global_context = meta.get("global_context")
    global_context = global_context if isinstance(global_context, dict) else {}
    nested = global_context.get("metadata")
    nested = nested if isinstance(nested, dict) else {}
    nested_snapshot = nested.get("data_snapshot")
    nested_snapshot = nested_snapshot if isinstance(nested_snapshot, dict) else {}
    generation = str(snapshot.get("snapshot_id") or nested_snapshot.get("snapshot_id") or "")
    raw_date = str(
        snapshot.get("local_latest_trade_date")
        or snapshot.get("latest_complete_trade_date")
        or global_context.get("latest_trade_date")
        or ""
    )
    digits = "".join(ch for ch in raw_date if ch.isdigit())[:8]
    if len(digits) != 8 or not generation:
        raise LongitudinalProducerError("analysis manifest snapshot lineage is incomplete")
    trading_date = date.fromisoformat(f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}")
    universe = str(global_context.get("universe_key") or meta.get("universe") or "full_a")
    produced_at = datetime.fromisoformat(str(payload.get("generated_at") or ""))
    if produced_at.tzinfo is None or produced_at.utcoffset() is None:
        raise LongitudinalProducerError("analysis manifest generated_at is invalid")
    return {
        "variant": variant,
        "weights": weights,
        "value": float(weights.get(symbol, 0.0)),
        "generation": generation,
        "trading_date": trading_date,
        "run_key": f"CN:{universe}:{digits}",
        "produced_at": produced_at,
    }


def _application(
    *, root: Path, request_id: str, dossier_id: str, run_key: str
) -> ApplicationEventV1:
    path = root / APPLICATION_LEDGER
    if not path.exists():
        raise LongitudinalProducerError("application ledger is missing")
    matches = []
    for record in HashChainLedger(root, path).read_records():
        event = ApplicationEventV1.model_validate(record.get("event") or {})
        if (
            event.request_id == request_id
            and event.dossier_id == dossier_id
            and event.run_key == run_key
        ):
            matches.append(event)
    if len(matches) != 1:
        raise LongitudinalProducerError("exactly one matching application is required")
    return matches[0]


def _request(root: Path, request_path: str | Path) -> FundamentalResearchRequestV1:
    path = Path(request_path)
    if not path.is_absolute():
        path = root / path
    return load_json_model(root, path, FundamentalResearchRequestV1)


def _immutable_write(root: Path, path: Path, model: Any) -> str:
    if path.exists():
        existing = load_json_model(root, path, type(model))
        if existing != model:
            raise LongitudinalProducerError(f"immutable artifact conflict: {path.name}")
        return sha256_bytes(path.read_bytes())
    return atomic_write_json_model(root, path, model)


def _write_comparison(
    *,
    root: Path,
    request: FundamentalResearchRequestV1,
    dossier_id: str,
    application: ApplicationEventV1,
    observation_type: Literal["target_weight", "nav_attribution"],
    trading_date: date,
    occurred_at: datetime,
    actual_source: LongitudinalSourceArtifactV1,
    counterfactual_source: LongitudinalSourceArtifactV1,
) -> dict[str, Any]:
    base = root / "longitudinal" / trading_date.isoformat() / request.request_id / observation_type
    actual_source_path = base / "actual.source.v1.json"
    counter_source_path = base / "counterfactual.source.v1.json"
    actual_source_sha = _immutable_write(root, actual_source_path, actual_source)
    counter_source_sha = _immutable_write(root, counter_source_path, counterfactual_source)
    actual = LongitudinalOutcomeArtifactV1(
        observation_type=observation_type,
        variant="actual",
        run_key=application.run_key,
        trading_date=trading_date,
        value=actual_source.value,
        produced_at=actual_source.produced_at,
        source_generation=actual_source.generation,
        source_kind=actual_source.source_kind,
        source_artifact_path=actual_source_path.relative_to(root).as_posix(),
        source_artifact_sha256=actual_source_sha,
    )
    counterfactual = LongitudinalOutcomeArtifactV1(
        observation_type=observation_type,
        variant="counterfactual",
        run_key=application.run_key,
        trading_date=trading_date,
        value=counterfactual_source.value,
        produced_at=counterfactual_source.produced_at,
        source_generation=counterfactual_source.generation,
        source_kind=counterfactual_source.source_kind,
        source_artifact_path=counter_source_path.relative_to(root).as_posix(),
        source_artifact_sha256=counter_source_sha,
    )
    actual_path = base / "actual.outcome.v1.json"
    counter_path = base / "counterfactual.outcome.v1.json"
    actual_sha = _immutable_write(root, actual_path, actual)
    counter_sha = _immutable_write(root, counter_path, counterfactual)
    observation = LongitudinalObservationV1(
        event_id=f"longitudinal:{observation_type}:{request.request_id}:{trading_date.isoformat()}",
        request_id=request.request_id,
        dossier_id=dossier_id,
        observation_type=observation_type,
        run_key=application.run_key,
        trading_date=trading_date,
        application_trading_date=application.run_cutoff.date(),
        occurred_at=occurred_at,
        actual_artifact_path=actual_path.relative_to(root).as_posix(),
        counterfactual_artifact_path=counter_path.relative_to(root).as_posix(),
        actual_artifact_sha256=actual_sha,
        counterfactual_artifact_sha256=counter_sha,
        actual_value=actual.value,
        counterfactual_value=counterfactual.value,
    )
    observation_path = base / "observation.v1.json"
    observation_sha = _immutable_write(root, observation_path, observation)
    appended = append_longitudinal_observation(root=root, observation_path=observation_path)
    return {
        **appended,
        "observation_type": observation_type,
        "observation_path": observation_path.relative_to(root).as_posix(),
        "observation_sha256": observation_sha,
        "actual_value": actual.value,
        "counterfactual_value": counterfactual.value,
        "trading_date": trading_date.isoformat(),
    }


def produce_target_weight_observation(
    *,
    root: str | Path,
    request_path: str | Path,
    dossier_id: str,
    actual_analysis_manifest: str | Path,
    counterfactual_analysis_manifest: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Import the two real control-chain manifests and append one weight comparison."""

    root_path = Path(root)
    request = _request(root_path, request_path)
    actual_payload, actual_path, actual_sha = _load_analysis_manifest(actual_analysis_manifest)
    counter_payload, counter_path, counter_sha = _load_analysis_manifest(
        counterfactual_analysis_manifest
    )
    counter_meta = dict(counter_payload.get("analysis_meta") or {})
    if counter_meta.get(
        "fundamental_research_source_manifest_sha256"
    ) != actual_sha or counter_meta.get("fundamental_research_source_run_id") != actual_payload.get(
        "run_id"
    ):
        raise LongitudinalProducerError(
            "counterfactual manifest is not the companion of the actual run"
        )
    actual = _manifest_facts(actual_payload, symbol=request.symbol)
    counter = _manifest_facts(counter_payload, symbol=request.symbol)
    if (
        actual["run_key"] != counter["run_key"]
        or actual["trading_date"] != counter["trading_date"]
        or actual["generation"] != counter["generation"]
        or actual["variant"] == counter["variant"]
    ):
        raise LongitudinalProducerError("analysis manifest pair lineage mismatch")
    application = _application(
        root=root_path,
        request_id=request.request_id,
        dossier_id=dossier_id,
        run_key=str(actual["run_key"]),
    )
    if application.state not in {
        ApplicationState.SHADOW_EVALUATED,
        ApplicationState.LIMITED_APPLIED,
        ApplicationState.PRODUCTION_APPLIED,
    }:
        raise LongitudinalProducerError("application state is not eligible")
    expected_actual = (
        "without_dossier"
        if application.state == ApplicationState.SHADOW_EVALUATED
        else "with_dossier"
    )
    if actual["variant"] != expected_actual:
        raise LongitudinalProducerError("actual manifest does not match application mode")
    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise LongitudinalProducerError("now must be timezone-aware")
    if observed_at < max(actual["produced_at"], counter["produced_at"]):
        raise LongitudinalProducerError("observation cannot predate analysis manifests")
    common = {
        "source_kind": "analysis_run_manifest",
        "run_key": application.run_key,
        "trading_date": actual["trading_date"],
        "generation": actual["generation"],
        "symbol": request.symbol,
    }
    return _write_comparison(
        root=root_path,
        request=request,
        dossier_id=dossier_id,
        application=application,
        observation_type="target_weight",
        trading_date=actual["trading_date"],
        occurred_at=observed_at,
        actual_source=LongitudinalSourceArtifactV1(
            **common,
            variant="actual",
            dossier_variant=actual["variant"],
            produced_at=actual["produced_at"],
            value=actual["value"],
            canonical_artifact_path=_repo_relative(actual_path),
            canonical_artifact_sha256=actual_sha,
        ),
        counterfactual_source=LongitudinalSourceArtifactV1(
            **common,
            variant="counterfactual",
            dossier_variant=counter["variant"],
            produced_at=counter["produced_at"],
            value=counter["value"],
            canonical_artifact_path=_repo_relative(counter_path),
            canonical_artifact_sha256=counter_sha,
        ),
    )


def _portfolio_return(
    *,
    weights: dict[str, float],
    prior_trading_date: date,
    trading_date: date,
    data_root: str | Path,
) -> tuple[float, dict[str, float], str]:
    from quant_investor.market.market_data_reader import MarketDataReader

    reader = MarketDataReader(market="CN", data_root=data_root, mode_policy="strict")
    snapshot = reader.snapshot()
    if not snapshot.get("healthy"):
        raise LongitudinalProducerError("strict Parquet snapshot is not healthy")
    returns: dict[str, float] = {}
    end = trading_date.strftime("%Y%m%d")
    for symbol, weight in weights.items():
        if abs(float(weight)) <= 1e-12:
            continue
        result = reader.read_symbol_frame(symbol, end_date=end)
        frame = result.frame.copy()
        if result.issues or len(frame) < 2:
            raise LongitudinalProducerError(f"insufficient canonical bars for {symbol}")
        date_column = "trade_date" if "trade_date" in frame.columns else "date"
        frame["_date"] = frame[date_column].astype(str).str.replace("-", "", regex=False).str[:8]
        frame = frame.sort_values("_date")
        if str(frame.iloc[-1]["_date"]) != end:
            raise LongitudinalProducerError(f"canonical bar missing on {end}: {symbol}")
        if str(frame.iloc[-2]["_date"]) != prior_trading_date.strftime("%Y%m%d"):
            raise LongitudinalProducerError(
                f"attribution is not the next realized session for {symbol}"
            )
        if "adj_close" in frame.columns:
            prices = frame["adj_close"].astype(float)
        elif "adj_factor" in frame.columns:
            prices = frame["close"].astype(float) * frame["adj_factor"].astype(float)
        else:
            prices = frame["close"].astype(float)
        prior = float(prices.iloc[-2])
        current = float(prices.iloc[-1])
        if prior <= 0:
            raise LongitudinalProducerError(f"invalid prior close for {symbol}")
        returns[symbol] = current / prior - 1.0
    value = sum(float(weights.get(symbol, 0.0)) * value for symbol, value in returns.items())
    return value, returns, str(snapshot.get("snapshot_id") or "")


def produce_nav_attribution_observation(
    *,
    root: str | Path,
    target_weight_observation: str | Path,
    attribution_date: str | date,
    data_root: str | Path = "data",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Roll both canonical target portfolios through the next realized CN session."""

    root_path = Path(root)
    target_observation_path = Path(target_weight_observation)
    if not target_observation_path.is_absolute():
        target_observation_path = root_path / target_observation_path
    target = load_json_model(root_path, target_observation_path, LongitudinalObservationV1)
    target_observation_sha = sha256_bytes(target_observation_path.read_bytes())
    if target.observation_type != "target_weight":
        raise LongitudinalProducerError("target-weight observation is required")
    attribution = (
        attribution_date
        if isinstance(attribution_date, date)
        else date.fromisoformat(str(attribution_date))
    )
    if attribution <= target.trading_date:
        raise LongitudinalProducerError("NAV attribution date must follow target weights")
    request_path = next(
        (
            path
            for path in root_path.glob("CN/*/*/requests/*.request.v1.json")
            if load_json_model(root_path, path, FundamentalResearchRequestV1).request_id
            == target.request_id
        ),
        None,
    )
    if request_path is None:
        raise LongitudinalProducerError("request artifact is missing")
    request = _request(root_path, request_path)
    application = _application(
        root=root_path,
        request_id=target.request_id,
        dossier_id=target.dossier_id,
        run_key=target.run_key,
    )
    sources = []
    weights_by_variant: dict[str, dict[str, float]] = {}
    analysis_hashes: dict[str, str] = {}
    dossier_variants: dict[str, str] = {}
    for variant, outcome_relative in (
        ("actual", target.actual_artifact_path),
        ("counterfactual", target.counterfactual_artifact_path),
    ):
        outcome = load_json_model(
            root_path, root_path / outcome_relative, LongitudinalOutcomeArtifactV1
        )
        source = load_json_model(
            root_path,
            root_path / outcome.source_artifact_path,
            LongitudinalSourceArtifactV1,
        )
        payload, _, digest = _load_analysis_manifest(
            governance_module.REPO_ROOT / source.canonical_artifact_path
        )
        facts = _manifest_facts(payload, symbol=request.symbol)
        weights_by_variant[variant] = facts["weights"]
        analysis_hashes[variant] = digest
        dossier_variants[variant] = facts["variant"]
        sources.append(source)
    actual_value, actual_returns, generation = _portfolio_return(
        weights=weights_by_variant["actual"],
        prior_trading_date=target.trading_date,
        trading_date=attribution,
        data_root=data_root,
    )
    counter_value, counter_returns, counter_generation = _portfolio_return(
        weights=weights_by_variant["counterfactual"],
        prior_trading_date=target.trading_date,
        trading_date=attribution,
        data_root=data_root,
    )
    if generation != counter_generation:
        raise LongitudinalProducerError("NAV snapshot generation mismatch")
    produced_at = now or datetime.now(timezone.utc)
    common = {
        "run_key": target.run_key,
        "trading_date": attribution,
        "application_trading_date": target.trading_date,
        "generation": generation,
        "symbol": request.symbol,
    }
    nav_sources = []
    for variant, value, symbol_returns in (
        ("actual", actual_value, actual_returns),
        ("counterfactual", counter_value, counter_returns),
    ):
        metric = CanonicalLongitudinalMetricV1(
            **common,
            variant=variant,
            dossier_variant=dossier_variants[variant],
            value=value,
        )
        unsigned = {
            "schema_version": "portfolio-nav-snapshot.v1",
            "produced_at": produced_at.isoformat(),
            "longitudinal_metric": metric.model_dump(mode="json"),
            "analysis_manifest_sha256": analysis_hashes[variant],
            "target_weights": weights_by_variant[variant],
            "symbol_returns": symbol_returns,
            "target_weight_observation_sha256": target_observation_sha,
        }
        draft = CanonicalNavSnapshotV1(
            **unsigned,
            snapshot_sha256="0" * 64,
        )
        normalized_unsigned = draft.model_dump(mode="json", exclude={"snapshot_sha256"})
        snapshot_model = draft.model_copy(
            update={"snapshot_sha256": sha256_bytes(canonical_json_bytes(normalized_unsigned))}
        )
        path = (
            root_path
            / "canonical_nav"
            / attribution.isoformat()
            / variant
            / "canonical_nav_snapshot.v1.json"
        )
        digest = _immutable_write(root_path, path, snapshot_model)
        nav_sources.append(
            LongitudinalSourceArtifactV1(
                source_kind="canonical_nav_snapshot",
                variant=variant,
                run_key=target.run_key,
                trading_date=attribution,
                generation=generation,
                produced_at=produced_at,
                value=value,
                symbol=request.symbol,
                dossier_variant=dossier_variants[variant],
                canonical_artifact_path=path.relative_to(root_path).as_posix(),
                canonical_artifact_sha256=digest,
                parent_observation_path=target_observation_path.relative_to(root_path).as_posix(),
                parent_observation_sha256=target_observation_sha,
            )
        )
    return _write_comparison(
        root=root_path,
        request=request,
        dossier_id=target.dossier_id,
        application=application,
        observation_type="nav_attribution",
        trading_date=attribution,
        occurred_at=produced_at,
        actual_source=nav_sources[0],
        counterfactual_source=nav_sources[1],
    )


__all__ = [
    "LongitudinalProducerError",
    "produce_nav_attribution_observation",
    "produce_target_weight_observation",
]
