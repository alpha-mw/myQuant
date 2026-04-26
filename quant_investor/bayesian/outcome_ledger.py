"""Append-only Bayesian prediction and outcome ledger.

This module is intentionally offline and JSONL-only.  It captures current
posterior artifacts for future empirical calibration without changing the
posterior, risk, portfolio, data, LLM, or web runtime paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from quant_investor.agent_protocol import GlobalContext
from quant_investor.bayesian.types import PosteriorResult
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    OUTCOME_LEDGER_SCHEMA_VERSION,
)

try:
    from quant_investor.branch_config import BRANCH_WEIGHT_VERSION, CANONICAL_BRANCH_ORDER
except ImportError:  # pragma: no cover - compatibility fallback for older trees
    BRANCH_WEIGHT_VERSION = ""
    CANONICAL_BRANCH_ORDER = (
        "quant",
        "kline",
        "intelligence",
        "fundamental",
        "macro",
    )


DEFAULT_OUTCOME_LEDGER_DIR = Path("data/bayesian_outcome_ledger")
DEFAULT_PREDICTIONS_FILENAME = "predictions.jsonl"
DEFAULT_OUTCOMES_FILENAME = "outcomes.jsonl"

OUTCOME_STATUS_PENDING = "pending"
OUTCOME_STATUS_RESOLVED = "resolved"
OUTCOME_STATUS_MISSING_DATA = "missing_data"
OUTCOME_STATUS_UNTRADABLE = "untradable"

_ID_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_T = TypeVar("_T")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_safe(getattr(value, "value"))
    if isinstance(value, Path):
        return str(value)
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_id_part(value: Any) -> str:
    text = str(value or "").strip() or "na"
    text = _ID_SAFE_PATTERN.sub("-", text)
    return text.strip("-") or "na"


def _digest(*parts: Any, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return dict(_json_safe(payload))
    if isinstance(value, Mapping):
        return dict(_json_safe(value))
    return {}


def _mapping_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_json_safe(value))


def horizon_label_for_days(horizon_days: int) -> str:
    return f"{int(horizon_days)}D"


def make_deterministic_run_id(
    *,
    market: str,
    universe_key: str,
    rebalance_date: str,
    universe_hash: str,
) -> str:
    suffix = _digest(market, universe_key, rebalance_date, universe_hash, length=10)
    return "-".join(
        [
            "run",
            _safe_id_part(market).lower(),
            _safe_id_part(universe_key),
            _safe_id_part(rebalance_date),
            suffix,
        ]
    )


def make_prediction_id(*, run_id: str, symbol: str, horizon_days: int) -> str:
    suffix = _digest(run_id, symbol, horizon_days, length=10)
    return "-".join(
        [
            "pred",
            _safe_id_part(run_id),
            _safe_id_part(symbol),
            f"{int(horizon_days)}d",
            suffix,
        ]
    )


def make_outcome_id(*, prediction_id: str, resolution_date: str, status: str) -> str:
    suffix = _digest(prediction_id, resolution_date, status, length=10)
    return "-".join(
        [
            "out",
            _safe_id_part(prediction_id),
            _safe_id_part(resolution_date),
            _safe_id_part(status),
            suffix,
        ]
    )


def extract_branch_scores(branch_results: Mapping[str, Any], symbol: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for branch_name in CANONICAL_BRANCH_ORDER:
        branch = branch_results.get(branch_name)
        value = 0.0
        if branch is not None:
            symbol_scores = _mapping_attr_or_key(branch, "symbol_scores", {})
            if isinstance(symbol_scores, Mapping) and symbol in symbol_scores:
                value = float(symbol_scores.get(symbol, 0.0) or 0.0)
            else:
                value = float(_mapping_attr_or_key(branch, "final_score", _mapping_attr_or_key(branch, "score", 0.0)) or 0.0)
        scores[str(branch_name)] = value
    return scores


def extract_branch_confidences(branch_results: Mapping[str, Any], symbol: str) -> dict[str, float]:
    del symbol
    confidences: dict[str, float] = {}
    for branch_name in CANONICAL_BRANCH_ORDER:
        branch = branch_results.get(branch_name)
        value = 0.0
        if branch is not None:
            final_confidence = _mapping_attr_or_key(branch, "final_confidence", None)
            if final_confidence is None:
                final_confidence = _mapping_attr_or_key(branch, "confidence", 0.0)
            value = float(final_confidence or 0.0)
        confidences[str(branch_name)] = value
    return confidences


@dataclass
class PredictionRecord:
    schema_version: str = OUTCOME_LEDGER_SCHEMA_VERSION
    record_type: str = "prediction"
    prediction_id: str = ""
    run_id: str = ""
    run_date: str = ""
    rebalance_date: str = ""
    latest_trade_date: str = ""
    horizon_days: int = 20
    horizon_label: str = "20D"
    symbol: str = ""
    company_name: str = ""
    market: str = ""
    universe_key: str = ""
    universe_hash: str = ""
    macro_regime: str = ""
    rank: int = 0
    prior: dict[str, Any] = field(default_factory=dict)
    likelihoods: dict[str, Any] = field(default_factory=dict)
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_confidences: dict[str, float] = field(default_factory=dict)
    posterior_win_rate: float = 0.0
    posterior_expected_alpha: float = 0.0
    posterior_confidence: float = 0.0
    posterior_action_score: float = 0.0
    posterior_edge_after_costs: float = 0.0
    posterior_capacity_penalty: float = 0.0
    correlation_discount: float = 0.0
    coverage_discount: float = 0.0
    data_quality_penalty: float = 0.0
    fallback_penalty: float = 0.0
    regime_adjustment: float = 0.0
    action_threshold_used: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionRecord":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OUTCOME_LEDGER_SCHEMA_VERSION)),
            record_type=str(data.get("record_type", "prediction")),
            prediction_id=str(data.get("prediction_id", "")),
            run_id=str(data.get("run_id", "")),
            run_date=str(data.get("run_date", "")),
            rebalance_date=str(data.get("rebalance_date", "")),
            latest_trade_date=str(data.get("latest_trade_date", "")),
            horizon_days=int(data.get("horizon_days", 20) or 20),
            horizon_label=str(data.get("horizon_label", horizon_label_for_days(int(data.get("horizon_days", 20) or 20)))),
            symbol=str(data.get("symbol", "")),
            company_name=str(data.get("company_name", "")),
            market=str(data.get("market", "")),
            universe_key=str(data.get("universe_key", "")),
            universe_hash=str(data.get("universe_hash", "")),
            macro_regime=str(data.get("macro_regime", "")),
            rank=int(data.get("rank", 0) or 0),
            prior=dict(data.get("prior", {}) or {}),
            likelihoods=dict(data.get("likelihoods", {}) or {}),
            branch_scores={str(key): float(value or 0.0) for key, value in dict(data.get("branch_scores", {}) or {}).items()},
            branch_confidences={str(key): float(value or 0.0) for key, value in dict(data.get("branch_confidences", {}) or {}).items()},
            posterior_win_rate=float(data.get("posterior_win_rate", 0.0) or 0.0),
            posterior_expected_alpha=float(data.get("posterior_expected_alpha", 0.0) or 0.0),
            posterior_confidence=float(data.get("posterior_confidence", 0.0) or 0.0),
            posterior_action_score=float(data.get("posterior_action_score", 0.0) or 0.0),
            posterior_edge_after_costs=float(data.get("posterior_edge_after_costs", 0.0) or 0.0),
            posterior_capacity_penalty=float(data.get("posterior_capacity_penalty", 0.0) or 0.0),
            correlation_discount=float(data.get("correlation_discount", 0.0) or 0.0),
            coverage_discount=float(data.get("coverage_discount", 0.0) or 0.0),
            data_quality_penalty=float(data.get("data_quality_penalty", 0.0) or 0.0),
            fallback_penalty=float(data.get("fallback_penalty", 0.0) or 0.0),
            regime_adjustment=float(data.get("regime_adjustment", 0.0) or 0.0),
            action_threshold_used=float(data.get("action_threshold_used", 0.0) or 0.0),
            evidence_sources=[str(item) for item in list(data.get("evidence_sources", []) or [])],
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class OutcomeRecord:
    schema_version: str = OUTCOME_LEDGER_SCHEMA_VERSION
    record_type: str = "outcome"
    outcome_id: str = ""
    prediction_id: str = ""
    run_id: str = ""
    symbol: str = ""
    market: str = ""
    horizon_days: int = 20
    horizon_label: str = "20D"
    run_date: str = ""
    resolution_date: str = ""
    status: str = OUTCOME_STATUS_PENDING
    entry_price: float | None = None
    exit_price: float | None = None
    realized_return: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    cost_estimate: float | None = None
    slippage_estimate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.excess_return is None and self.realized_return is not None and self.benchmark_return is not None:
            self.excess_return = float(self.realized_return) - float(self.benchmark_return)
        if not self.horizon_label:
            self.horizon_label = horizon_label_for_days(self.horizon_days)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeRecord":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", OUTCOME_LEDGER_SCHEMA_VERSION)),
            record_type=str(data.get("record_type", "outcome")),
            outcome_id=str(data.get("outcome_id", "")),
            prediction_id=str(data.get("prediction_id", "")),
            run_id=str(data.get("run_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            horizon_days=int(data.get("horizon_days", 20) or 20),
            horizon_label=str(data.get("horizon_label", horizon_label_for_days(int(data.get("horizon_days", 20) or 20)))),
            run_date=str(data.get("run_date", "")),
            resolution_date=str(data.get("resolution_date", "")),
            status=str(data.get("status", OUTCOME_STATUS_PENDING)),
            entry_price=_float_or_none(data.get("entry_price")),
            exit_price=_float_or_none(data.get("exit_price")),
            realized_return=_float_or_none(data.get("realized_return")),
            benchmark_return=_float_or_none(data.get("benchmark_return")),
            excess_return=_float_or_none(data.get("excess_return")),
            max_drawdown=_float_or_none(data.get("max_drawdown")),
            turnover=_float_or_none(data.get("turnover")),
            cost_estimate=_float_or_none(data.get("cost_estimate")),
            slippage_estimate=_float_or_none(data.get("slippage_estimate")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def build_prediction_record(
    result: PosteriorResult,
    global_context: GlobalContext,
    branch_results: Mapping[str, Any],
    *,
    horizon_days: int = 20,
    run_id: str | None = None,
    run_date: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PredictionRecord:
    rebalance_date = str(getattr(global_context, "rebalance_date", "") or "")
    latest_trade_date = str(getattr(global_context, "latest_trade_date", "") or "")
    resolved_run_date = str(run_date or rebalance_date or latest_trade_date)
    resolved_run_id = run_id or make_deterministic_run_id(
        market=str(getattr(global_context, "market", "") or ""),
        universe_key=str(getattr(global_context, "universe_key", "") or ""),
        rebalance_date=rebalance_date or resolved_run_date,
        universe_hash=str(getattr(global_context, "universe_hash", "") or ""),
    )
    branch_scores = extract_branch_scores(branch_results, result.symbol)
    branch_confidences = extract_branch_confidences(branch_results, result.symbol)
    record_metadata = _coerce_mapping(metadata)
    result_metadata = getattr(result, "metadata", {}) or {}
    if isinstance(result_metadata, Mapping) and result_metadata:
        record_metadata.setdefault("posterior_metadata", dict(_json_safe(result_metadata)))
    record_metadata.setdefault("architecture_version", ARCHITECTURE_VERSION)
    record_metadata.setdefault("calibration_schema_version", CALIBRATION_SCHEMA_VERSION)
    record_metadata.setdefault("outcome_ledger_schema_version", OUTCOME_LEDGER_SCHEMA_VERSION)
    if BRANCH_WEIGHT_VERSION:
        record_metadata.setdefault("branch_weight_version", BRANCH_WEIGHT_VERSION)

    return PredictionRecord(
        schema_version=OUTCOME_LEDGER_SCHEMA_VERSION,
        prediction_id=make_prediction_id(
            run_id=resolved_run_id,
            symbol=result.symbol,
            horizon_days=horizon_days,
        ),
        run_id=resolved_run_id,
        run_date=resolved_run_date,
        rebalance_date=rebalance_date,
        latest_trade_date=latest_trade_date,
        horizon_days=int(horizon_days),
        horizon_label=horizon_label_for_days(horizon_days),
        symbol=str(result.symbol),
        company_name=str(result.company_name),
        market=str(getattr(global_context, "market", "") or ""),
        universe_key=str(getattr(global_context, "universe_key", "") or ""),
        universe_hash=str(getattr(global_context, "universe_hash", "") or ""),
        macro_regime=str(getattr(global_context, "macro_regime", "") or ""),
        rank=int(getattr(result, "rank", 0) or 0),
        prior=_object_to_dict(getattr(result, "prior", None)),
        likelihoods=_object_to_dict(getattr(result, "likelihoods", None)),
        branch_scores=branch_scores,
        branch_confidences=branch_confidences,
        posterior_win_rate=float(getattr(result, "posterior_win_rate", 0.0) or 0.0),
        posterior_expected_alpha=float(getattr(result, "posterior_expected_alpha", 0.0) or 0.0),
        posterior_confidence=float(getattr(result, "posterior_confidence", 0.0) or 0.0),
        posterior_action_score=float(getattr(result, "posterior_action_score", 0.0) or 0.0),
        posterior_edge_after_costs=float(getattr(result, "posterior_edge_after_costs", 0.0) or 0.0),
        posterior_capacity_penalty=float(getattr(result, "posterior_capacity_penalty", 0.0) or 0.0),
        correlation_discount=float(getattr(result, "correlation_discount", 0.0) or 0.0),
        coverage_discount=float(getattr(result, "coverage_discount", 0.0) or 0.0),
        data_quality_penalty=float(getattr(result, "data_quality_penalty", 0.0) or 0.0),
        fallback_penalty=float(getattr(result, "fallback_penalty", 0.0) or 0.0),
        regime_adjustment=float(getattr(result, "regime_adjustment", 0.0) or 0.0),
        action_threshold_used=float(getattr(result, "action_threshold_used", 0.0) or 0.0),
        evidence_sources=[str(item) for item in list(getattr(result, "evidence_sources", []) or [])],
        metadata=record_metadata,
    )


def build_prediction_records(
    results: Sequence[PosteriorResult],
    global_context: GlobalContext,
    branch_results: Mapping[str, Any],
    *,
    horizon_days: int = 20,
    run_id: str | None = None,
    run_date: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[PredictionRecord]:
    records = [
        build_prediction_record(
            result,
            global_context,
            branch_results,
            horizon_days=horizon_days,
            run_id=run_id,
            run_date=run_date,
            metadata=metadata,
        )
        for result in results
    ]
    prediction_ids = [record.prediction_id for record in records]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Duplicate prediction_id values generated for prediction batch.")
    return records


class OutcomeLedgerStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_OUTCOME_LEDGER_DIR
        self.predictions_path = self.root_dir / DEFAULT_PREDICTIONS_FILENAME
        self.outcomes_path = self.root_dir / DEFAULT_OUTCOMES_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(_json_safe(payload)), ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path} at line {line_number}.")
                rows.append(payload)
        return rows

    def append_prediction(self, record: PredictionRecord) -> None:
        if record.prediction_id in self.get_prediction_ids():
            raise ValueError(f"Duplicate prediction_id in prediction ledger: {record.prediction_id}")
        self._append_jsonl(self.predictions_path, record.to_dict())

    def append_predictions(self, records: Sequence[PredictionRecord]) -> int:
        existing_ids = self.get_prediction_ids()
        batch_ids: set[str] = set()
        for record in records:
            if record.prediction_id in existing_ids or record.prediction_id in batch_ids:
                raise ValueError(f"Duplicate prediction_id in prediction ledger: {record.prediction_id}")
            batch_ids.add(record.prediction_id)
        for record in records:
            self._append_jsonl(self.predictions_path, record.to_dict())
        return len(records)

    def append_outcome(self, record: OutcomeRecord) -> None:
        if record.prediction_id in self.get_resolved_prediction_ids():
            raise ValueError(f"Duplicate prediction_id in outcome ledger: {record.prediction_id}")
        self._append_jsonl(self.outcomes_path, record.to_dict())

    def read_predictions(self) -> list[PredictionRecord]:
        return [PredictionRecord.from_dict(payload) for payload in self._read_jsonl(self.predictions_path)]

    def read_outcomes(self) -> list[OutcomeRecord]:
        return [OutcomeRecord.from_dict(payload) for payload in self._read_jsonl(self.outcomes_path)]

    def get_prediction_ids(self) -> set[str]:
        return {record.prediction_id for record in self.read_predictions()}

    def get_resolved_prediction_ids(self) -> set[str]:
        return {record.prediction_id for record in self.read_outcomes()}

    def iter_unresolved_predictions(self, *, horizon_days: int | None = None) -> list[PredictionRecord]:
        resolved_ids = self.get_resolved_prediction_ids()
        records = [record for record in self.read_predictions() if record.prediction_id not in resolved_ids]
        if horizon_days is not None:
            records = [record for record in records if record.horizon_days == horizon_days]
        return records

    def resolve_prediction(
        self,
        prediction_id: str,
        *,
        resolution_date: str,
        status: str = OUTCOME_STATUS_RESOLVED,
        realized_return: float | None = None,
        benchmark_return: float | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        max_drawdown: float | None = None,
        turnover: float | None = None,
        cost_estimate: float | None = None,
        slippage_estimate: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OutcomeRecord:
        predictions = {record.prediction_id: record for record in self.read_predictions()}
        if prediction_id not in predictions:
            raise ValueError(f"Unknown prediction_id: {prediction_id}")
        if prediction_id in self.get_resolved_prediction_ids():
            raise ValueError(f"Duplicate prediction_id in outcome ledger: {prediction_id}")
        prediction = predictions[prediction_id]
        outcome = OutcomeRecord(
            schema_version=OUTCOME_LEDGER_SCHEMA_VERSION,
            outcome_id=make_outcome_id(
                prediction_id=prediction_id,
                resolution_date=resolution_date,
                status=status,
            ),
            prediction_id=prediction_id,
            run_id=prediction.run_id,
            symbol=prediction.symbol,
            market=prediction.market,
            horizon_days=prediction.horizon_days,
            horizon_label=prediction.horizon_label,
            run_date=prediction.run_date,
            resolution_date=resolution_date,
            status=status,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_return=realized_return,
            benchmark_return=benchmark_return,
            max_drawdown=max_drawdown,
            turnover=turnover,
            cost_estimate=cost_estimate,
            slippage_estimate=slippage_estimate,
            metadata=_coerce_mapping(metadata),
        )
        self.append_outcome(outcome)
        return outcome


__all__ = [
    "DEFAULT_OUTCOME_LEDGER_DIR",
    "DEFAULT_PREDICTIONS_FILENAME",
    "DEFAULT_OUTCOMES_FILENAME",
    "OUTCOME_STATUS_PENDING",
    "OUTCOME_STATUS_RESOLVED",
    "OUTCOME_STATUS_MISSING_DATA",
    "OUTCOME_STATUS_UNTRADABLE",
    "PredictionRecord",
    "OutcomeRecord",
    "OutcomeLedgerStore",
    "build_prediction_record",
    "build_prediction_records",
    "extract_branch_confidences",
    "extract_branch_scores",
    "horizon_label_for_days",
    "make_deterministic_run_id",
    "make_outcome_id",
    "make_prediction_id",
]
