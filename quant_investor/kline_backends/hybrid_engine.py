"""Kline hybrid engine.

This module owns the explicit Kronos + Chronos orchestration layer and emits a
structured trace alongside the final BranchResult.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time as _time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

import pandas as pd

from quant_investor.branch_contracts import BranchResult

from .base import KLineBackend
from .evaluator import KLineEvaluationInput, get_kline_evaluator
from .heuristic import HeuristicBackend
from .chronos_adapter import ChronosBackend
from .kronos_adapter import KronosBackend

FULL_MARKET_THRESHOLD = 20
CHRONOS_TIMEOUT_SEC = 18.0


def _process_context() -> mp.context.BaseContext:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _terminate_process(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)


def _clone_branch_result(result: BranchResult) -> BranchResult:
    return BranchResult(
        branch_name=result.branch_name,
        score=result.score,
        confidence=result.confidence,
        signals=dict(result.signals),
        risks=list(result.risks),
        explanation=result.explanation,
        symbol_scores=dict(result.symbol_scores),
        success=result.success,
        metadata=dict(result.metadata),
        base_score=result.base_score,
        final_score=result.final_score,
        base_confidence=result.base_confidence,
        final_confidence=result.final_confidence,
        horizon_days=result.horizon_days,
        evidence=result.evidence,
        debate_verdict=result.debate_verdict,
        data_quality=dict(result.data_quality),
        conclusion=result.conclusion,
        thesis_points=list(result.thesis_points),
        investment_risks=list(result.investment_risks),
        coverage_notes=list(result.coverage_notes),
        diagnostic_notes=list(result.diagnostic_notes),
        support_drivers=list(result.support_drivers),
        drag_drivers=list(result.drag_drivers),
        weight_cap_reasons=list(result.weight_cap_reasons),
        module_coverage=dict(result.module_coverage),
    )


def _chronos_process_entry(
    backend_kwargs: dict[str, object],
    symbol_data: dict[str, pd.DataFrame],
    stock_pool: list[str],
    send_conn: Connection,
) -> None:
    try:
        backend = ChronosBackend(**backend_kwargs)
        result = backend.predict(symbol_data, stock_pool)
        send_conn.send({"ok": True, "result": result})
    except Exception as exc:  # pragma: no cover - child process fallback path
        send_conn.send({"ok": False, "error": str(exc)})
    finally:
        send_conn.close()


@dataclass
class KlineHybridSignal:
    branch_result: BranchResult
    execution_trace: "KlineExecutionTrace" = field(default_factory=lambda: KlineExecutionTrace())


@dataclass
class KlineExecutionTrace:
    """Structured trace for the Kline hybrid engine."""

    kronos_status: str = ""
    chronos_status: str = ""
    hybrid_mode: str = ""
    fallback_mode: str = ""
    degradation_reason: str = ""
    full_market_mode: bool = False
    shortlist: list[str] = field(default_factory=list)
    shortlist_count: int = 0
    screening_mode: str = ""
    model_components: dict[str, Any] = field(default_factory=dict)
    kronos_latency_ms: float = 0.0
    chronos_latency_ms: float = 0.0
    fusion_method: str = ""
    symbol_count: int = 0
    kronos_available: bool = False
    chronos_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kronos_status": self.kronos_status,
            "chronos_status": self.chronos_status,
            "hybrid_mode": self.hybrid_mode,
            "fallback_mode": self.fallback_mode,
            "degradation_reason": self.degradation_reason,
            "full_market_mode": self.full_market_mode,
            "shortlist": list(self.shortlist),
            "shortlist_count": int(self.shortlist_count),
            "screening_mode": self.screening_mode,
            "model_components": dict(self.model_components),
            "kronos_latency_ms": self.kronos_latency_ms,
            "chronos_latency_ms": self.chronos_latency_ms,
            "fusion_method": self.fusion_method,
            "symbol_count": self.symbol_count,
            "kronos_available": self.kronos_available,
            "chronos_available": self.chronos_available,
        }


class KlineHybridEngine:
    """Explicit Kronos + Chronos orchestration for the Kline branch."""

    name = "kline_hybrid_engine"

    def __init__(self, evaluator_name: str | None = None, **kwargs: object) -> None:
        self._kronos_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"kronos_path", "kronos_model_size", "allow_remote_download"}
        }
        self._chronos_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"model_name", "allow_remote_download"}
        }
        self._kronos = KronosBackend(**self._kronos_kwargs)
        self._heuristic = HeuristicBackend()
        self._evaluator = get_kline_evaluator(evaluator_name)

    def health_check(self) -> dict[str, Any]:
        """Probe model availability without running inference.

        Returns a dict with ``kronos_available``, ``chronos_available``,
        and the inferred ``mode`` (hybrid / kronos_only_degraded /
        chronos_only_degraded / statistical_only_fallback).
        """
        kronos_ok = getattr(self._kronos, "vendor_native_available", False)
        if not kronos_ok:
            try:
                dummy = pd.DataFrame(
                    {"close": [1.0] * 20, "date": pd.date_range("2025-01-01", periods=20)}
                )
                test_result = self._kronos.predict({"_probe": dummy}, ["_probe"])
                kronos_ok = str(test_result.metadata.get("model_runtime_mode", "")).startswith("vendor")
            except Exception:
                kronos_ok = False

        chronos_ok = False
        try:
            _backend = ChronosBackend(**self._chronos_kwargs)
            chronos_ok = getattr(_backend, "vendor_native_available", False)
        except Exception:
            chronos_ok = False

        if kronos_ok and chronos_ok:
            mode = "hybrid"
        elif kronos_ok:
            mode = "kronos_only_degraded"
        elif chronos_ok:
            mode = "chronos_only_degraded"
        else:
            mode = "statistical_only_fallback"

        return {
            "kronos_available": kronos_ok,
            "chronos_available": chronos_ok,
            "mode": mode,
        }

    @staticmethod
    def _shortlist_symbols(base_result: BranchResult, stock_pool: list[str]) -> list[str]:
        shortlist_count = min(max(8, math.ceil(len(stock_pool) * 0.35)), 12)
        ranked = [
            symbol
            for symbol, _ in sorted(
                base_result.symbol_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        shortlisted = ranked[:shortlist_count]
        if shortlisted:
            return shortlisted
        return stock_pool[:shortlist_count]

    @staticmethod
    def _status_from_result(result: BranchResult) -> str:
        status = str(result.metadata.get("model_runtime_mode", "") or "").strip()
        return status or "unknown"

    @staticmethod
    def _degradation_reason(kronos_status: str, chronos_status: str) -> str:
        missing: list[str] = []
        if kronos_status != "vendor_native":
            missing.append(f"Kronos={kronos_status or 'unknown'}")
        if chronos_status != "vendor_native":
            missing.append(f"Chronos={chronos_status or 'unknown'}")
        if not missing:
            return ""
        if kronos_status != "vendor_native" and chronos_status != "vendor_native":
            return "Kronos 与 Chronos 均未命中原生模型，已切换统计替代模式。"
        return f"检测到结构化降级：{', '.join(missing)}。"

    def _run_chronos_with_timeout(
        self,
        symbol_data: dict[str, pd.DataFrame],
        stock_pool: list[str],
        fallback_result: BranchResult,
    ) -> BranchResult:
        if not stock_pool:
            return _clone_branch_result(fallback_result)

        recv_conn, send_conn = _process_context().Pipe(duplex=False)
        process = _process_context().Process(
            target=_chronos_process_entry,
            args=(self._chronos_kwargs, symbol_data, stock_pool, send_conn),
            name="qi-chronos-shortlist",
        )
        process.start()
        send_conn.close()
        try:
            process.join(timeout=CHRONOS_TIMEOUT_SEC)
            if process.is_alive():
                recv_conn.close()
                _terminate_process(process)
                degraded = _clone_branch_result(fallback_result)
                degraded.confidence = max(degraded.confidence * 0.78, 0.28)
                degraded.metadata["model_runtime_mode"] = "timed_out_fallback"
                degraded.diagnostic_notes.append("Chronos 深度模型阶段超时，已自动保留快筛结论。")
                degraded.conclusion = "Chronos 深度模型阶段超时，已自动保留快筛结论，K 线判断未缺失。"
                return degraded

            payload = recv_conn.recv() if recv_conn.poll(0.1) else None
        finally:
            if not recv_conn.closed:
                recv_conn.close()
            process.join(timeout=0.1)

        if not payload or not payload.get("ok", False):
            degraded = _clone_branch_result(fallback_result)
            degraded.confidence = max(degraded.confidence * 0.82, 0.30)
            degraded.metadata["model_runtime_mode"] = "error_fallback"
            degraded.diagnostic_notes.append("Chronos 深度模型未完成本轮推理，已自动保留快筛结论。")
            degraded.conclusion = "Chronos 深度模型未完成本轮推理，已自动保留快筛结论，K 线判断未缺失。"
            return degraded

        return payload["result"]

    @staticmethod
    def _merge_shortlist_result(
        base_result: BranchResult,
        chronos_result: BranchResult,
        stock_pool: list[str],
        shortlist: list[str],
    ) -> BranchResult:
        merged = _clone_branch_result(chronos_result)
        merged.branch_name = "kline"
        base_returns = dict(base_result.signals.get("predicted_return", {}))
        base_regimes = dict(base_result.signals.get("trend_regime", {}))
        merged_returns = dict(chronos_result.signals.get("predicted_return", {}))
        merged_regimes = dict(chronos_result.signals.get("trend_regime", {}))

        for symbol in stock_pool:
            if symbol not in shortlist:
                merged.symbol_scores[symbol] = float(base_result.symbol_scores.get(symbol, 0.0))
                merged_returns[symbol] = float(base_returns.get(symbol, 0.0))
                merged_regimes[symbol] = str(base_regimes.get(symbol, "震荡"))

        merged.signals["predicted_return"] = merged_returns
        merged.signals["trend_regime"] = merged_regimes
        merged.coverage_notes = list(
            dict.fromkeys(
                [
                    *base_result.coverage_notes,
                    *chronos_result.coverage_notes,
                    f"全市场模式先完成 {len(stock_pool)}/{len(stock_pool)} 标的轻量 K 线快筛，仅对 {len(shortlist)}/{len(stock_pool)} 标的运行 Chronos 深度模型。",
                ]
            )
        )
        merged.diagnostic_notes = list(dict.fromkeys([*base_result.diagnostic_notes, *chronos_result.diagnostic_notes]))
        return merged

    def _build_trace(
        self,
        *,
        kronos_result: BranchResult,
        chronos_result: BranchResult,
        kronos_status: str,
        chronos_status: str,
        shortlist: list[str],
        full_market_mode: bool,
        kronos_latency_ms: float = 0.0,
        chronos_latency_ms: float = 0.0,
        symbol_count: int = 0,
    ) -> KlineExecutionTrace:
        fallback_mode = ""
        if kronos_status != "vendor_native" and chronos_status != "vendor_native":
            fallback_mode = "statistical_only"
        elif kronos_status != "vendor_native" or chronos_status != "vendor_native":
            fallback_mode = "structured_degraded"

        hybrid_mode = "dual_model" if not full_market_mode else "dual_model_shortlist"
        if fallback_mode == "statistical_only":
            hybrid_mode = "statistical_only"

        kronos_available = kronos_status == "vendor_native"
        chronos_available = chronos_status == "vendor_native"

        if kronos_available and chronos_available:
            fusion = "weighted_average"
        elif kronos_available or chronos_available:
            fusion = "single_model_dominant"
        else:
            fusion = "heuristic_only"

        return KlineExecutionTrace(
            kronos_status=kronos_status,
            chronos_status=chronos_status,
            hybrid_mode=hybrid_mode,
            fallback_mode=fallback_mode,
            degradation_reason=self._degradation_reason(kronos_status, chronos_status),
            full_market_mode=full_market_mode,
            shortlist=list(shortlist),
            shortlist_count=len(shortlist),
            screening_mode="heuristic_shortlist" if full_market_mode else "kronos_full_run",
            model_components={
                "kronos": {
                    "score": kronos_result.score,
                    "confidence": kronos_result.confidence,
                    "runtime_mode": kronos_status,
                    "reliability": float(kronos_result.metadata.get("reliability", 0.0)),
                },
                "chronos": {
                    "score": chronos_result.score,
                    "confidence": chronos_result.confidence,
                    "runtime_mode": chronos_status,
                    "reliability": float(chronos_result.metadata.get("reliability", 0.0)),
                },
            },
            kronos_latency_ms=kronos_latency_ms,
            chronos_latency_ms=chronos_latency_ms,
            fusion_method=fusion,
            symbol_count=symbol_count,
            kronos_available=kronos_available,
            chronos_available=chronos_available,
        )

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> KlineHybridSignal:
        full_market_mode = len(stock_pool) >= FULL_MARKET_THRESHOLD
        screening_result: BranchResult | None = None

        kronos_t0 = _time.monotonic()
        if full_market_mode:
            screening_result = self._heuristic.predict(symbol_data, stock_pool)
            screening_result.metadata["screening_mode"] = "heuristic_shortlist"
            shortlist = self._shortlist_symbols(screening_result, stock_pool)
            kronos_input = {symbol: symbol_data[symbol] for symbol in shortlist if symbol in symbol_data}
            kronos_deep_result = self._kronos.predict(kronos_input, shortlist)
        else:
            shortlist = list(stock_pool)
            kronos_input = symbol_data
            kronos_deep_result = self._kronos.predict(kronos_input, stock_pool)
        kronos_latency_ms = (_time.monotonic() - kronos_t0) * 1000.0
        kronos_deep_result.metadata["screening_mode"] = "heuristic_shortlist" if full_market_mode else "kronos_full_run"

        shortlist_symbol_data = {symbol: symbol_data[symbol] for symbol in shortlist if symbol in symbol_data}
        chronos_t0 = _time.monotonic()
        chronos_deep_result = self._run_chronos_with_timeout(shortlist_symbol_data, shortlist, kronos_deep_result)
        chronos_latency_ms = (_time.monotonic() - chronos_t0) * 1000.0
        if full_market_mode:
            assert screening_result is not None
            kronos_result = self._merge_shortlist_result(screening_result, kronos_deep_result, stock_pool, shortlist)
            chronos_result = self._merge_shortlist_result(screening_result, chronos_deep_result, stock_pool, shortlist)
        else:
            kronos_result = kronos_deep_result
            chronos_result = chronos_deep_result

        evaluation = self._evaluator.evaluate(
            KLineEvaluationInput(
                stock_pool=stock_pool,
                symbol_data=symbol_data,
                kronos_result=kronos_result,
                chronos_result=chronos_result,
            )
        )

        trace = self._build_trace(
            kronos_result=kronos_result,
            chronos_result=chronos_result,
            kronos_status=self._status_from_result(kronos_deep_result),
            chronos_status=self._status_from_result(chronos_deep_result),
            shortlist=shortlist,
            full_market_mode=full_market_mode,
            kronos_latency_ms=kronos_latency_ms,
            chronos_latency_ms=chronos_latency_ms,
            symbol_count=len(stock_pool),
        )

        metadata = dict(evaluation.metadata)
        metadata.update(
            {
                "model_mode": "hybrid",
                "screening_mode": trace.screening_mode,
                "kronos_status": trace.kronos_status,
                "chronos_status": trace.chronos_status,
                "hybrid_mode": trace.hybrid_mode,
                "fallback_mode": trace.fallback_mode,
                "degradation_reason": trace.degradation_reason,
                "chronos_shortlist": list(shortlist),
                "kline_execution_trace": trace.to_dict(),
                "model_runtime_mode": trace.fallback_mode or "hybrid",
            }
        )

        combined_diagnostic_notes = list(
            dict.fromkeys([*kronos_result.diagnostic_notes, *chronos_result.diagnostic_notes])
        )
        if trace.degradation_reason:
            combined_diagnostic_notes.append(trace.degradation_reason)
        if trace.fallback_mode == "statistical_only":
            combined_diagnostic_notes.append("Kronos 与 Chronos 均已回退到统计替代模式。")
        elif trace.fallback_mode == "structured_degraded":
            combined_diagnostic_notes.append("Kline 双模型当前处于结构化降级模式。")

        conclusion = evaluation.explanation
        if trace.fallback_mode == "statistical_only":
            conclusion = "Kronos 与 Chronos 均未命中原生模型，已切换统计替代融合，但 K 线判断仍可用。"
        elif trace.fallback_mode == "structured_degraded":
            conclusion = "Kline 双模型当前处于结构化降级融合，结果仍可用于 shortlist。"
        elif full_market_mode:
            conclusion = "Kline 双模型在全市场快筛模式下完成了 Kronos + Chronos 融合。"

        branch_result = BranchResult(
            branch_name="kline",
            score=evaluation.score,
            confidence=evaluation.confidence,
            signals={
                "predicted_return": evaluation.predicted_returns,
                "trend_regime": evaluation.regimes,
                "model_mode": "hybrid",
                "evaluator_name": self._evaluator.name,
                "kronos_status": trace.kronos_status,
                "chronos_status": trace.chronos_status,
                "hybrid_mode": trace.hybrid_mode,
                "fallback_mode": trace.fallback_mode,
            },
            risks=list(dict.fromkeys(evaluation.risks)),
            explanation=evaluation.explanation,
            symbol_scores=evaluation.symbol_scores,
            metadata=metadata,
            conclusion=conclusion,
            diagnostic_notes=combined_diagnostic_notes,
        )
        return KlineHybridSignal(branch_result=branch_result, execution_trace=trace)
