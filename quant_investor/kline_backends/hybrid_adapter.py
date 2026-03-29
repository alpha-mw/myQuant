"""Kline 混合后端。

常规模式下执行 Kronos + Chronos 双模型。
全市场批量模式下先执行轻量快筛，再仅对 shortlist 运行 Chronos，
并通过子进程超时保护保留 base result，避免整批丢失结论。
"""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import Connection

import pandas as pd

from quant_investor.branch_contracts import BranchResult

from .base import KLineBackend
from .evaluator import KLineEvaluationInput, get_kline_evaluator

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
        from .chronos_adapter import ChronosBackend

        backend = ChronosBackend(**backend_kwargs)
        result = backend.predict(symbol_data, stock_pool)
        send_conn.send({"ok": True, "result": result})
    except Exception as exc:  # pragma: no cover - child process fallback path
        send_conn.send({"ok": False, "error": str(exc)})
    finally:
        send_conn.close()


class HybridBackend(KLineBackend):
    """K-line 生产后端。"""

    name = "hybrid"
    reliability = 0.84
    horizon_days = 5

    def __init__(self, evaluator_name: str | None = None, **kwargs: object) -> None:
        from .heuristic import HeuristicBackend
        from .kronos_adapter import KronosBackend

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
        merged.coverage_notes = list(dict.fromkeys([
            *base_result.coverage_notes,
            *chronos_result.coverage_notes,
            f"全市场模式先完成 {len(stock_pool)}/{len(stock_pool)} 标的轻量 K 线快筛，仅对 {len(shortlist)}/{len(stock_pool)} 标的运行 Chronos 深度模型。",
        ]))
        merged.diagnostic_notes = list(dict.fromkeys([*base_result.diagnostic_notes, *chronos_result.diagnostic_notes]))
        return merged

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        full_market_mode = len(stock_pool) >= FULL_MARKET_THRESHOLD
        if full_market_mode:
            base_result = self._heuristic.predict(symbol_data, stock_pool)
            base_result.metadata["screening_mode"] = "heuristic_shortlist"
            base_result.coverage_notes.append(
                f"全市场模式已先对 {len(stock_pool)}/{len(stock_pool)} 标的执行轻量 K 线快筛。"
            )
        else:
            base_result = self._kronos.predict(symbol_data, stock_pool)
            base_result.metadata["screening_mode"] = "kronos_full_run"

        shortlist = self._shortlist_symbols(base_result, stock_pool) if full_market_mode else list(stock_pool)
        shortlist_symbol_data = {
            symbol: symbol_data[symbol]
            for symbol in shortlist
            if symbol in symbol_data
        }
        chronos_result = self._run_chronos_with_timeout(shortlist_symbol_data, shortlist, base_result)
        if full_market_mode:
            chronos_result = self._merge_shortlist_result(base_result, chronos_result, stock_pool, shortlist)

        evaluation = self._evaluator.evaluate(
            KLineEvaluationInput(
                stock_pool=stock_pool,
                symbol_data=symbol_data,
                kronos_result=base_result,
                chronos_result=chronos_result,
            )
        )

        metadata = dict(evaluation.metadata)
        metadata.setdefault("reliability", self.reliability)
        metadata.setdefault("horizon_days", self.horizon_days)
        metadata["screening_mode"] = base_result.metadata.get("screening_mode", "kronos_full_run")
        metadata["chronos_shortlist"] = shortlist

        support_drivers = []
        drag_drivers = []
        if evaluation.score > 0.05:
            support_drivers.append("K 线快筛与深度模型结论整体偏正。")
        elif evaluation.score < -0.05:
            drag_drivers.append("K 线快筛与深度模型结论整体偏弱。")
        if full_market_mode:
            support_drivers.append(
                f"全市场模式仅对 {len(shortlist)}/{len(stock_pool)} 标的运行 Chronos，降低了整批超时风险。"
            )

        return BranchResult(
            branch_name="kline",
            score=evaluation.score,
            confidence=evaluation.confidence,
            signals={
                "predicted_return": evaluation.predicted_returns,
                "trend_regime": evaluation.regimes,
                "model_mode": "dual_model_evaluated",
                "evaluator_name": self._evaluator.name,
            },
            risks=evaluation.risks,
            explanation=evaluation.explanation,
            symbol_scores=evaluation.symbol_scores,
            metadata=metadata,
            conclusion=(
                "K 线分支已形成完整结论，趋势判断由快筛结果与 Chronos 深度模型共同给出。"
            ),
            coverage_notes=list(dict.fromkeys([*base_result.coverage_notes, *chronos_result.coverage_notes])),
            diagnostic_notes=list(dict.fromkeys([*base_result.diagnostic_notes, *chronos_result.diagnostic_notes])),
            support_drivers=support_drivers,
            drag_drivers=drag_drivers,
            investment_risks=list(dict.fromkeys(evaluation.risks))[:4],
        )
