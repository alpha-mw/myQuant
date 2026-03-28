"""
KlineAgent：对现有 K 线分支做 agent 化包装。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.config import config
from quant_investor.agents.base import BaseAgent


class KlineAgent(BaseAgent):
    """K 线 research agent。

    full market 模式只做轻量快筛；
    shortlist 模式才允许调用深模型链路。
    """

    agent_name = "KlineAgent"
    FULL_MARKET_THRESHOLD = 20

    def run(self, payload: Mapping[str, Any]) -> Any:
        from quant_investor.kline_backends import get_backend

        envelope = self.ensure_payload(payload)
        symbol_data, stock_pool = self._extract_symbol_inputs(envelope)
        mode = str(
            envelope.get("mode")
            or ("full_market" if len(stock_pool) >= self.FULL_MARKET_THRESHOLD else "shortlist")
        ).strip().lower()

        if mode == "full_market":
            backend = envelope.get("heuristic_backend") or get_backend("heuristic")
            result = backend.predict(symbol_data, stock_pool)
            result.metadata["agent_runtime_mode"] = "full_market_fast_screen"
            thesis = self._build_thesis(result, mode=mode)
            return self.branch_result_to_verdict(
                result,
                thesis=thesis,
                metadata={"mode": mode, "backend_used": "heuristic"},
            )

        requested_backend = str(envelope.get("backend_name", "hybrid")).strip().lower() or "hybrid"
        backend_kwargs = {
            "kronos_path": envelope.get("kronos_path", config.KRONOS_MODEL_PATH),
            "kronos_model_size": envelope.get("kronos_model_size", config.KRONOS_MODEL_SIZE),
            "model_name": envelope.get("model_name", config.CHRONOS_MODEL_NAME),
            "evaluator_name": envelope.get("evaluator_name", config.KLINE_EVALUATOR),
            "allow_remote_download": envelope.get(
                "allow_remote_download",
                config.KLINE_ALLOW_REMOTE_MODEL_DOWNLOAD,
            ),
        }

        backend = envelope.get("backend")
        try:
            if backend is None:
                backend = get_backend(requested_backend, **backend_kwargs)
            result = backend.predict(symbol_data, stock_pool)
        except TimeoutError:
            result = self._fallback_base_result(symbol_data, stock_pool, reason="timeout")
        except Exception as exc:
            result = self._fallback_base_result(symbol_data, stock_pool, reason=str(exc))

        thesis = self._build_thesis(result, mode=mode)
        return self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={"mode": mode, "backend_used": requested_backend},
        )

    @staticmethod
    def _extract_symbol_inputs(
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        data_bundle = payload.get("data_bundle")
        if isinstance(data_bundle, UnifiedDataBundle):
            symbol_data = dict(data_bundle.symbol_data)
            stock_pool = list(payload.get("stock_pool") or data_bundle.symbols)
            return symbol_data, stock_pool

        symbol_data = payload.get("symbol_data")
        if not isinstance(symbol_data, Mapping):
            raise TypeError("KlineAgent 需要 `data_bundle` 或 `symbol_data`")
        stock_pool = list(payload.get("stock_pool") or symbol_data.keys())
        return dict(symbol_data), stock_pool

    def _fallback_base_result(
        self,
        symbol_data: Mapping[str, Any],
        stock_pool: list[str],
        reason: str,
    ) -> BranchResult:
        from quant_investor.kline_backends import get_backend

        base_backend = get_backend("heuristic")
        result = base_backend.predict(dict(symbol_data), list(stock_pool))
        lowered_conf = max(float(result.confidence) * 0.72, 0.25)
        result.confidence = lowered_conf
        lowered_base_conf = result.base_confidence if result.base_confidence is not None else lowered_conf
        result.base_confidence = min(float(lowered_base_conf), lowered_conf)
        note = (
            "Chronos 深度模型阶段超时，已自动保留 base result。"
            if "timeout" in reason.lower()
            else "Chronos 深度模型失败，已自动保留 base result。"
        )
        if note not in result.diagnostic_notes:
            result.diagnostic_notes.append(note)
        result.metadata["agent_runtime_mode"] = "deep_model_fallback"
        result.metadata["fallback_reason"] = reason
        result.conclusion = (
            "深度模型未完成本轮推理，已保留 K 线基础结论，趋势判断仍然有效。"
        )
        return result

    def _build_thesis(self, result: BranchResult, mode: str) -> str:
        if str(result.conclusion or "").strip():
            base = str(result.conclusion).strip()
        else:
            top_symbols = self.top_symbols(result.symbol_scores, limit=2)
            if result.score >= 0.15:
                base = f"K线分支当前整体偏多，重点关注 {'、'.join(top_symbols) or '核心候选标的'}。"
            elif result.score <= -0.15:
                base = f"K线分支当前整体偏谨慎，需回避 {'、'.join(top_symbols) or '高波动标的'}。"
            else:
                base = "K线分支当前整体中性，趋势尚未形成单边共识。"

        if mode == "full_market":
            return f"{base} 当前处于 full market 快筛模式，仅使用轻量 K 线筛选结果。"
        return base
