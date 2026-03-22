"""Kronos + Chronos 双模型综合后端。

当前生产 K-line 主链固定使用该后端。它会先执行 Kronos 与 Chronos，
再把两个子模型结果交给预留的 evaluator 接口做综合评估。
"""

from __future__ import annotations

import pandas as pd

from quant_investor.contracts import BranchResult

from .base import KLineBackend
from .evaluator import KLineEvaluationInput, get_kline_evaluator


class HybridBackend(KLineBackend):
    """K-line 生产后端：固定双模型 + evaluator 综合输出。"""

    name = "hybrid"
    reliability = 0.84
    horizon_days = 5

    def __init__(self, evaluator_name: str | None = None, **kwargs: object) -> None:
        from .chronos_adapter import ChronosBackend
        from .kronos_adapter import KronosBackend

        self._kronos = KronosBackend(
            **{key: value for key, value in kwargs.items() if key in {"kronos_path", "kronos_model_size", "allow_remote_download"}}
        )
        self._chronos = ChronosBackend(
            **{key: value for key, value in kwargs.items() if key in {"model_name", "allow_remote_download"}}
        )
        self._evaluator = get_kline_evaluator(evaluator_name)

    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        kronos_result = self._kronos.predict(symbol_data, stock_pool)
        chronos_result = self._chronos.predict(symbol_data, stock_pool)

        evaluation = self._evaluator.evaluate(
            KLineEvaluationInput(
                stock_pool=stock_pool,
                symbol_data=symbol_data,
                kronos_result=kronos_result,
                chronos_result=chronos_result,
            )
        )

        metadata = dict(evaluation.metadata)
        metadata.setdefault("reliability", self.reliability)
        metadata.setdefault("horizon_days", self.horizon_days)

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
        )
