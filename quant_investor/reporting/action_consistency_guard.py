"""
报告层动作一致性守卫。
"""

from __future__ import annotations

from typing import Any, Iterable

from quant_investor.agent_protocol import ActionLabel


class ActionConsistencyGuard:
    """保证报告文案动作与 IC / risk 约束一致。"""

    _LEVELS = {
        "回避": 0,
        "观察": 1,
        "轻仓试错": 2,
        "持有": 3,
        "买入": 4,
        "强烈买入": 5,
    }
    _ALIASES = {
        "avoid": "回避",
        "sell": "回避",
        "watch": "观察",
        "observe": "观察",
        "hold": "持有",
        "buy": "买入",
        "light_buy": "轻仓试错",
        "probe": "轻仓试错",
        "strong_buy": "强烈买入",
        "回避": "回避",
        "观察": "观察",
        "持有": "持有",
        "买入": "买入",
        "轻仓试错": "轻仓试错",
        "强烈买入": "强烈买入",
    }

    @classmethod
    def guard(
        cls,
        action: ActionLabel | str,
        conclusion: str,
        ic_actions: Iterable[ActionLabel | str],
        risk_action_cap: ActionLabel | str | None = None,
        subject: str = "该标的",
    ) -> dict[str, Any]:
        raw_display_action = cls._normalize_action(action)
        ic_cap = cls._cap_from_ic_actions(ic_actions)
        risk_cap = cls._normalize_action(risk_action_cap) if risk_action_cap is not None else None

        display_action = raw_display_action
        if ic_cap is not None:
            display_action = cls._more_restrictive(display_action, ic_cap)
        if risk_cap is not None:
            display_action = cls._more_restrictive(display_action, risk_cap)

        normalized_conclusion = cls._normalize_conclusion(
            conclusion=conclusion,
            display_action=display_action,
            subject=subject,
        )
        return {
            "display_action": display_action,
            "conclusion": normalized_conclusion,
            "raw_display_action": raw_display_action,
            "ic_action_cap": ic_cap,
            "risk_action_cap": risk_cap,
        }

    @classmethod
    def _normalize_action(cls, value: ActionLabel | str | None) -> str:
        if value is None:
            return "观察"
        if isinstance(value, ActionLabel):
            value = value.value
        text = str(value).strip()
        lowered = text.lower()
        return cls._ALIASES.get(text, cls._ALIASES.get(lowered, "观察"))

    @classmethod
    def _cap_from_ic_actions(cls, actions: Iterable[ActionLabel | str]) -> str | None:
        normalized = [cls._normalize_action(item) for item in actions]
        if not normalized:
            return None
        return min(normalized, key=lambda item: cls._LEVELS[item])

    @classmethod
    def _more_restrictive(cls, left: str, right: str) -> str:
        return left if cls._LEVELS[left] <= cls._LEVELS[right] else right

    @classmethod
    def _normalize_conclusion(cls, conclusion: str, display_action: str, subject: str) -> str:
        text = str(conclusion or "").strip()
        if not text:
            return cls._default_conclusion(display_action, subject)

        cautious_words = ("观察", "轻仓试错")
        aggressive_words = ("强烈买入", "买入")
        if display_action in {"买入", "强烈买入", "持有"} and any(word in text for word in cautious_words):
            return cls._default_conclusion(display_action, subject)
        if display_action in {"观察", "轻仓试错", "回避"} and any(word in text for word in aggressive_words):
            return cls._default_conclusion(display_action, subject)
        return text

    @staticmethod
    def _default_conclusion(display_action: str, subject: str) -> str:
        if display_action == "强烈买入":
            return f"{subject} 当前结构化信号高度一致，可按纪律计划执行买入。"
        if display_action == "买入":
            return f"{subject} 当前结构化信号支持按计划分批买入。"
        if display_action == "持有":
            return f"{subject} 当前更适合持有并继续等待验证。"
        if display_action == "轻仓试错":
            return f"{subject} 当前仍有正向依据，但只适合轻仓试错。"
        if display_action == "回避":
            return f"{subject} 当前风险回报比不足，建议回避。"
        return f"{subject} 当前更适合继续观察，暂不进入激进执行。"
