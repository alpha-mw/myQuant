"""Regression tests: RiskGuard hard veto must trigger on Chinese risk texts.

CN-market branches emit ``investment_risks`` in Chinese. Before the bilingual
``DEFAULT_VETO_KEYWORDS`` set, texts such as "存在退市风险" sailed past the
English-only defaults and the one-vote veto silently failed for A-share runs.
"""

from __future__ import annotations

import pytest

from quant_investor.agent_protocol import ActionLabel, AgentStatus, BranchVerdict
from quant_investor.agents.risk_guard import DEFAULT_VETO_KEYWORDS, RiskGuard


SYMBOL = "000001.SZ"


def _payload(
    *,
    investment_risks: list[str] | None = None,
    constraints: dict[str, object] | None = None,
) -> dict[str, object]:
    base_constraints: dict[str, object] = {
        "gross_exposure_cap": 0.90,
        "max_weight": 0.20,
        "risk_flags": [],
    }
    base_constraints.update(constraints or {})
    return {
        "branch_verdicts": {
            "quant": BranchVerdict(
                agent_name="quant",
                thesis="quant ok",
                symbol=SYMBOL,
                final_score=0.35,
                final_confidence=0.75,
                investment_risks=list(investment_risks or []),
            )
        },
        "macro_verdict": BranchVerdict(agent_name="macro", thesis="macro stable"),
        "portfolio_state": {
            "candidate_symbols": [SYMBOL],
            "current_weights": {},
        },
        "constraints": base_constraints,
    }


def _assert_hard_veto(decision) -> None:
    assert decision.hard_veto is True
    assert decision.veto is True
    assert decision.status == AgentStatus.VETOED
    assert decision.action_cap == ActionLabel.HOLD
    assert decision.gross_exposure_cap == pytest.approx(0.0)
    assert decision.max_weight == pytest.approx(0.0)
    assert SYMBOL in decision.blocked_symbols


@pytest.mark.parametrize(
    "risk_text",
    [
        # Delisting risk phrased purely in Chinese (退市).
        "存在退市风险，可能被交易所实施退市风险警示",
        # Regulatory fraud investigation (造假 / 立案调查).
        "公司因涉嫌财务造假被证监会立案调查",
        # Explicit *ST designation marker only.
        "标的当前为*ST状态",
        # Risk-warning designation wording without the *ST marker (风险警示).
        "已被实施其他风险警示",
        # Trading halt (停牌).
        "标的临时停牌，复牌时间未定",
        # Liquidity freeze (流动性冻结).
        "极端情况下可能出现流动性冻结",
        # Explicit one-vote veto phrase (一票否决).
        "情报分支建议一票否决该标的",
    ],
)
def test_chinese_only_risk_text_triggers_hard_veto(risk_text: str) -> None:
    decision = RiskGuard().run(_payload(investment_risks=[risk_text]))
    _assert_hard_veto(decision)


def test_english_default_keywords_still_trigger_hard_veto() -> None:
    decision = RiskGuard().run(
        _payload(investment_risks=["ongoing fraud investigation by regulator"])
    )
    _assert_hard_veto(decision)


def test_chinese_risk_flag_in_constraints_triggers_hard_veto() -> None:
    decision = RiskGuard().run(
        _payload(constraints={"risk_flags": ["监管公告：涉嫌欺诈发行"]})
    )
    _assert_hard_veto(decision)


def test_benign_chinese_risk_text_does_not_veto() -> None:
    decision = RiskGuard().run(
        _payload(investment_risks=["行业竞争加剧，毛利率承压"])
    )

    assert decision.hard_veto is False
    assert decision.status != AgentStatus.VETOED
    # A single non-veto risk text must not consume the risk budget.
    assert decision.gross_exposure_cap == pytest.approx(0.90)
    assert decision.max_weight == pytest.approx(0.20)
    assert SYMBOL not in decision.blocked_symbols


def test_explicit_veto_keywords_override_replaces_defaults() -> None:
    override = {"veto_keywords": ["黑天鹅"]}

    # Default triggers are replaced by the explicit override...
    not_vetoed = RiskGuard().run(
        _payload(investment_risks=["存在退市风险"], constraints=dict(override))
    )
    assert not_vetoed.hard_veto is False

    # ...and the override keyword itself still vetoes.
    vetoed = RiskGuard().run(
        _payload(investment_risks=["突发黑天鹅事件"], constraints=dict(override))
    )
    _assert_hard_veto(vetoed)


def test_blank_veto_keywords_are_ignored() -> None:
    # An empty keyword would substring-match everything; it must be dropped
    # instead of forcing a permanent veto.
    decision = RiskGuard().run(
        _payload(
            investment_risks=["行业竞争加剧"],
            constraints={"veto_keywords": ["", "   "]},
        )
    )
    assert decision.hard_veto is False


def test_default_keyword_set_is_bilingual() -> None:
    lowered = {keyword.lower() for keyword in DEFAULT_VETO_KEYWORDS}
    # English parity terms retained.
    assert {"fraud", "halt", "delist", "veto", "liquidity freeze"} <= lowered
    # Chinese regulatory / delisting triggers present.
    assert {"退市", "停牌", "造假", "立案调查", "风险警示", "*st"} <= lowered
    # No blank entries that would match every text.
    assert all(keyword.strip() for keyword in DEFAULT_VETO_KEYWORDS)
