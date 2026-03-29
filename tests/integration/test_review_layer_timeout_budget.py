from __future__ import annotations

from types import MethodType

from quant_investor.agents.agent_contracts import BaseBranchAgentOutput, MasterAgentOutput, RiskAgentOutput
from quant_investor.agents.orchestrator import AgentOrchestrator, _AGENT_REGISTRY
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.versioning import CURRENT_BRANCH_ORDER
import quant_investor.agents.orchestrator as orchestrator_module


class _TimedAwaitable:
    def __init__(self, *, required_timeout: float, result):
        self.required_timeout = required_timeout
        self._result = result

    def __await__(self):
        async def _resolve():
            return self._result

        return _resolve().__await__()


class _TimedBranchAgent:
    def __init__(self, branch_name: str, **kwargs):
        self.branch_name = branch_name

    def analyze(self, agent_input):
        required_timeout = 40.0 if self.branch_name in {"kline", "fundamental", "macro"} else 5.0
        return _TimedAwaitable(
            required_timeout=required_timeout,
            result=BaseBranchAgentOutput(
                branch_name=self.branch_name,
                conviction="buy",
                conviction_score=0.2,
                confidence=0.7,
            ),
        )


class _RequestBudgetAwareBranchAgent:
    observed_timeouts: dict[str, float] = {}
    observed_max_tokens: dict[str, int] = {}

    def __init__(self, branch_name: str, llm_client, max_tokens: int, **kwargs):
        self.branch_name = branch_name
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    async def analyze(self, agent_input):
        self.observed_timeouts[self.branch_name] = float(self.llm_client.timeout)
        self.observed_max_tokens[self.branch_name] = int(self.max_tokens)
        required_timeout = 40.0 if self.branch_name == "kline" else 5.0
        if float(self.llm_client.timeout) < required_timeout:
            raise orchestrator_module.asyncio.TimeoutError()
        return BaseBranchAgentOutput(
            branch_name=self.branch_name,
            conviction="buy",
            conviction_score=0.2,
            confidence=0.7,
        )


async def _fake_wait_for(awaitable, timeout: float):
    required_timeout = getattr(awaitable, "required_timeout", 0.0)
    if timeout < required_timeout:
        raise orchestrator_module.asyncio.TimeoutError()
    return await awaitable


def _make_branch_result(branch_name: str) -> BranchResult:
    return BranchResult(
        branch_name=branch_name,
        final_score=0.1,
        final_confidence=0.6,
        explanation=f"{branch_name} branch",
        symbol_scores={"000001.SZ": 0.1},
        signals={"mode": "test"},
    )


def test_review_layer_budget_allows_slow_branch_agents(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "has_any_provider", lambda: True)
    monkeypatch.setattr(orchestrator_module, "has_provider_for_model", lambda _model: True)
    monkeypatch.setattr(orchestrator_module.asyncio, "wait_for", _fake_wait_for)

    for branch_name in CURRENT_BRANCH_ORDER:
        monkeypatch.setitem(_AGENT_REGISTRY, branch_name, _TimedBranchAgent)

    orchestrator = AgentOrchestrator(
        branch_model="deepseek-chat",
        master_model="deepseek-chat",
        timeout_per_agent=30.0,
        master_timeout=60.0,
    )

    async def _fake_risk_agent(self, *args, **kwargs):
        return RiskAgentOutput(risk_assessment="acceptable")

    async def _fake_master_agent(self, *args, **kwargs):
        return MasterAgentOutput(final_conviction="buy", final_score=0.2, confidence=0.7)

    orchestrator._run_risk_agent = MethodType(_fake_risk_agent, orchestrator)
    orchestrator._run_master_agent = MethodType(_fake_master_agent, orchestrator)

    branch_results = {
        branch_name: _make_branch_result(branch_name)
        for branch_name in CURRENT_BRANCH_ORDER
    }

    result = orchestrator.enhance_sync(
        branch_results=branch_results,
        calibrated_signals={},
        risk_result={},
        ensemble_output={},
        data_bundle=UnifiedDataBundle(stock_data={"000001.SZ": {}}),
        market_regime="neutral",
        algorithmic_strategy=None,
    )

    assert AgentOrchestrator.compute_outer_timeout(
        30.0,
        max_retries=2,
        cushion_seconds=AgentOrchestrator._BRANCH_TIMEOUT_CUSHION_SECONDS,
    ) == 71.0
    assert all(result.branch_agent_outputs[name] is not None for name in CURRENT_BRANCH_ORDER)
    assert result.agent_layer_success is True


def test_kline_branch_uses_extended_request_budget(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "has_any_provider", lambda: True)
    monkeypatch.setattr(orchestrator_module, "has_provider_for_model", lambda _model: True)

    _RequestBudgetAwareBranchAgent.observed_timeouts = {}
    _RequestBudgetAwareBranchAgent.observed_max_tokens = {}
    for branch_name in CURRENT_BRANCH_ORDER:
        monkeypatch.setitem(_AGENT_REGISTRY, branch_name, _RequestBudgetAwareBranchAgent)

    orchestrator = AgentOrchestrator(
        branch_model="deepseek-chat",
        master_model="deepseek-chat",
        timeout_per_agent=30.0,
        master_timeout=60.0,
    )

    async def _fake_risk_agent(self, *args, **kwargs):
        return RiskAgentOutput(risk_assessment="acceptable")

    async def _fake_master_agent(self, *args, **kwargs):
        return MasterAgentOutput(final_conviction="buy", final_score=0.2, confidence=0.7)

    orchestrator._run_risk_agent = MethodType(_fake_risk_agent, orchestrator)
    orchestrator._run_master_agent = MethodType(_fake_master_agent, orchestrator)

    branch_results = {
        branch_name: _make_branch_result(branch_name)
        for branch_name in CURRENT_BRANCH_ORDER
    }

    result = orchestrator.enhance_sync(
        branch_results=branch_results,
        calibrated_signals={},
        risk_result={},
        ensemble_output={},
        data_bundle=UnifiedDataBundle(stock_data={"000001.SZ": {}}),
        market_regime="neutral",
        algorithmic_strategy=None,
    )

    assert _RequestBudgetAwareBranchAgent.observed_timeouts["kline"] == 45.0
    assert _RequestBudgetAwareBranchAgent.observed_max_tokens["kline"] == 600
    assert _RequestBudgetAwareBranchAgent.observed_timeouts["quant"] == 30.0
    assert result.branch_agent_outputs["kline"] is not None
    assert result.agent_layer_success is True
