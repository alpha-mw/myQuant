from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util

from quant_investor.agent_orchestrator import AgentOrchestrator, ControlChainOrchestrator
from quant_investor.agent_protocol import ActionLabel


@dataclass
class _SerializableFixture:
    action: ActionLabel
    symbols: set[str]
    metadata: dict[str, object]


def test_control_chain_orchestrator_is_the_canonical_control_chain_entrypoint():
    assert AgentOrchestrator is ControlChainOrchestrator


def test_agent_orchestrator_persistence_helpers_are_split_and_delegated():
    spec = importlib.util.find_spec("quant_investor.agent_orchestrator_persistence")
    assert spec is not None
    agent_orchestrator_persistence = importlib.import_module(
        "quant_investor.agent_orchestrator_persistence"
    )

    assert (
        ControlChainOrchestrator._serialize
        is agent_orchestrator_persistence.serialize_agent_payload
    )

    payload = _SerializableFixture(
        action=ActionLabel.BUY,
        symbols={"000001.SZ", "600519.SH"},
        metadata={"path": __file__},
    )

    serialized = ControlChainOrchestrator._serialize(payload)

    assert serialized["action"] == "buy"
    assert sorted(serialized["symbols"]) == ["000001.SZ", "600519.SH"]
    assert serialized["metadata"]["path"].endswith("test_control_chain_orchestrator.py")
