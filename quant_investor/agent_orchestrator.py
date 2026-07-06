"""Backward-compatible import shim for the renamed control chain module."""

from __future__ import annotations

import warnings

from quant_investor.control_chain import AgentOrchestrator, ControlChainOrchestrator


warnings.warn(
    "quant_investor.agent_orchestrator is deprecated; import "
    "quant_investor.control_chain instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AgentOrchestrator", "ControlChainOrchestrator"]
