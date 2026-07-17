"""V16 research eligibility gate.

The gate owns only PIT membership, canonical data readiness, and factor
readiness.  It does not inspect quotes, cash, lots, or human authorization.
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.agent_protocol import EligibilityDecision
from quant_investor.agents.base import BaseAgent


class EligibilityGate(BaseAgent):
    """Fail closed on unconfirmed PIT/data/factor readiness."""

    agent_name = "EligibilityGate"
    protocol_version = "v1"
    _READINESS_FIELDS = ("pit_ready", "data_ready", "factor_ready")

    def run(self, payload: Mapping[str, Any]) -> EligibilityDecision:
        envelope = self.ensure_payload(payload)
        self.require_keys(envelope, "symbol", "readiness")
        symbol = str(envelope["symbol"]).strip()
        if not symbol:
            raise ValueError("EligibilityGate symbol must be non-empty")
        readiness = self.ensure_payload(envelope["readiness"])

        checks: dict[str, bool] = {}
        blockers: list[str] = []
        for field_name in self._READINESS_FIELDS:
            value = readiness.get(field_name)
            checks[field_name] = value is True
            if value is not True:
                blockers.append(f"{field_name}_unconfirmed")
            blockers.extend(self._normalize_blockers(readiness.get(f"{field_name}_blockers")))

        blockers = list(dict.fromkeys(blockers))
        return EligibilityDecision(
            symbol=symbol,
            research_eligible=all(checks.values()) and not blockers,
            pit_ready=checks["pit_ready"],
            data_ready=checks["data_ready"],
            factor_ready=checks["factor_ready"],
            blockers=blockers,
        )

    @staticmethod
    def _normalize_blockers(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("EligibilityGate readiness blockers must be a list")
        return [str(item).strip() for item in value if str(item).strip()]


__all__ = ["EligibilityGate"]
