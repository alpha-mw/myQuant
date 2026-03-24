"""
架构版本与 schema 常量。
"""

from __future__ import annotations

ARCHITECTURE_VERSION_V8 = "8.0.0-legacy-frozen"
ARCHITECTURE_VERSION_V9 = "9.0.0-current"
ARCHITECTURE_VERSION_V10 = "10.0.0-multi-agent"
ARCHITECTURE_VERSION_CURRENT = ARCHITECTURE_VERSION_V9
ARCHITECTURE_VERSION_LATEST = ARCHITECTURE_VERSION_V10

AGENT_SCHEMA_VERSION = "2026-03-23.agent.v1"
IC_PROTOCOL_VERSION = "2026-03-24.ic-protocol.v1"
REPORT_PROTOCOL_VERSION = "2026-03-24.report-protocol.v1"

BRANCH_SCHEMA_VERSION_V8 = "v8-legacy-llm-debate"
BRANCH_SCHEMA_VERSION_V9 = "v9-fundamental-first-class"
CALIBRATION_SCHEMA_VERSION = "2026-03-22.calibration.v2"
BRANCH_TRACKER_SCHEMA_VERSION = "2026-03-22.branch-tracker.v2"
DEBATE_TEMPLATE_VERSION = "2026-03-22.branch-debate.v2"

LEGACY_BRANCH_ORDER = [
    "kline",
    "quant",
    "llm_debate",
    "intelligence",
    "macro",
]

CURRENT_BRANCH_ORDER = [
    "kline",
    "quant",
    "fundamental",
    "intelligence",
    "macro",
]

LEGACY_BRANCH_WEIGHTS: dict[str, float] = {
    "kline": 0.22,
    "quant": 0.28,
    "llm_debate": 0.15,
    "intelligence": 0.20,
    "macro": 0.15,
}

CURRENT_BRANCH_WEIGHTS: dict[str, float] = {
    "kline": 0.22,
    "quant": 0.28,
    "fundamental": 0.15,
    "intelligence": 0.20,
    "macro": 0.15,
}


def output_version_payload(
    architecture_version: str,
    branch_schema_version: str,
) -> dict[str, str]:
    return {
        "architecture_version": architecture_version,
        "branch_schema_version": branch_schema_version,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "debate_template_version": DEBATE_TEMPLATE_VERSION,
    }
