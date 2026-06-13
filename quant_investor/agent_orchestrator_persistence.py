"""Persistence helpers for the v13 control-chain orchestrator."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    BranchVerdict,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
)


def serialize_agent_payload(value: Any) -> Any:
    """Convert protocol dataclasses/enums into JSON-safe payloads."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            key: serialize_agent_payload(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Mapping):
        return {
            str(key): serialize_agent_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [serialize_agent_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def persist_agent_outputs(
    *,
    macro_verdict: BranchVerdict,
    research_by_symbol: Mapping[str, Mapping[str, BranchVerdict]],
    risk_by_symbol: Mapping[str, RiskDecision],
    ic_by_symbol: Mapping[str, ICDecision],
    portfolio_plan: PortfolioPlan,
    report_bundle: ReportBundle,
    persist_dir: str | Path | None,
    version_info: Mapping[str, str],
) -> dict[str, str]:
    base_dir = (
        Path(persist_dir)
        if persist_dir is not None
        else Path(tempfile.mkdtemp(prefix="quant_investor_agent_orchestrator_"))
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "macro_verdict": run_dir / "macro_verdict.json",
        "research_by_symbol": run_dir / "research_by_symbol.json",
        "risk_by_symbol": run_dir / "risk_by_symbol.json",
        "ic_by_symbol": run_dir / "ic_by_symbol.json",
        "portfolio_plan": run_dir / "portfolio_plan.json",
        "report_bundle": run_dir / "report_bundle.json",
        "markdown_report": run_dir / "report.md",
    }

    json_payloads = {
        "macro_verdict": macro_verdict,
        "research_by_symbol": research_by_symbol,
        "risk_by_symbol": risk_by_symbol,
        "ic_by_symbol": ic_by_symbol,
        "portfolio_plan": portfolio_plan,
        "report_bundle": report_bundle,
    }
    for key, payload in json_payloads.items():
        files[key].write_text(
            json.dumps(serialize_agent_payload(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    files["markdown_report"].write_text(report_bundle.markdown_report, encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                **dict(version_info),
                "files": {key: str(path) for key, path in files.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        **{key: str(path) for key, path in files.items()},
    }


__all__ = ["persist_agent_outputs", "serialize_agent_payload"]
