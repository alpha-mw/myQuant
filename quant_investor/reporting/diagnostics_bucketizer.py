"""
报告层诊断分桶器。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from quant_investor.agent_protocol import BranchVerdict


def dedupe_texts(items: Sequence[str]) -> list[str]:
    """按出现顺序去重并去空。"""

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def format_count(count: int, unit: str) -> str:
    """把裸数字渲染成带单位的文本。"""

    return f"{int(count)} {unit}"


def sanitize_report_text(text: str) -> str:
    """把工程异常和 provider 标记收敛为可展示文案。"""

    normalized = str(text or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return ""
    if "could not infer frequency" in lowered:
        return "部分批次 K 线深度模型未完成频率对齐，已自动回退统计预测。"
    if "provider_missing" in lowered or "snapshot_missing" in lowered:
        return "部分模块当前缺少覆盖，已只计入数据覆盖摘要。"
    if "provider_error" in lowered:
        return "部分数据接口本轮不可用，已只计入数据覆盖摘要。"
    if "timeout" in lowered or "timed out" in lowered:
        return "部分模型阶段超时，已自动保留基础结论。"
    if "traceback" in lowered:
        return "工程异常堆栈已隐藏，系统已记录诊断摘要。"
    if re.search(r"\b(?:valueerror|runtimeerror|typeerror|keyerror|exception)\b", lowered):
        return "工程异常已被收敛为诊断摘要。"
    if lowered.startswith("[debug]") or lowered.startswith("[info]") or lowered.startswith("[warning]"):
        cleaned = re.sub(r"^\[[A-Z]+\]\s*", "", normalized, flags=re.IGNORECASE).strip()
        return f"运行日志摘要：{cleaned}" if cleaned else "运行日志摘要已归档。"
    if lowered == "unknown":
        return "已按默认状态处理。"
    return normalized


class DiagnosticsBucketizer:
    """把 investment_risks / coverage_notes / diagnostic_notes 严格分桶。"""

    def __init__(
        self,
        branch_summaries: Mapping[str, BranchVerdict | Mapping[str, Any]],
        run_diagnostics: Any = None,
    ) -> None:
        self.branch_summaries = dict(branch_summaries)
        self.run_diagnostics = run_diagnostics

    def bucket(self) -> dict[str, Any]:
        branch_count = max(len(self.branch_summaries), 1)
        investment_risks: list[str] = []
        coverage_notes: list[str] = []
        diagnostic_notes: list[str] = []
        risk_branch_count = 0
        coverage_branch_count = 0
        diagnostic_branch_count = 0

        for _, branch in self.branch_summaries.items():
            summary = self._normalize_branch_summary(branch)
            branch_has_risk = False
            branch_has_coverage = False
            branch_has_diagnostic = False

            for note in summary["investment_risks"]:
                sanitized = sanitize_report_text(note)
                lowered = sanitized.lower()
                if not sanitized:
                    continue
                if "缺少覆盖" in sanitized or "数据接口本轮不可用" in sanitized:
                    coverage_notes.append(sanitized)
                    branch_has_coverage = True
                    continue
                if "运行日志摘要" in sanitized or "工程异常" in sanitized:
                    diagnostic_notes.append(sanitized)
                    branch_has_diagnostic = True
                    continue
                investment_risks.append(sanitized)
                branch_has_risk = True

            for note in summary["coverage_notes"]:
                sanitized = sanitize_report_text(note)
                if not sanitized:
                    continue
                coverage_notes.append(sanitized)
                branch_has_coverage = True

            for note in summary["diagnostic_notes"]:
                sanitized = sanitize_report_text(note)
                if not sanitized:
                    continue
                diagnostic_notes.append(sanitized)
                branch_has_diagnostic = True

            risk_branch_count += int(branch_has_risk)
            coverage_branch_count += int(branch_has_coverage)
            diagnostic_branch_count += int(branch_has_diagnostic)

        extra = self._normalize_run_diagnostics(self.run_diagnostics)
        investment_risks.extend(extra["investment_risks"])
        coverage_notes.extend(extra["coverage_notes"])
        diagnostic_notes.extend(extra["diagnostic_notes"])
        if extra["investment_risks"]:
            risk_branch_count = min(branch_count, risk_branch_count + 1)
        if extra["coverage_notes"]:
            coverage_branch_count = min(branch_count, coverage_branch_count + 1)
        if extra["diagnostic_notes"]:
            diagnostic_branch_count = min(branch_count, diagnostic_branch_count + 1)

        investment_risks = dedupe_texts(investment_risks)
        coverage_notes = dedupe_texts(coverage_notes)
        diagnostic_notes = dedupe_texts(diagnostic_notes)

        return {
            "investment_risks": [
                f"共整理 {format_count(len(investment_risks), '条投资风险')}，涉及 {risk_branch_count}/{branch_count} 个分支。"
            ] + investment_risks,
            "coverage_summary": [
                f"共整理 {format_count(len(coverage_notes), '条覆盖说明')}，涉及 {coverage_branch_count}/{branch_count} 个分支。"
            ] + coverage_notes,
            "appendix_diagnostics": [
                f"共归档 {format_count(len(diagnostic_notes), '条工程诊断')}，涉及 {diagnostic_branch_count}/{branch_count} 个分支。"
            ] + [
                f"{note}（1 条诊断，1/{branch_count} 分支）"
                for note in diagnostic_notes
            ],
            "counts": {
                "branch_count": branch_count,
                "investment_risk_count": len(investment_risks),
                "coverage_count": len(coverage_notes),
                "diagnostic_count": len(diagnostic_notes),
            },
        }

    @staticmethod
    def _normalize_branch_summary(branch: BranchVerdict | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(branch, BranchVerdict):
            return {
                "investment_risks": list(branch.investment_risks),
                "coverage_notes": list(branch.coverage_notes),
                "diagnostic_notes": list(branch.diagnostic_notes),
            }
        if not isinstance(branch, Mapping):
            return {
                "investment_risks": [],
                "coverage_notes": [],
                "diagnostic_notes": [],
            }
        return {
            "investment_risks": [str(item) for item in branch.get("investment_risks", branch.get("risks", []))],
            "coverage_notes": [str(item) for item in branch.get("coverage_notes", [])],
            "diagnostic_notes": [str(item) for item in branch.get("diagnostic_notes", [])],
        }

    @staticmethod
    def _normalize_run_diagnostics(payload: Any) -> dict[str, list[str]]:
        result = {
            "investment_risks": [],
            "coverage_notes": [],
            "diagnostic_notes": [],
        }
        if payload is None:
            return result
        if isinstance(payload, Mapping):
            result["investment_risks"].extend(
                sanitize_report_text(item)
                for item in payload.get("investment_risks", [])
            )
            result["coverage_notes"].extend(
                sanitize_report_text(item)
                for item in payload.get("coverage_notes", [])
            )
            result["diagnostic_notes"].extend(
                sanitize_report_text(item)
                for item in payload.get("diagnostic_notes", payload.get("execution_logs", []))
            )
            return {
                key: dedupe_texts(values)
                for key, values in result.items()
            }
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            result["diagnostic_notes"] = dedupe_texts(
                sanitize_report_text(item) for item in payload
            )
        elif str(payload).strip():
            result["diagnostic_notes"] = [sanitize_report_text(str(payload))]
        return result
