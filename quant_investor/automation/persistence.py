from __future__ import annotations

from datetime import datetime
from typing import Any

from quant_investor.automation import daily_runner as _runner


class PersistenceManager:
    """仅将当前版本 daily 报告保存到文件系统。"""

    def save(
        self,
        report_md: str,
        pipeline_result: dict[str, Any],
        config: dict[str, Any],
    ) -> str:
        """保存报告到 v14 输出目录，返回报告路径。"""
        report_dir = _runner.resolve_daily_report_dir(config.get("report_dir"))
        report_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_analysis.md"
        report_path = report_dir / filename
        report_path.write_text(report_md, encoding="utf-8")
        _runner.log.info("报告已保存: %s", report_path)
        return str(report_path)
