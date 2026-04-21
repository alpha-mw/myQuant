"""Async job manager wrapping QuantInvestor.run() for the web API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from web.models.research_models import ResearchRunRequest
from web.services.run_history_store import history_store
from quant_investor.research_run_config import ResearchRunConfig


@dataclass
class ResearchJob:
    job_id: str
    status: str  # queued | running | completed | failed
    created_at: str
    request: ResearchRunRequest
    log_lines: list[str] = field(default_factory=list)
    log_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    progress_pct: float = 0.0
    phase_key: str = ""
    phase_label: str = ""
    result_summary: Optional[dict[str, Any]] = None
    report_markdown: str = ""
    error: Optional[str] = None
    total_time: Optional[float] = None


# Each entry: (keyword, progress_fraction, phase_key, phase_label)
_PHASE_PROGRESS: list[tuple[str, float, str, str]] = [
    ("data",        0.10, "data",        "数据获取"),
    ("kline",       0.25, "kline",       "K线预测"),
    ("quant",       0.35, "quant",       "量化因子"),
    ("fundamental", 0.45, "fundamental", "基本面分析"),
    ("intelligence",0.55, "intelligence","智能融合"),
    ("macro",       0.60, "macro",       "宏观风险"),
    ("risk",        0.65, "risk",        "风险评估"),
    ("calibrat",    0.70, "calibration", "信号校准"),
    ("review",      0.75, "review",      "分支复盘"),
    ("agent",       0.80, "agent",       "Agent 层"),
    ("portfolio",   0.85, "portfolio",   "组合构建"),
    ("narrator",    0.90, "narrator",    "报告生成"),
    ("report",      0.95, "report",      "输出报告"),
]


def _estimate_phase(message: str) -> Optional[tuple[float, str, str]]:
    """Return (progress_pct, phase_key, phase_label) from a log message, or None."""
    lower = message.lower()
    for keyword, pct, key, label in reversed(_PHASE_PROGRESS):
        if keyword in lower:
            return pct, key, label
    return None


class _JobLogHandler(logging.Handler):
    """Bridges QuantInvestor logger to the job's log queue."""

    def __init__(self, job: ResearchJob, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.job = job
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.job.log_lines.append(msg)

            phase = _estimate_phase(msg)
            if phase is not None:
                pct, key, label = phase
                if pct > self.job.progress_pct:
                    self.job.progress_pct = pct
                    self.job.phase_key = key
                    self.job.phase_label = label

            if self.job.log_queue is not None:
                self.loop.call_soon_threadsafe(self.job.log_queue.put_nowait, msg)
        except Exception:
            pass


def _extract_trades(result: Any, stock_pool: list[str]) -> list[dict[str, Any]]:
    """
    Best-effort extraction of suggested trades from a QuantInvestorPipelineResult.
    Returns a list of {symbol, direction, rationale} dicts.
    Only subagent/master layer suggestions are extracted; deterministic
    RiskGuard/PortfolioConstructor outputs are NOT included.
    """
    trades: list[dict[str, Any]] = []
    try:
        master_output = getattr(result, "master_review_output", None)
        if master_output is None:
            return trades
        conviction_drivers = list(getattr(master_output, "conviction_drivers", []) or [])
        top_picks = getattr(master_output, "top_picks", []) or []
        for pick in top_picks:
            symbol = getattr(pick, "symbol", None) or ""
            action = str(getattr(pick, "action", "") or "").strip().lower()
            rationale = str(getattr(pick, "rationale", "") or "").strip()
            if symbol and action in {"buy", "sell"}:
                if not rationale:
                    rationale = "；".join(str(item).strip() for item in conviction_drivers[:2] if str(item).strip())
                trades.append({
                    "trade_id": uuid.uuid4().hex[:16],
                    "symbol": symbol,
                    "direction": action,
                    "rationale": rationale[:500],
                })
    except Exception:
        pass
    return trades


def _save_workspace_learning(
    job_id: str,
    created_at: str,
    stock_pool: list[str],
    market: str,
    trades: list[dict[str, Any]],
    recall_context: dict[str, Any],
) -> None:
    """Auto-save one pending workspace trade case per suggested symbol."""
    if not trades:
        return
    base_dir = Path("data") / "workspace_learning"
    base_dir.mkdir(parents=True, exist_ok=True)
    for trade in trades:
        symbol = trade.get("symbol", "unknown")
        filename = base_dir / f"{job_id}_{symbol}.json"
        payload = {
            "job_id": job_id,
            "created_at": created_at,
            "market": market,
            "stock_pool": stock_pool,
            "symbol": symbol,
            "direction": trade.get("direction", "buy"),
            "rationale": trade.get("rationale", ""),
            "outcome_status": "pending",
            "recall_context": recall_context,
        }
        try:
            filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass


def _build_recall_context(result: Any, req: ResearchRunRequest) -> dict[str, Any]:
    """Build a bounded structured recall context from a completed result."""
    recall: dict[str, Any] = {
        "market": req.market,
        "stock_pool": req.stock_pool[:10],  # bounded to first 10
        "risk_level": req.risk_level,
        "llm_usage": {},
        "conviction": "",
        "top_picks": [],
    }
    try:
        usage = result.llm_usage_summary
        recall["llm_usage"] = {
            "total_calls": getattr(usage, "total_calls", getattr(usage, "call_count", 0)),
            "estimated_cost_usd": round(getattr(usage, "estimated_cost_usd", 0.0), 4),
        }
    except Exception:
        pass
    try:
        agent_strategy = getattr(result, "master_review_output", None)
        if agent_strategy is None:
            return recall
        recall["conviction"] = str(getattr(agent_strategy, "final_conviction", ""))
        recall["top_picks"] = [
            {
                "symbol": getattr(p, "symbol", ""),
                "action": getattr(p, "action", ""),
                "rationale": getattr(p, "rationale", ""),
            }
            for p in (getattr(agent_strategy, "top_picks", []) or [])[:5]
        ]
        conviction_drivers = [
            str(item).strip()
            for item in (getattr(agent_strategy, "conviction_drivers", []) or [])[:3]
            if str(item).strip()
        ]
        if conviction_drivers:
            recall["conviction_drivers"] = conviction_drivers
        debate_resolution = [
            str(item).strip()
            for item in (getattr(agent_strategy, "debate_resolution", []) or [])[:3]
            if str(item).strip()
        ]
        if debate_resolution:
            recall["debate_resolution"] = debate_resolution
    except Exception:
        pass
    return recall


def _serialize_usage_summary(summary: Any) -> dict[str, Any]:
    return {
        "total_calls": getattr(summary, "total_calls", getattr(summary, "call_count", 0)),
        "total_prompt_tokens": getattr(summary, "total_prompt_tokens", getattr(summary, "prompt_tokens", 0)),
        "total_completion_tokens": getattr(summary, "total_completion_tokens", getattr(summary, "completion_tokens", 0)),
        "estimated_cost_usd": getattr(summary, "estimated_cost_usd", 0.0),
        "failed_count": getattr(summary, "failed_count", 0),
        "fallback_count": getattr(summary, "fallback_count", 0),
    }


def _compact_trace_summary(payload: Any) -> dict[str, Any]:
    trace = payload if isinstance(payload, dict) else getattr(payload, "to_dict", lambda: {})()
    if not isinstance(trace, dict):
        trace = {}
    steps: list[dict[str, Any]] = []
    for step in list(trace.get("steps", []) or [])[:8]:
        step_map = step if isinstance(step, dict) else getattr(step, "to_dict", lambda: {})()
        if not isinstance(step_map, dict):
            step_map = {}
        steps.append(
            {
                "stage": step_map.get("stage", ""),
                "role": step_map.get("role", ""),
                "success": bool(step_map.get("success", False)),
                "conclusion": str(step_map.get("conclusion", ""))[:160],
            }
        )
    return {
        "model_roles": trace.get("model_roles", {}),
        "key_parameters": {
            "selected_count": trace.get("key_parameters", {}).get("selected_count", 0),
            "target_exposure": trace.get("key_parameters", {}).get("target_exposure", 0.0),
            "max_single_weight": trace.get("key_parameters", {}).get("max_single_weight", 0.0),
            "data_quality_issue_count": trace.get("key_parameters", {}).get("data_quality_issue_count", 0),
        },
        "final_deterministic_outcome": trace.get("final_deterministic_outcome", {}),
        "steps": steps,
    }


def _compact_whatif_summary(payload: Any) -> dict[str, Any]:
    plan = payload if isinstance(payload, dict) else getattr(payload, "to_dict", lambda: {})()
    if not isinstance(plan, dict):
        plan = {}
    scenarios: list[dict[str, Any]] = []
    for scenario in list(plan.get("scenarios", []) or [])[:6]:
        scenario_map = scenario if isinstance(scenario, dict) else getattr(scenario, "to_dict", lambda: {})()
        if not isinstance(scenario_map, dict):
            scenario_map = {}
        scenarios.append(
            {
                "scenario_name": scenario_map.get("scenario_name", ""),
                "trigger": str(scenario_map.get("trigger", ""))[:120],
                "action": str(scenario_map.get("action", ""))[:120],
                "rerun_full_market_daily_path": bool(scenario_map.get("rerun_full_market_daily_path", False)),
            }
        )
    return {
        "generated_by": plan.get("generated_by", ""),
        "scenario_count": len(plan.get("scenarios", []) or []),
        "scenarios": scenarios,
        "metadata": plan.get("metadata", {}),
    }


def _run_research(job: ResearchJob, loop: asyncio.AbstractEventLoop) -> None:
    """Execute QuantInvestor.run() in a worker thread."""
    from quant_investor.pipeline import QuantInvestor

    handler = _JobLogHandler(job, loop)
    handler.setFormatter(logging.Formatter("%(message)s"))

    qi_logger = logging.getLogger("QuantInvestor")
    qi_logger.addHandler(handler)

    job.status = "running"
    t0 = time.time()
    final_status = "failed"
    recall_context: dict[str, Any] = {}
    trades: list[dict[str, Any]] = []
    review_recall_context: dict[str, Any] = {}
    trace_summary: dict[str, Any] = {}
    whatif_summary: dict[str, Any] = {}

    try:
        req = job.request
        review_recall_context = history_store.build_review_recall_context(
            market=req.market,
            stock_pool=req.stock_pool,
        )
        run_config = ResearchRunConfig.from_mapping(
            req.model_dump(),
            recall_context=review_recall_context,
        )
        investor = QuantInvestor(**run_config.to_quant_investor_kwargs(verbose=True))
        result = investor.run()
        trace_summary = _compact_trace_summary(getattr(result, "execution_trace", None))
        whatif_summary = _compact_whatif_summary(getattr(result, "what_if_plan", None))

        job.total_time = time.time() - t0
        job.progress_pct = 1.0
        job.phase_key = "done"
        job.phase_label = "完成"
        job.report_markdown = result.final_report or ""

        recall_context = _build_recall_context(result, req)
        trades = _extract_trades(result, req.stock_pool)
        execution_log = list(getattr(result, "execution_log", []) or [])

        job.result_summary = {
            "total_time": job.total_time,
            "layer_timings": result.layer_timings,
            "execution_log_excerpt": execution_log[:12],
            "execution_log_count": len(execution_log),
            "llm_usage_summary": _serialize_usage_summary(getattr(result, "llm_usage_summary", None)),
            "llm_effective_summary": _serialize_usage_summary(getattr(result, "llm_effective_summary", None)),
            "llm_usage_session_id": str(getattr(result, "llm_usage_session_id", "") or ""),
            "market": req.market,
            "stock_pool": req.stock_pool,
            "data_snapshot": dict(getattr(result, "data_snapshot", {}) or {}),
            "trace_summary": trace_summary,
            "whatif_summary": whatif_summary,
        }
        final_status = "completed"

    except Exception as exc:
        job.total_time = time.time() - t0
        job.error = str(exc)

    finally:
        if job.result_summary is None:
            job.result_summary = {
                "total_time": job.total_time,
                "market": job.request.market,
                "stock_pool": job.request.stock_pool,
            }

        req = job.request
        selection_meta: dict[str, Any] = {
            "stock_input_mode": req.stock_input_mode,
            "universe_keys": req.universe_keys,
            "universe_operation": req.universe_operation,
            "resolved_count": len(req.stock_pool),
        }

        history_store.save_run(
            job_id=job.job_id,
            created_at=job.created_at,
            status=final_status,
            request_json=job.request.model_dump_json(),
            report_markdown=job.report_markdown,
            result_summary_json=json.dumps(job.result_summary or {}),
            report_path="",
            trace_summary_json=json.dumps(trace_summary, ensure_ascii=False),
            whatif_summary_json=json.dumps(whatif_summary, ensure_ascii=False),
            total_time=job.total_time,
            market=job.request.market,
            stock_pool=json.dumps(job.request.stock_pool),
            risk_level=job.request.risk_level,
            preset_id=job.request.preset_id or "",
            error=job.error or "",
            selection_meta_json=json.dumps(selection_meta),
            recall_context_json=json.dumps(recall_context),
        )

        if final_status == "completed" and trades:
            history_store.save_trade_records(job.job_id, trades)
            _save_workspace_learning(
                job_id=job.job_id,
                created_at=job.created_at,
                stock_pool=job.request.stock_pool,
                market=job.request.market,
                trades=trades,
                recall_context=recall_context,
            )

        job.status = final_status
        qi_logger.removeHandler(handler)
        if job.log_queue is not None:
            loop.call_soon_threadsafe(job.log_queue.put_nowait, "__DONE__")


class ResearchJobManager:
    """Manages research jobs in a thread pool."""

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def submit_job(
        self, request: ResearchRunRequest, loop: asyncio.AbstractEventLoop
    ) -> ResearchJob:
        job_id = uuid.uuid4().hex[:12]
        job = ResearchJob(
            job_id=job_id,
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            request=request,
            log_queue=asyncio.Queue(),
        )
        with self._lock:
            self._jobs[job_id] = job
        history_store.save_run(
            job_id=job.job_id,
            created_at=job.created_at,
            status=job.status,
            request_json=request.model_dump_json(),
            market=request.market,
            stock_pool=json.dumps(request.stock_pool),
            risk_level=request.risk_level,
            preset_id=request.preset_id or "",
            error="",
            selection_meta_json=json.dumps({
                "stock_input_mode": request.stock_input_mode,
                "universe_keys": request.universe_keys,
                "universe_operation": request.universe_operation,
                "resolved_count": len(request.stock_pool),
            }),
            trace_summary_json=json.dumps({}),
            whatif_summary_json=json.dumps({}),
        )
        self._executor.submit(_run_research, job, loop)
        return job

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    async def stream_logs(self, job_id: str) -> AsyncGenerator[dict[str, str], None]:
        job = self.get_job(job_id)
        if job is None:
            yield {"event": "error", "data": "Job not found"}
            return

        # First, replay any log lines already collected
        for line in list(job.log_lines):
            yield {"event": "log", "data": line}

        if job.log_queue is None or job.status in ("completed", "failed"):
            final = "completed" if job.status == "completed" else "failed"
            yield {
                "event": final,
                "data": json.dumps({"status": job.status, "error": job.error}),
            }
            return

        # Stream new log lines
        while True:
            try:
                msg = await asyncio.wait_for(job.log_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": ""}
                if job.status in ("completed", "failed"):
                    break
                continue

            if msg == "__DONE__":
                break
            yield {"event": "log", "data": msg}
            if job.progress_pct > 0:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "progress_pct": round(job.progress_pct, 3),
                        "phase_key": job.phase_key,
                        "phase_label": job.phase_label,
                    }),
                }

        final_event = "completed" if job.status == "completed" else "failed"
        yield {
            "event": final_event,
            "data": json.dumps(
                {"status": job.status, "total_time": job.total_time, "error": job.error}
            ),
        }


job_manager = ResearchJobManager()
