"""Research run endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from web.models.research_models import (
    ResearchHistoryResponse,
    ResearchHistoryItem,
    ResearchJobResponse,
    ResearchReportResponse,
    ResearchRunRequest,
    RecentRunSummary,
    StartupContextResponse,
)
from web.services.research_runner import job_manager
from web.services.run_history_store import history_store

router = APIRouter(prefix="/api/research", tags=["research"])


def _encode_sse_event(event: str, data: str) -> str:
    lines = [f"event: {event}"]
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


@router.post("/run", response_model=ResearchJobResponse)
async def submit_run(request: ResearchRunRequest):
    loop = asyncio.get_event_loop()
    job = job_manager.submit_job(request, loop)

    return ResearchJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )


# ── Static paths must come before /{job_id} ──────────────────────────────────

@router.get("/history/list", response_model=ResearchHistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    market: Optional[str] = Query(None),
):
    items, total = history_store.get_history(page=page, per_page=per_page, market=market)
    return ResearchHistoryResponse(
        items=[ResearchHistoryItem(**item) for item in items],
        total=total,
    )


@router.get("/startup-context", response_model=StartupContextResponse)
async def get_startup_context():
    """
    Return recent completed runs, suggested trades, and a bounded recall summary
    for workspace initialisation.  Loaded automatically on page open.
    """
    ctx = history_store.get_startup_context(recent_n=5)
    runs = [
        RecentRunSummary(
            job_id=r["job_id"],
            created_at=r["created_at"],
            market=r["market"],
            stock_pool=r["stock_pool"],
            status=r["status"],
            total_time=r.get("total_time"),
            recall_context=r.get("recall_context", {}),
            selection_meta=r.get("selection_meta", {}),
        )
        for r in ctx["recent_runs"]
    ]
    return StartupContextResponse(
        recent_runs=runs,
        suggested_trades=ctx["suggested_trades"],
        recall_summary=ctx["recall_summary"],
    )


# ── Dynamic /{job_id} routes ──────────────────────────────────────────────────

@router.get("/{job_id}", response_model=ResearchJobResponse)
async def get_job(job_id: str):
    # Check in-memory first (active jobs)
    job = job_manager.get_job(job_id)
    if job:
        return ResearchJobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            progress_pct=job.progress_pct,
            error=job.error,
            result_summary=job.result_summary,
        )

    # Fall back to history store
    run = history_store.get_run(job_id)
    if run:
        summary = None
        if run.get("result_summary_json"):
            try:
                summary = json.loads(run["result_summary_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return ResearchJobResponse(
            job_id=run["job_id"],
            status=run["status"],
            created_at=run["created_at"],
            progress_pct=1.0 if run["status"] in {"completed", "failed"} else 0.0,
            error=run.get("error") or None,
            result_summary=summary,
        )

    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/{job_id}/stream")
async def stream_logs(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or already finished")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in job_manager.stream_logs(job_id):
            yield _encode_sse_event(event["event"], event["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/report", response_model=ResearchReportResponse)
async def get_report(job_id: str):
    # Check in-memory first
    job = job_manager.get_job(job_id)
    if job and job.report_markdown:
        return ResearchReportResponse(markdown=job.report_markdown)

    # Fall back to history
    markdown = history_store.get_report(job_id)
    if markdown is not None:
        return ResearchReportResponse(markdown=markdown)

    raise HTTPException(status_code=404, detail="Report not found")


@router.delete("/{job_id}")
async def delete_run(job_id: str):
    deleted = history_store.delete_run(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True, "deleted": job_id}
