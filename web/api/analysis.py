"""Analysis API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from web.models.analysis_models import (
    AnalysisDeleteResponse,
    AnalysisJobResponse,
    AnalysisHistoryResponse,
    AnalysisOptionsResponse,
    AnalysisResult,
    AnalysisRunRequest,
    AnalysisRunResponse,
)
from web.services import analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/history", response_model=AnalysisHistoryResponse)
def get_analysis_history(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None),
    market: str | None = Query(None),
):
    result = analysis_service.list_analysis_history(
        limit=limit, offset=offset, search=search or None, market=market or None,
    )
    return AnalysisHistoryResponse(items=result["items"], total=result["total"])


@router.get("/options", response_model=AnalysisOptionsResponse)
def get_analysis_options():
    return AnalysisOptionsResponse(**analysis_service.get_analysis_options())


@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job(job_id: str):
    result = analysis_service.get_analysis_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return AnalysisJobResponse(**result)


@router.get("/jobs", response_model=list[AnalysisJobResponse])
def list_analysis_jobs(limit: int = Query(8, ge=1, le=50)):
    return [AnalysisJobResponse(**item) for item in analysis_service.get_recent_jobs(limit=limit)]


@router.delete("/history", response_model=AnalysisDeleteResponse)
def clear_analysis_history():
    return AnalysisDeleteResponse(**analysis_service.clear_analysis_history())


@router.post("/run", response_model=AnalysisRunResponse)
def run_analysis(req: AnalysisRunRequest):
    try:
        job = analysis_service.create_analysis_job(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalysisRunResponse(
        ok=True,
        job_id=job["job_id"],
        status=job["status"],
        result=None,
        error=None,
    )


@router.get("/{analysis_id}", response_model=AnalysisResult)
def get_analysis_result(analysis_id: str):
    result = analysis_service.get_analysis_result(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisResult(**result)


@router.delete("/{analysis_id}", response_model=AnalysisDeleteResponse)
def delete_analysis_result(analysis_id: str):
    result = analysis_service.delete_analysis_result(analysis_id)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisDeleteResponse(**result)
