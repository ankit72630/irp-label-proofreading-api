"""Report router — full JSON result + summary."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import AnalysisStatus
from services import analysis_service

router = APIRouter()


@router.get("/{job_id}")
async def get_report(job_id: str):
    job: AnalysisStatus = analysis_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(400, f"Job not complete — status: {job.status}")
    return JSONResponse(content=job.model_dump(mode="json"))


@router.get("/{job_id}/summary")
async def get_summary(job_id: str):
    job = analysis_service.get_job(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "Job not found or incomplete")

    total = passed = failed = warnings = 0
    for label in (job.result or []):
        total += label.total_changes
        passed += label.passed
        failed += label.failed
        warnings += label.warnings

    return {
        "job_id": job_id,
        "labels_checked": len(job.result or []),
        "total_changes": total,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "compliance_rate": round(passed / total * 100, 1) if total else 0,
        "completed_at": job.completed_at,
    }
