"""Analysis router — start a job and stream status via SSE."""

import asyncio
import json
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import AnalysisRequest, AnalysisStatus
from services import analysis_service
from services.ai_service import AIService

router = APIRouter()
_ai_service: AIService = None


def set_ai_service(svc: AIService):
    global _ai_service
    _ai_service = svc


@router.post("/start", response_model=AnalysisStatus)
async def start_analysis(req: AnalysisRequest, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background.add_task(
        analysis_service.run_analysis,
        job_id=job_id,
        redline_file_id=req.redline_file_id,
        lrf_file_id=req.lrf_file_id,           # NEW — optional LRF
        final_label_file_ids=req.final_label_file_ids,
        ai_service=_ai_service,
    )
    return AnalysisStatus(job_id=job_id, status="queued", progress=0, message="Job queued")


@router.get("/status/{job_id}", response_model=AnalysisStatus)
async def get_status(job_id: str):
    job = analysis_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/stream/{job_id}")
async def stream_status(job_id: str):
    """Server-Sent Events — real-time progress updates."""
    async def event_generator():
        while True:
            job = analysis_service.get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            payload = job.model_dump(exclude={"result"}, mode="json")
            for k, v in payload.items():
                if hasattr(v, "isoformat"):
                    payload[k] = v.isoformat()
            yield f"data: {json.dumps(payload)}\n\n"
            if job.status in ("done", "error"):
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
