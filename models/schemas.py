"""
Domain models for IRP Label Proofreading
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


class BoundingBox(BaseModel):
    top: float
    left: float
    width: float
    height: float


class ChangeResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    reason: str
    outcome: Literal["pass", "fail", "warn"]
    confidence: float = Field(ge=0.0, le=1.0)
    redline_bbox: Optional[BoundingBox] = None
    final_bbox: Optional[BoundingBox] = None
    ai_explanation: Optional[str] = None
    group: str = "General"
    # Which page of the redline this change came from (1-based)
    redline_page: int = 1


class LabelResult(BaseModel):
    label_id: str
    filename: str
    # PLM label title extracted from PDF e.g. "LCN-299967042_1"
    plm_title: Optional[str] = None
    total_changes: int
    passed: int
    failed: int
    warnings: int
    changes: list[ChangeResult]


class AnalysisStatus(BaseModel):
    job_id: str
    status: Literal["queued", "extracting", "ocr", "ai_verify", "locating", "report", "done", "error"]
    progress: int = Field(ge=0, le=100)
    message: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[list[LabelResult]] = None
    error: Optional[str] = None


class AnalysisRequest(BaseModel):
    redline_file_id: str
    lrf_file_id: Optional[str] = None          # NEW — LRF is optional but enriches AI
    final_label_file_ids: list[str]             # bulk — as many as uploaded


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    page_count: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class ReportRequest(BaseModel):
    job_id: str
    format: Literal["json", "pdf"] = "json"
