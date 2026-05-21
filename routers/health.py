from fastapi import APIRouter
from services.ai_service import AIService
import sys

router = APIRouter()
_ai_service: AIService = None


def set_ai_service(svc: AIService):
    global _ai_service
    _ai_service = svc


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "python": sys.version,
        "ai_backend": _ai_service.backend_name if _ai_service else "unknown",
    }
