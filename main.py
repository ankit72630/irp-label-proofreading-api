"""
IRP Label Proofreading — FastAPI Backend
AI: OpenAI gpt-4o-mini (text) + gpt-4o (vision)
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import upload, analysis, report, health
from routers import analysis as analysis_router
from routers import health as health_router
from services.ai_service import AIService

ai_service = AIService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ai_service.initialize()
    # Inject ai_service into routers that need it
    analysis_router.set_ai_service(ai_service)
    health_router.set_ai_service(ai_service)
    yield
    await ai_service.shutdown()


app = FastAPI(
    title="IRP Label Proofreading API",
    description="AI-powered pharmaceutical label change verification — OpenAI powered",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(health.router,    prefix="/api",          tags=["Health"])
app.include_router(upload.router,    prefix="/api/upload",   tags=["Upload"])
app.include_router(analysis.router,  prefix="/api/analysis", tags=["Analysis"])
app.include_router(report.router,    prefix="/api/report",   tags=["Report"])
