"""
ResearchMind FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.routes import router
from backend.core.config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ResearchMind starting up", provider=settings.llm_provider, model=settings.llm_model)
    yield
    logger.info("ResearchMind shutting down")


app = FastAPI(
    title="ResearchMind API",
    description="Autonomous Multi-Agent Research System with Eval Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "ResearchMind",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }
