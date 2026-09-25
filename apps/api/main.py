from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from apps.api.core.config import settings
from apps.api.core.database import init_db
from apps.api.core.exceptions import NexusException
from apps.api.routes import (
    auth_router,
    health_router,
    projects_router,
    datasets_router,
    jobs_router,
    agents_router,
    ml_router,
    forecast_router,
    anomalies_router,
    rag_router,
    graph_router,
    streaming_router,
    mlops_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database Tables
    init_db()
    yield
    # Shutdown logic if needed


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="Autonomous Multi-Agent Data Intelligence & Prediction Platform API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler for NexusException
@app.exception_handler(NexusException)
async def nexus_exception_handler(request: Request, exc: NexusException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


# Register Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(projects_router, prefix=settings.API_V1_STR)
app.include_router(datasets_router, prefix=settings.API_V1_STR)
app.include_router(jobs_router, prefix=settings.API_V1_STR)
app.include_router(agents_router, prefix=settings.API_V1_STR)
app.include_router(ml_router, prefix=settings.API_V1_STR)
app.include_router(forecast_router, prefix=settings.API_V1_STR)
app.include_router(anomalies_router, prefix=settings.API_V1_STR)
app.include_router(rag_router, prefix=settings.API_V1_STR)
app.include_router(graph_router, prefix=settings.API_V1_STR)
app.include_router(streaming_router, prefix=settings.API_V1_STR)
app.include_router(mlops_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    return {
        "platform": "NEXUS Autonomous Multi-Agent Data Intelligence Platform",
        "version": "0.1.0",
        "status": "online",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR
    }
