from .auth import router as auth_router
from .health import router as health_router
from .projects import router as projects_router
from .datasets import router as datasets_router
from .jobs import router as jobs_router
from .agents import router as agents_router
from .ml import router as ml_router
from .forecasting import router as forecast_router
from .anomalies import router as anomalies_router
from .rag import router as rag_router
from .graph import router as graph_router
from .streaming import router as streaming_router
from .mlops import router as mlops_router

__all__ = [
    "auth_router",
    "health_router",
    "projects_router",
    "datasets_router",
    "jobs_router",
    "agents_router",
    "ml_router",
    "forecast_router",
    "anomalies_router",
    "rag_router",
    "graph_router",
    "streaming_router",
    "mlops_router",
]

