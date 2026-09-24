from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel


class SystemHealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime
    services: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
