"""
NEXUS Streaming API Routes — Phase 8
Endpoints for launching, monitoring, and receiving real-time streaming anomaly detection sessions.
Includes a WebSocket endpoint that pushes live alert events to connected UI clients.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from apps.api.core.database import get_db, SessionLocal
from apps.api.core.exceptions import EntityNotFoundException
from apps.api.routes.auth import get_current_user
from apps.api.models.user import User
from apps.api.models.dataset import Dataset
from apps.api.models.streaming_alert import StreamingAlert
from apps.api.schemas.streaming import (
    StreamStartRequest,
    StreamSessionResponse,
    StreamingAlertResponse,
    StreamAlertsResponse,
)
from services.streaming.event_bus import get_event_bus
from services.streaming.stream_processor import StreamProcessor
from services.streaming.simulator import DataStreamSimulator
from services.ingestion.factory import IngestionFactory

router = APIRouter(prefix="/stream", tags=["Live Streaming Engine"])

# ──────────────────────────────────────────────────────────────────────────────
# In-memory session registry (session_id -> session metadata dict)
# In production this would live in Redis.
# ──────────────────────────────────────────────────────────────────────────────
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _numeric_columns(df: pd.DataFrame, requested: Optional[List[str]] = None) -> List[str]:
    """Return numeric column names, optionally filtered to requested list."""
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    if not requested:
        return all_numeric[:8]   # Cap to 8 to avoid overwhelming streams
    return [c for c in requested if c in all_numeric]


# ──────────────────────────────────────────────────────────────────────────────
# Background task: simulate + process
# ──────────────────────────────────────────────────────────────────────────────

async def _run_streaming_session(
    session_id: str,
    dataset_id: str,
    df: pd.DataFrame,
    numeric_cols: List[str],
    window_size: int,
    z_threshold: float,
    rows_per_second: float,
    spike_probability: float,
    spike_factor: float,
) -> None:
    """Coroutine that runs the simulator + processor for one session."""
    bus = get_event_bus()
    processor = StreamProcessor(
        session_id=session_id,
        bus=bus,
        numeric_columns=numeric_cols,
        window_size=window_size,
        z_threshold=z_threshold,
    )

    simulator = DataStreamSimulator(
        df=df,
        numeric_columns=numeric_cols,
        rows_per_second=rows_per_second,
        spike_probability=spike_probability,
        spike_factor=spike_factor,
        loop=True,
    )

    # Store references in session registry
    meta = _SESSIONS.get(session_id, {})
    meta["processor"] = processor
    meta["simulator"] = simulator
    meta["status"] = "running"
    _SESSIONS[session_id] = meta

    # Persist anomalies to DB as they occur
    async def persist_alert(record: Dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            alert = StreamingAlert(
                id=str(uuid.uuid4()),
                session_id=session_id,
                dataset_id=dataset_id,
                anomaly_id=record["anomaly_id"],
                timestamp=record["timestamp"],
                column=record["column"],
                value=record["value"],
                z_score=record["z_score"],
                iqr_score=record.get("iqr_score"),
                severity=record["severity"],
                anomaly_type=record["anomaly_type"],
                window_mean=record.get("window_mean"),
                window_std=record.get("window_std"),
                window_size=record.get("window_size"),
            )
            db.add(alert)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    # Subscribe to our own alert topic to persist
    alert_topic = f"{StreamProcessor.ALERTS_TOPIC}.{session_id}"
    async with await bus.subscribe(alert_topic) as sub:
        # Start simulator in a concurrent task
        async def run_sim():
            await simulator.stream(
                on_record=lambda rec: asyncio.ensure_future(processor.process_record(rec))
            )

        sim_task = asyncio.ensure_future(run_sim())

        # Persist all alerts until session is stopped
        while meta.get("status") == "running":
            alert_dict = await sub.get(timeout=0.5)
            if alert_dict:
                await persist_alert(alert_dict)

        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass

    meta["status"] = "stopped"


# ──────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StreamSessionResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_streaming_session(
    payload: StreamStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Starts a real-time streaming simulation session for the given dataset.
    Records are replayed at rows_per_second rate with synthetic spike injection.
    Detected anomalies are published via WebSocket and persisted to the database.
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise EntityNotFoundException("Dataset", payload.dataset_id)

    adapter = IngestionFactory.get_adapter(dataset.file_type)
    df = adapter.load(dataset.file_path)

    numeric_cols = _numeric_columns(df, payload.target_columns)
    if not numeric_cols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No numeric columns found in dataset for streaming."
        )

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "session_id": session_id,
        "dataset_id": payload.dataset_id,
        "status": "queued",
        "columns": numeric_cols,
        "rows_per_second": payload.rows_per_second,
        "window_size": payload.window_size,
        "z_threshold": payload.z_threshold,
        "processor": None,
        "simulator": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Launch background coroutine via asyncio (not thread pool)
    async def _launch():
        await _run_streaming_session(
            session_id=session_id,
            dataset_id=payload.dataset_id,
            df=df,
            numeric_cols=numeric_cols,
            window_size=payload.window_size,
            z_threshold=payload.z_threshold,
            rows_per_second=payload.rows_per_second,
            spike_probability=payload.spike_probability,
            spike_factor=payload.spike_factor,
        )

    asyncio.ensure_future(_launch())

    return StreamSessionResponse(
        session_id=session_id,
        dataset_id=payload.dataset_id,
        status="queued",
        rows_per_second=payload.rows_per_second,
        window_size=payload.window_size,
        z_threshold=payload.z_threshold,
        columns_monitored=numeric_cols,
        records_processed=0,
        anomalies_detected=0,
        anomaly_rate=0.0,
        spikes_injected=0,
        message="Streaming session queued. Connect to WebSocket for live alerts.",
    )


@router.get("/sessions", response_model=List[StreamSessionResponse])
def list_sessions(current_user: User = Depends(get_current_user)):
    """Lists all active and recent streaming sessions."""
    results = []
    for session_id, meta in _SESSIONS.items():
        proc: Optional[StreamProcessor] = meta.get("processor")
        sim: Optional[DataStreamSimulator] = meta.get("simulator")
        results.append(StreamSessionResponse(
            session_id=session_id,
            dataset_id=meta.get("dataset_id", ""),
            status=meta.get("status", "unknown"),
            rows_per_second=meta.get("rows_per_second", 0),
            window_size=meta.get("window_size", 50),
            z_threshold=meta.get("z_threshold", 2.8),
            columns_monitored=meta.get("columns", []),
            records_processed=proc.records_processed if proc else 0,
            anomalies_detected=proc.anomalies_detected if proc else 0,
            anomaly_rate=proc.get_stats()["anomaly_rate"] if proc else 0.0,
            spikes_injected=sim.spikes_injected if sim else 0,
        ))
    return results


@router.get("/sessions/{session_id}", response_model=StreamSessionResponse)
def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Returns current status and stats for a streaming session."""
    meta = _SESSIONS.get(session_id)
    if not meta:
        raise EntityNotFoundException("StreamSession", session_id)

    proc: Optional[StreamProcessor] = meta.get("processor")
    sim: Optional[DataStreamSimulator] = meta.get("simulator")

    return StreamSessionResponse(
        session_id=session_id,
        dataset_id=meta.get("dataset_id", ""),
        status=meta.get("status", "unknown"),
        rows_per_second=meta.get("rows_per_second", 0),
        window_size=meta.get("window_size", 50),
        z_threshold=meta.get("z_threshold", 2.8),
        columns_monitored=meta.get("columns", []),
        records_processed=proc.records_processed if proc else 0,
        anomalies_detected=proc.anomalies_detected if proc else 0,
        anomaly_rate=proc.get_stats()["anomaly_rate"] if proc else 0.0,
        spikes_injected=sim.spikes_injected if sim else 0,
    )


@router.post("/sessions/{session_id}/stop")
def stop_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Gracefully stops a running streaming session."""
    meta = _SESSIONS.get(session_id)
    if not meta:
        raise EntityNotFoundException("StreamSession", session_id)

    sim: Optional[DataStreamSimulator] = meta.get("simulator")
    if sim:
        sim.stop()
    meta["status"] = "stopped"

    return {"session_id": session_id, "status": "stopped"}


@router.get("/sessions/{session_id}/alerts", response_model=StreamAlertsResponse)
def get_session_alerts(
    session_id: str,
    limit: int = 100,
    severity: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns paginated list of persisted anomaly alerts for a session.
    Optionally filter by severity: low | medium | high | critical.
    """
    query = db.query(StreamingAlert).filter(StreamingAlert.session_id == session_id)
    if severity:
        query = query.filter(StreamingAlert.severity == severity)
    rows = query.order_by(StreamingAlert.created_at.desc()).limit(limit).all()

    severity_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    all_rows = db.query(StreamingAlert).filter(StreamingAlert.session_id == session_id).all()
    for r in all_rows:
        if r.severity in severity_counts:
            severity_counts[r.severity] += 1

    alerts = [
        StreamingAlertResponse(
            anomaly_id=r.anomaly_id,
            session_id=r.session_id,
            timestamp=r.timestamp,
            column=r.column,
            value=r.value,
            z_score=r.z_score,
            iqr_score=r.iqr_score,
            severity=r.severity,
            anomaly_type=r.anomaly_type,
            window_mean=r.window_mean,
            window_std=r.window_std,
            window_size=r.window_size,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]

    return StreamAlertsResponse(
        session_id=session_id,
        total=len(all_rows),
        alerts=alerts,
        severity_counts=severity_counts,
    )


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket: Live Alert Push
# ──────────────────────────────────────────────────────────────────────────────

@router.websocket("/sessions/{session_id}/live")
async def websocket_live_alerts(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint that pushes real-time anomaly alert events to the client.
    Clients receive JSON-encoded StreamingAnomaly dicts as they are detected.
    Also sends periodic heartbeat pings every 5 seconds so clients can detect
    dropped connections.
    """
    await websocket.accept()

    bus = get_event_bus()
    alert_topic = f"{StreamProcessor.ALERTS_TOPIC}.{session_id}"

    try:
        async with await bus.subscribe(alert_topic) as sub:
            last_ping = asyncio.get_event_loop().time()
            while True:
                alert = await sub.get(timeout=0.2)
                if alert:
                    await websocket.send_json({"type": "alert", "data": alert})

                # Send heartbeat every 5 seconds
                now = asyncio.get_event_loop().time()
                if now - last_ping >= 5.0:
                    meta = _SESSIONS.get(session_id, {})
                    proc: Optional[StreamProcessor] = meta.get("processor")
                    await websocket.send_json({
                        "type": "heartbeat",
                        "session_id": session_id,
                        "status": meta.get("status", "unknown"),
                        "records_processed": proc.records_processed if proc else 0,
                        "anomalies_detected": proc.anomalies_detected if proc else 0,
                    })
                    last_ping = now

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
