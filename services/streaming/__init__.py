"""
NEXUS Streaming Engine — Phase 8
Real-time data stream processing, event bus pub/sub, and live anomaly detection.
"""

from .event_bus import EventBus
from .stream_processor import StreamProcessor, StreamingAnomaly
from .simulator import DataStreamSimulator

__all__ = [
    "EventBus",
    "StreamProcessor",
    "StreamingAnomaly",
    "DataStreamSimulator",
]
