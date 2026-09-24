"""
NEXUS Phase 8 — Streaming Engine Tests
Tests EventBus pub/sub, StreamProcessor sliding-window detection, and simulator replay.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np

from services.streaming.event_bus import EventBus, get_event_bus
from services.streaming.stream_processor import StreamProcessor, _compute_severity
from services.streaming.simulator import DataStreamSimulator


# ─────────────────────────────────────────────────────────────────────────────
# EventBus tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = EventBus()
    async with await bus.subscribe("test.topic") as sub:
        await bus.publish("test.topic", {"value": 42})
        record = await sub.get(timeout=1.0)
    assert record == {"value": 42}


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    bus = EventBus()
    async with await bus.subscribe("topic.a") as s1, await bus.subscribe("topic.a") as s2:
        await bus.publish("topic.a", "hello")
        r1 = await s1.get(timeout=1.0)
        r2 = await s2.get(timeout=1.0)
    assert r1 == "hello"
    assert r2 == "hello"


@pytest.mark.asyncio
async def test_event_bus_no_cross_topic_leakage():
    bus = EventBus()
    async with await bus.subscribe("topic.x") as sub_x:
        await bus.publish("topic.y", "should_not_arrive")
        record = await sub_x.get(timeout=0.2)  # Should timeout
    assert record is None


@pytest.mark.asyncio
async def test_event_bus_subscriber_count():
    bus = EventBus()
    assert bus.topic_subscriber_count("empty.topic") == 0
    async with await bus.subscribe("count.test") as _:
        assert bus.topic_subscriber_count("count.test") == 1
    # After context exit the subscription should be removed
    await asyncio.sleep(0.05)
    assert bus.topic_subscriber_count("count.test") == 0


# ─────────────────────────────────────────────────────────────────────────────
# StreamProcessor tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_processor_no_anomaly_normal_values():
    """Normal values within 2σ should not trigger anomalies."""
    bus = EventBus()
    proc = StreamProcessor(
        session_id="test-sess-1",
        bus=bus,
        numeric_columns=["val"],
        window_size=30,
        z_threshold=3.0,
    )
    # Fill window with normal values first
    for v in np.random.normal(100, 5, 50):
        record = {"val": float(v), "timestamp": "2024-01-01T00:00:00"}
        await proc.process_record(record)

    # Now send a normal value
    result = await proc.process_record({"val": 102.0, "timestamp": "2024-01-01T00:01:00"})
    assert len(result) == 0


@pytest.mark.asyncio
async def test_stream_processor_detects_spike():
    """A massive spike far beyond 3σ should be detected as an anomaly."""
    bus = EventBus()
    proc = StreamProcessor(
        session_id="test-sess-2",
        bus=bus,
        numeric_columns=["sensor"],
        window_size=30,
        z_threshold=2.5,
    )
    # Fill window with stable values
    for v in np.linspace(50, 55, 40):
        await proc.process_record({"sensor": float(v), "timestamp": "t0"})

    # Inject a spike 15σ above mean
    spike_val = 50 + 15 * 2.0   # well above threshold
    result = await proc.process_record({"sensor": spike_val, "timestamp": "t1"})
    assert len(result) == 1
    assert result[0].column == "sensor"
    assert abs(result[0].z_score) >= 2.5
    assert result[0].severity in ("high", "critical")


@pytest.mark.asyncio
async def test_stream_processor_publishes_to_bus():
    """Detected anomalies should appear on the event bus alert topic."""
    bus = EventBus()
    session_id = "test-sess-3"
    proc = StreamProcessor(
        session_id=session_id,
        bus=bus,
        numeric_columns=["metric"],
        window_size=20,
        z_threshold=2.0,
    )
    alert_topic = f"{StreamProcessor.ALERTS_TOPIC}.{session_id}"

    # Fill window
    for v in [10.0] * 25:
        await proc.process_record({"metric": v, "timestamp": "t"})

    async with await bus.subscribe(alert_topic) as sub:
        # Inject spike
        await proc.process_record({"metric": 500.0, "timestamp": "t_spike"})
        alert_dict = await sub.get(timeout=1.0)

    assert alert_dict is not None
    assert alert_dict["column"] == "metric"
    assert alert_dict["session_id"] == session_id


def test_compute_severity_levels():
    assert _compute_severity(1.5) == "low"
    assert _compute_severity(2.6) == "medium"
    assert _compute_severity(3.1) == "high"
    assert _compute_severity(4.5) == "critical"
    assert _compute_severity(-4.5) == "critical"


@pytest.mark.asyncio
async def test_stream_processor_stats():
    bus = EventBus()
    proc = StreamProcessor(
        session_id="stats-test",
        bus=bus,
        numeric_columns=["x"],
        window_size=20,
        z_threshold=2.5,
    )
    for v in [1.0] * 10:
        await proc.process_record({"x": v, "timestamp": "t"})

    stats = proc.get_stats()
    assert stats["records_processed"] == 10
    assert stats["columns_monitored"] == ["x"]
    assert "window_fills" in stats


# ─────────────────────────────────────────────────────────────────────────────
# DataStreamSimulator tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simulator_emits_records():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    sim = DataStreamSimulator(df, numeric_columns=["a", "b"], rows_per_second=500.0, loop=False)

    collected = []
    await sim.stream(on_record=collected.append, max_records=5)

    assert len(collected) == 5
    assert all("a" in r and "b" in r for r in collected)
    assert all("timestamp" in r for r in collected)


@pytest.mark.asyncio
async def test_simulator_stop():
    df = pd.DataFrame({"x": list(range(1000))})
    sim = DataStreamSimulator(df, numeric_columns=["x"], rows_per_second=1000.0, loop=True)

    collected = []

    async def run():
        await sim.stream(on_record=collected.append)

    task = asyncio.ensure_future(run())
    await asyncio.sleep(0.05)
    sim.stop()
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert sim.records_emitted >= 1
    assert not sim._running


@pytest.mark.asyncio
async def test_simulator_spike_injection():
    """With 100% spike probability, all records should be spiked."""
    df = pd.DataFrame({"v": [0.0] * 20})
    sim = DataStreamSimulator(
        df, numeric_columns=["v"],
        rows_per_second=500.0,
        spike_probability=1.0,
        spike_factor=5.0,
        loop=False
    )

    collected = []
    await sim.stream(on_record=collected.append, max_records=20)

    assert sim.spikes_injected == 20
    # All values should be non-zero due to spikes
    assert all(r["v"] != 0.0 for r in collected)
