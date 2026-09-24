"""
NEXUS Pipeline Scheduler — Phase 9
Manages cron-based and interval-based ML re-training pipeline definitions.
Uses APScheduler if available, falls back to simple in-memory schedule tracking.
"""

import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PipelineDefinition:
    """A scheduled re-training pipeline definition."""
    pipeline_id: str
    name: str
    description: Optional[str]
    project_id: str
    dataset_id: str
    model_type: str          # 'automl' | 'forecast'
    cron_expression: str     # e.g. '0 2 * * *' (2 AM daily)
    target_column: str
    is_active: bool
    last_run_at: Optional[str]
    last_run_status: Optional[str]  # 'success' | 'failed' | None
    next_run_at: Optional[str]
    run_count: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineScheduler:
    """
    Lightweight pipeline scheduler that stores pipeline definitions and tracks
    execution history. Delegates actual scheduling to APScheduler when available.

    Architecture note: The `register()` / `trigger()` interface is designed to be
    Airflow-compatible: replacing the APScheduler backend with an Airflow DAG trigger
    only requires changing the `_execute_pipeline()` method body.
    """

    def __init__(self) -> None:
        self._pipelines: Dict[str, PipelineDefinition] = {}
        self._scheduler = None
        self._try_init_apscheduler()

    def _try_init_apscheduler(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self._scheduler = BackgroundScheduler(daemon=True)
            self._scheduler.start()
        except ImportError:
            self._scheduler = None

    def create(
        self,
        name: str,
        project_id: str,
        dataset_id: str,
        model_type: str,
        target_column: str,
        cron_expression: str = "0 2 * * *",
        description: Optional[str] = None,
        on_trigger: Optional[Callable[[PipelineDefinition], None]] = None,
    ) -> PipelineDefinition:
        """Create and optionally schedule a pipeline."""
        pipeline_id = str(uuid.uuid4())
        pipeline = PipelineDefinition(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            project_id=project_id,
            dataset_id=dataset_id,
            model_type=model_type,
            cron_expression=cron_expression,
            target_column=target_column,
            is_active=True,
            last_run_at=None,
            last_run_status=None,
            next_run_at=None,
            run_count=0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._pipelines[pipeline_id] = pipeline

        if self._scheduler and on_trigger:
            self._schedule_with_apscheduler(pipeline, on_trigger)

        return pipeline

    def _schedule_with_apscheduler(
        self,
        pipeline: PipelineDefinition,
        on_trigger: Callable[[PipelineDefinition], None],
    ) -> None:
        try:
            from apscheduler.triggers.cron import CronTrigger
            parts = pipeline.cron_expression.split()
            if len(parts) == 5:
                minute, hour, day, month, day_of_week = parts
                trigger = CronTrigger(
                    minute=minute, hour=hour, day=day,
                    month=month, day_of_week=day_of_week
                )
                self._scheduler.add_job(
                    func=lambda: self._execute_pipeline(pipeline, on_trigger),
                    trigger=trigger,
                    id=pipeline.pipeline_id,
                    name=pipeline.name,
                    replace_existing=True,
                )
        except Exception:
            pass  # APScheduler config failed — run in manual-trigger mode

    def _execute_pipeline(
        self,
        pipeline: PipelineDefinition,
        on_trigger: Callable[[PipelineDefinition], None],
    ) -> None:
        pipeline.run_count += 1
        pipeline.last_run_at = datetime.now(timezone.utc).isoformat()
        try:
            on_trigger(pipeline)
            pipeline.last_run_status = "success"
        except Exception as e:
            pipeline.last_run_status = f"failed: {e}"

    def trigger_now(
        self,
        pipeline_id: str,
        on_trigger: Optional[Callable[[PipelineDefinition], None]] = None,
    ) -> Optional[PipelineDefinition]:
        """Manually trigger a pipeline immediately."""
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return None
        if on_trigger:
            self._execute_pipeline(pipeline, on_trigger)
        else:
            pipeline.run_count += 1
            pipeline.last_run_at = datetime.now(timezone.utc).isoformat()
            pipeline.last_run_status = "manually_triggered"
        return pipeline

    def delete(self, pipeline_id: str) -> bool:
        pipeline = self._pipelines.pop(pipeline_id, None)
        if pipeline and self._scheduler:
            try:
                self._scheduler.remove_job(pipeline_id)
            except Exception:
                pass
        return pipeline is not None

    def get(self, pipeline_id: str) -> Optional[PipelineDefinition]:
        return self._pipelines.get(pipeline_id)

    def list_all(self) -> List[PipelineDefinition]:
        return list(self._pipelines.values())

    def set_active(self, pipeline_id: str, is_active: bool) -> Optional[PipelineDefinition]:
        p = self._pipelines.get(pipeline_id)
        if p:
            p.is_active = is_active
        return p


# Global singleton scheduler
_scheduler: Optional[PipelineScheduler] = None


def get_scheduler() -> PipelineScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PipelineScheduler()
    return _scheduler
