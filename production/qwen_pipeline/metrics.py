from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class PipelineMetrics:
    """Track pipeline performance metrics."""

    total_queries: int = 0
    total_duration_sec: float = 0.0
    total_errors: int = 0
    total_rejections: int = 0

    min_duration_sec: float = float("inf")
    max_duration_sec: float = 0.0

    queries_by_type: dict[str, int] = field(default_factory=dict)
    error_types: dict[str, int] = field(default_factory=dict)

    _start_time: float | None = field(default=None, init=False, repr=False)

    @property
    def avg_duration_sec(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_duration_sec / self.total_queries

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        successful = self.total_queries - self.total_errors
        return (successful / self.total_queries) * 100.0

    def record_query_start(self) -> None:
        self._start_time = time.perf_counter()

    def record_query_end(
        self,
        *,
        success: bool = True,
        error_type: str | None = None,
        rejected: bool = False,
    ) -> None:
        if self._start_time is None:
            # If start wasn't called, avoid crashing; still count the query
            self.total_queries += 1
            if not success:
                self.total_errors += 1
                if error_type:
                    self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
            if rejected:
                self.total_rejections += 1
            return

        duration = time.perf_counter() - self._start_time
        self.total_queries += 1
        self.total_duration_sec += duration

        self.min_duration_sec = min(self.min_duration_sec, duration)
        self.max_duration_sec = max(self.max_duration_sec, duration)

        if not success:
            self.total_errors += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        if rejected:
            self.total_rejections += 1

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.total_queries - self.total_errors,
            "failed_queries": self.total_errors,
            "rejected_queries": self.total_rejections,
            "success_rate_percent": round(self.success_rate, 2),
            "avg_duration_sec": round(self.avg_duration_sec, 2),
            "min_duration_sec": (
                0.0 if self.min_duration_sec == float("inf") else round(self.min_duration_sec, 2)
            ),
            "max_duration_sec": round(self.max_duration_sec, 2),
            "total_duration_sec": round(self.total_duration_sec, 2),
            "error_types": self.error_types,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def reset(self) -> None:
        # Reset to defaults without calling __init__() directly
        self.total_queries = 0
        self.total_duration_sec = 0.0
        self.total_errors = 0
        self.total_rejections = 0
        self.min_duration_sec = float("inf")
        self.max_duration_sec = 0.0
        self.queries_by_type.clear()
        self.error_types.clear()
        self._start_time = None


@lru_cache(maxsize=1)
def get_metrics() -> PipelineMetrics:
    return PipelineMetrics()
