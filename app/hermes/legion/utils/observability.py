"""
Observability Utilities for Legion System.

This module provides structured logging, metrics collection, and tracing
for the Legion multi-agent orchestration system.
"""

import functools
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Span:
    """A tracing span for operation tracking."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_error(self, error: str) -> None:
        """Mark span as errored."""
        self.status = "error"
        self.error = error

    def finish(self) -> None:
        """Mark span as finished."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "events": self.events,
            "status": self.status,
            "error": self.error,
        }


class MetricsCollector:
    """
    Collects and stores metrics for the Legion system.

    Provides counters, gauges, histograms, and timers.
    """

    def __init__(self):
        self._metrics: List[Metric] = []
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

    def increment(
        self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value
        self._record(name, self._counters[key], MetricType.COUNTER, tags)

    def gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, tags)
        self._gauges[key] = value
        self._record(name, value, MetricType.GAUGE, tags)

    def histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        self._record(name, value, MetricType.HISTOGRAM, tags)

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000  # Convert to ms
            self._record(name, duration, MetricType.TIMER, tags)
            self.histogram(f"{name}_duration_ms", duration, tags)

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _record(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]],
    ) -> None:
        """Record a metric."""
        self._metrics.append(
            Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=tags or {},
            )
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        histogram_stats = {}
        for key, values in self._histograms.items():
            if values:
                sorted_values = sorted(values)
                histogram_stats[key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted_values[len(values) // 2],
                    "p95": (
                        sorted_values[int(len(values) * 0.95)]
                        if len(values) > 1
                        else sorted_values[0]
                    ),
                    "p99": (
                        sorted_values[int(len(values) * 0.99)]
                        if len(values) > 1
                        else sorted_values[0]
                    ),
                }

        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": histogram_stats,
            "total_metrics_recorded": len(self._metrics),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class StructuredLogger:
    """
    Structured logging for Legion operations.

    Outputs JSON-formatted log entries with consistent structure.
    """

    def __init__(self, name: str = "legion"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log entries."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()

    def _format_entry(
        self,
        level: str,
        message: str,
        event_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format a structured log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            "event_type": event_type,
            **self._context,
            **kwargs,
        }
        return entry

    def info(self, message: str, event_type: str = "info", **kwargs) -> None:
        """Log an info-level structured entry."""
        entry = self._format_entry("INFO", message, event_type, **kwargs)
        self.logger.info(json.dumps(entry))

    def warning(self, message: str, event_type: str = "warning", **kwargs) -> None:
        """Log a warning-level structured entry."""
        entry = self._format_entry("WARNING", message, event_type, **kwargs)
        self.logger.warning(json.dumps(entry))

    def error(
        self,
        message: str,
        event_type: str = "error",
        error: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an error-level structured entry."""
        if error:
            kwargs["error"] = error
        entry = self._format_entry("ERROR", message, event_type, **kwargs)
        self.logger.error(json.dumps(entry))

    def debug(self, message: str, event_type: str = "debug", **kwargs) -> None:
        """Log a debug-level structured entry."""
        entry = self._format_entry("DEBUG", message, event_type, **kwargs)
        self.logger.debug(json.dumps(entry))


class LegionObservability:
    """
    Central observability hub for Legion operations.

    Combines metrics, logging, and tracing into a unified interface.
    """

    def __init__(self, service_name: str = "legion"):
        self.service_name = service_name
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger(service_name)
        self._active_spans: Dict[str, Span] = {}
        self._trace_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique ID for traces/spans."""
        import uuid

        return str(uuid.uuid4())[:16]

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Span:
        """
        Start a new tracing span.

        Args:
            name: Name of the span
            parent_span_id: Optional parent span ID. If provided, inherits trace_id.
            tags: Optional tags to add to the span

        Returns:
            New Span object
        """
        span_id = self._generate_id()

        # Inherit trace_id from parent if exists, otherwise generate new
        if parent_span_id and parent_span_id in self._active_spans:
            trace_id = self._active_spans[parent_span_id].trace_id
        else:
            trace_id = self._generate_id()

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tags=tags or {},
        )

        self._active_spans[span_id] = span
        self.logger.debug(
            f"Span started: {name}",
            event_type="span_start",
            trace_id=trace_id,
            span_id=span_id,
        )

        return span

    def end_span(self, span: Span) -> None:
        """End a tracing span."""
        span.finish()

        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]

        # Record span duration as metric
        self.metrics.histogram(
            "span_duration_ms",
            span.duration_ms,
            tags={"span_name": span.name, "status": span.status},
        )

        self.logger.debug(
            f"Span ended: {span.name}",
            event_type="span_end",
            trace_id=span.trace_id,
            span_id=span.span_id,
            duration_ms=span.duration_ms,
            status=span.status,
        )

    @contextmanager
    def span(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for creating spans."""
        span = self.start_span(name, tags=tags)
        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            self.end_span(span)

    # Convenience methods for common Legion operations

    def log_orchestration_start(
        self,
        user_id: str,
        query: str,
        strategy: Optional[str] = None,
    ) -> None:
        """Log the start of an orchestration."""
        self.metrics.increment(
            "orchestration_started", tags={"strategy": strategy or "unknown"}
        )
        self.logger.info(
            "Orchestration started",
            event_type="orchestration_start",
            user_id=user_id[:8] + "...",  # Truncate for privacy
            query_length=len(query),
            strategy=strategy,
        )

    def log_orchestration_complete(
        self,
        user_id: str,
        duration_ms: float,
        worker_count: int,
        success: bool,
    ) -> None:
        """Log the completion of an orchestration."""
        status = "success" if success else "failure"
        self.metrics.increment("orchestration_completed", tags={"status": status})
        self.metrics.histogram("orchestration_duration_ms", duration_ms)
        self.metrics.histogram("orchestration_worker_count", worker_count)

        self.logger.info(
            "Orchestration completed",
            event_type="orchestration_complete",
            user_id=user_id[:8] + "...",
            duration_ms=duration_ms,
            worker_count=worker_count,
            success=success,
        )

    def log_worker_execution(
        self,
        worker_id: str,
        role: str,
        duration_ms: float,
        status: str,
    ) -> None:
        """Log worker execution."""
        self.metrics.increment("worker_executed", tags={"role": role, "status": status})
        self.metrics.histogram("worker_duration_ms", duration_ms, tags={"role": role})

        self.logger.info(
            f"Worker executed: {worker_id}",
            event_type="worker_execution",
            worker_id=worker_id,
            role=role,
            duration_ms=duration_ms,
            status=status,
        )

    def log_routing_decision(
        self,
        action: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Log a routing decision."""
        self.metrics.increment("routing_decisions", tags={"action": action})
        self.metrics.histogram("routing_confidence", confidence)

        self.logger.info(
            f"Routing decision: {action}",
            event_type="routing_decision",
            action=action,
            confidence=confidence,
            reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get observability summary."""
        return {
            "service": self.service_name,
            "metrics": self.metrics.get_summary(),
            "active_spans": len(self._active_spans),
        }


# Global singleton instance
_observability: Optional[LegionObservability] = None


def get_observability() -> LegionObservability:
    """Get the global observability instance."""
    global _observability
    if _observability is None:
        _observability = LegionObservability()
    return _observability


def trace(name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to trace function execution.

    Args:
        name: Name for the span
        tags: Optional tags to add to the span

    Example:
        @trace("process_request")
        async def process_request(user_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            obs = get_observability()
            with obs.span(name, tags=tags) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_error(str(e))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            obs = get_observability()
            with obs.span(name, tags=tags) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_error(str(e))
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
