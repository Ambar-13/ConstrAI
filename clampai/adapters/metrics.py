"""
clampai.adapters.metrics — Pluggable observability backends.

Provides ``PrometheusMetrics`` and ``OTelMetrics``, both satisfying the
``clampai.MetricsBackend`` protocol defined in ``clampai.formal``.

    from clampai.adapters.metrics import PrometheusMetrics
    kernel = SafetyKernel(budget=100.0, invariants=[...],
                          metrics=PrometheusMetrics())

    # Custom registry for tests or multi-process isolation:
    registry = prometheus_client.CollectorRegistry()
    kernel = SafetyKernel(..., metrics=PrometheusMetrics(registry=registry))

    # OpenTelemetry:
    meter = opentelemetry.metrics.get_meter("clampai")
    kernel = SafetyKernel(..., metrics=OTelMetrics(meter))

Metric names emitted (documented fully in BENCHMARKS.md):

    clampai_actions_total{status="approved|rejected", kernel_id="..."}
    clampai_kernel_latency_seconds{operation="evaluate|execute", ...}
    clampai_budget_utilization_ratio{kernel_id="..."}
    clampai_budget_remaining{kernel_id="..."}
    clampai_step_count{kernel_id="..."}
    clampai_invariant_violations_total{invariant="...", enforcement="..."}
    clampai_rollbacks_total{kernel_id="..."}

All metric emission failures are silently swallowed as required by the
MetricsBackend protocol. A broken registry or misconfigured exporter will
never affect safety-critical kernel logic.

Requires: pip install clampai[prometheus] or clampai[opentelemetry]
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PrometheusMetrics:
    """
    MetricsBackend implementation backed by ``prometheus_client``.

    Requires: pip install prometheus_client

    Counters and gauges are created lazily on first use.  Histograms use
    Prometheus-default buckets optimised for the 0.06 ms ClampAI baseline
    plus a fine-grained low-latency range.

    Thread-safety: prometheus_client metrics are thread-safe.

    Args:
        registry:
            Prometheus CollectorRegistry.  Defaults to the global
            ``REGISTRY`` (suitable for single-process servers).  Pass a
            custom registry for tests or multi-process deployments.
        namespace:
            Metric name prefix.  Default: ``"clampai"``.
    """

    _LATENCY_BUCKETS = (
        0.0001, 0.0002, 0.0005,          # < 1 ms (ClampAI normal range)
        0.001, 0.002, 0.005,             # 1–5 ms
        0.01, 0.025, 0.05, 0.1,         # 10–100 ms
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0, # > 100 ms (slow / degraded)
    )

    def __init__(self, registry=None, namespace: str = "clampai") -> None:
        try:
            import prometheus_client as prom
        except ImportError:
            raise ImportError(
                "prometheus_client is required for PrometheusMetrics. "
                "Install it with: pip install 'clampai[prometheus]'"
            )
        self._prom = prom
        self._registry = registry or prom.REGISTRY
        self._ns = namespace
        self._counters: Dict[str, object] = {}
        self._histograms: Dict[str, object] = {}
        self._gauges: Dict[str, object] = {}


    def increment(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        try:
            counter = self._get_counter(name, labels or {})
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
        except Exception:
            pass  # Never affect kernel logic

    def observe(self, name: str, value: float,
                labels: Optional[Dict[str, str]] = None) -> None:
        try:
            hist = self._get_histogram(name, labels or {})
            if labels:
                hist.labels(**labels).observe(value)
            else:
                hist.observe(value)
        except Exception:
            pass

    def gauge(self, name: str, value: float,
              labels: Optional[Dict[str, str]] = None) -> None:
        try:
            g = self._get_gauge(name, labels or {})
            if labels:
                g.labels(**labels).set(value)
            else:
                g.set(value)
        except Exception:
            pass


    def _metric_name(self, name: str) -> str:
        # name is already prefixed with "clampai_" by the kernel
        return name

    def _get_counter(self, name: str, labels: Dict[str, str]):
        if name not in self._counters:
            label_names = list(labels.keys())
            self._counters[name] = self._prom.Counter(
                self._metric_name(name),
                f"ClampAI counter: {name}",
                labelnames=label_names,
                registry=self._registry,
            )
        return self._counters[name]

    def _get_histogram(self, name: str, labels: Dict[str, str]):
        if name not in self._histograms:
            label_names = list(labels.keys())
            self._histograms[name] = self._prom.Histogram(
                self._metric_name(name),
                f"ClampAI histogram: {name}",
                labelnames=label_names,
                buckets=self._LATENCY_BUCKETS,
                registry=self._registry,
            )
        return self._histograms[name]

    def _get_gauge(self, name: str, labels: Dict[str, str]):
        if name not in self._gauges:
            label_names = list(labels.keys())
            self._gauges[name] = self._prom.Gauge(
                self._metric_name(name),
                f"ClampAI gauge: {name}",
                labelnames=label_names,
                registry=self._registry,
            )
        return self._gauges[name]

    def __repr__(self) -> str:
        return f"PrometheusMetrics(namespace={self._ns!r})"



class OTelMetrics:
    """
    MetricsBackend implementation backed by ``opentelemetry-api``.

    Requires: pip install opentelemetry-api

    Counters → OTel Counter
    observe() calls → OTel Histogram
    Gauges → OTel UpDownCounter (OTel does not have set-value Gauge in API)

    Args:
        meter:
            An ``opentelemetry.metrics.Meter`` instance obtained from
            ``opentelemetry.metrics.get_meter("clampai")``.  The caller is
            responsible for configuring the exporter (OTLP, Jaeger, etc.).
    """

    def __init__(self, meter) -> None:
        try:
            from opentelemetry import metrics as otel_metrics  # noqa: F401
        except ImportError:
            raise ImportError(
                "opentelemetry-api is required for OTelMetrics. "
                "Install it with: pip install 'clampai[opentelemetry]'"
            )
        self._meter = meter
        self._counters: Dict[str, object] = {}
        self._histograms: Dict[str, object] = {}
        self._updown_counters: Dict[str, object] = {}


    def increment(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        try:
            counter = self._get_counter(name)
            counter.add(1, attributes=labels or {})
        except Exception:
            pass

    def observe(self, name: str, value: float,
                labels: Optional[Dict[str, str]] = None) -> None:
        try:
            hist = self._get_histogram(name)
            hist.record(value, attributes=labels or {})
        except Exception:
            pass

    def gauge(self, name: str, value: float,
              labels: Optional[Dict[str, str]] = None) -> None:
        # OTel API doesn't provide a set-value Gauge; use UpDownCounter.
        # For absolute-value gauges (budget_remaining), callers should
        # reset between readings; UpDownCounter is best-effort here.
        try:
            udc = self._get_updown_counter(name)
            udc.add(value, attributes=labels or {})
        except Exception:
            pass


    def _get_counter(self, name: str):
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(
                name=name,
                description=f"ClampAI counter: {name}",
            )
        return self._counters[name]

    def _get_histogram(self, name: str):
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                name=name,
                description=f"ClampAI histogram: {name}",
                unit="s" if "latency" in name or "seconds" in name else "1",
            )
        return self._histograms[name]

    def _get_updown_counter(self, name: str):
        if name not in self._updown_counters:
            self._updown_counters[name] = self._meter.create_up_down_counter(
                name=name,
                description=f"ClampAI gauge: {name}",
            )
        return self._updown_counters[name]

    def __repr__(self) -> str:
        return f"OTelMetrics(meter={self._meter!r})"


class OTelTraceExporter:
    """
    Exports ClampAI audit ``TraceEntry`` objects as OpenTelemetry spans.

    Wire this into a ``SafetyKernel`` via the ``reconcile_fn`` hook or call
    ``export_entry()`` manually after each ``evaluate_and_execute_atomic()``
    to get a complete distributed-tracing view of every safety decision.

    Each ``TraceEntry`` becomes a single span with ClampAI-specific
    attributes:

        clampai.step             — step number in the kernel trace
        clampai.action_id        — action identifier
        clampai.action_name      — action display name
        clampai.cost             — budget charged for this action
        clampai.approved         — True if the action was executed
        clampai.state_before_fp  — fingerprint of state before the action
        clampai.state_after_fp   — fingerprint of state after the action
        clampai.reasoning        — LLM reasoning summary (if any)

    Rejected actions create a span with ``clampai.approved = False`` and
    ``clampai.rejection_reasons`` attribute listing the reasons.

    Args:
        tracer:
            An ``opentelemetry.trace.Tracer`` instance obtained from
            ``opentelemetry.trace.get_tracer("clampai")``. The caller is
            responsible for configuring the exporter (OTLP, Jaeger, etc.).

    Example::

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("clampai")
        exporter = OTelTraceExporter(tracer)

        kernel = SafetyKernel(budget=100.0, invariants=[...])
        new_state, entry = kernel.evaluate_and_execute_atomic(state, action)
        exporter.export_entry(entry)

    Requires: pip install clampai[opentelemetry]
    """

    def __init__(self, tracer: Any) -> None:
        try:
            from opentelemetry import trace as otel_trace  # noqa: F401
        except ImportError:
            raise ImportError(
                "opentelemetry-api is required for OTelTraceExporter. "
                "Install it with: pip install 'clampai[opentelemetry]'"
            )
        self._tracer = tracer

    def export_entry(self, entry: Any) -> None:
        """
        Export a single ``TraceEntry`` as an OpenTelemetry span.

        The span is created and immediately ended (point-in-time event).
        Export is best-effort: exceptions are silently swallowed.

        Args:
            entry: A ``clampai.formal.TraceEntry`` object.
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import StatusCode

            span_name = f"clampai.action.{getattr(entry, 'action_id', 'unknown')}"
            with self._tracer.start_as_current_span(span_name) as span:
                span.set_attribute("clampai.step", getattr(entry, "step", 0))
                span.set_attribute("clampai.action_id", getattr(entry, "action_id", ""))
                span.set_attribute("clampai.action_name", getattr(entry, "action_name", ""))
                span.set_attribute("clampai.cost", float(getattr(entry, "cost", 0.0)))
                span.set_attribute("clampai.approved", bool(getattr(entry, "approved", False)))
                span.set_attribute(
                    "clampai.state_before_fp", getattr(entry, "state_before_fp", "")
                )
                span.set_attribute(
                    "clampai.state_after_fp", getattr(entry, "state_after_fp", "")
                )
                reasoning = getattr(entry, "reasoning_summary", "")
                if reasoning:
                    span.set_attribute("clampai.reasoning", reasoning[:1024])

                rejection_reasons = getattr(entry, "rejection_reasons", ())
                if rejection_reasons:
                    span.set_attribute(
                        "clampai.rejection_reasons", "; ".join(rejection_reasons)
                    )

                if not getattr(entry, "approved", True):
                    span.set_status(
                        otel_trace.Status(StatusCode.ERROR, "action blocked by safety kernel")
                    )
        except Exception:
            pass  # trace export must never affect application logic

    def make_reconcile_fn(self) -> Any:
        """
        Return a ``reconcile_fn`` that auto-exports every committed action.

        Pass the returned callable to ``SafetyKernel(reconcile_fn=...)`` to
        automatically export every successful execution as an OTel span.

        Returns:
            A callable suitable for ``SafetyKernel(reconcile_fn=...)``.

        Example::

            exporter = OTelTraceExporter(tracer)
            kernel = SafetyKernel(
                budget=100.0,
                invariants=[...],
                reconcile_fn=exporter.make_reconcile_fn(),
            )
        """

        def _export_on_reconcile(
            model_state: Any,
            action: Any,
            entry: Any,
        ) -> None:
            self.export_entry(entry)
            return None  # do not modify the model state

        return _export_on_reconcile

    def __repr__(self) -> str:
        return f"OTelTraceExporter(tracer={self._tracer!r})"


__all__ = [
    "OTelMetrics",
    "OTelTraceExporter",
    "PrometheusMetrics",
]
