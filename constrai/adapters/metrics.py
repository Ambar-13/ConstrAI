"""
constrai.adapters.metrics — Pluggable observability backends.

Provides ``PrometheusMetrics`` and ``OTelMetrics``, both satisfying the
``constrai.MetricsBackend`` protocol defined in ``constrai.formal``.

    from constrai.adapters.metrics import PrometheusMetrics
    kernel = SafetyKernel(budget=100.0, invariants=[...],
                          metrics=PrometheusMetrics())

    # Custom registry for tests or multi-process isolation:
    registry = prometheus_client.CollectorRegistry()
    kernel = SafetyKernel(..., metrics=PrometheusMetrics(registry=registry))

    # OpenTelemetry:
    meter = opentelemetry.metrics.get_meter("constrai")
    kernel = SafetyKernel(..., metrics=OTelMetrics(meter))

Metric names emitted (documented fully in BENCHMARKS.md):

    constrai_actions_total{status="approved|rejected", kernel_id="..."}
    constrai_kernel_latency_seconds{operation="evaluate|execute", ...}
    constrai_budget_utilization_ratio{kernel_id="..."}
    constrai_budget_remaining{kernel_id="..."}
    constrai_step_count{kernel_id="..."}
    constrai_invariant_violations_total{invariant="...", enforcement="..."}
    constrai_rollbacks_total{kernel_id="..."}

All metric emission failures are silently swallowed as required by the
MetricsBackend protocol. A broken registry or misconfigured exporter will
never affect safety-critical kernel logic.

Requires: pip install constrai[prometheus] or constrai[opentelemetry]
"""

from __future__ import annotations

from typing import Dict, Optional


class PrometheusMetrics:
    """
    MetricsBackend implementation backed by ``prometheus_client``.

    Requires: pip install prometheus_client

    Counters and gauges are created lazily on first use.  Histograms use
    Prometheus-default buckets optimised for the 0.06 ms ConstrAI baseline
    plus a fine-grained low-latency range.

    Thread-safety: prometheus_client metrics are thread-safe.

    Args:
    registry:
        Prometheus CollectorRegistry.  Defaults to the global
        ``REGISTRY`` (suitable for single-process servers).  Pass a
        custom registry for tests or multi-process deployments.
    namespace:
        Metric name prefix.  Default: ``"constrai"``.
    """

    _LATENCY_BUCKETS = (
        0.0001, 0.0002, 0.0005,          # < 1 ms (ConstrAI normal range)
        0.001, 0.002, 0.005,             # 1–5 ms
        0.01, 0.025, 0.05, 0.1,         # 10–100 ms
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0, # > 100 ms (slow / degraded)
    )

    def __init__(self, registry=None, namespace: str = "constrai") -> None:
        try:
            import prometheus_client as prom
        except ImportError:
            raise ImportError(
                "prometheus_client is required for PrometheusMetrics. "
                "Install it with: pip install 'constrai[prometheus]'"
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
        # name is already prefixed with "constrai_" by the kernel
        return name

    def _get_counter(self, name: str, labels: Dict[str, str]):
        if name not in self._counters:
            label_names = list(labels.keys())
            self._counters[name] = self._prom.Counter(
                self._metric_name(name),
                f"ConstrAI counter: {name}",
                labelnames=label_names,
                registry=self._registry,
            )
        return self._counters[name]

    def _get_histogram(self, name: str, labels: Dict[str, str]):
        if name not in self._histograms:
            label_names = list(labels.keys())
            self._histograms[name] = self._prom.Histogram(
                self._metric_name(name),
                f"ConstrAI histogram: {name}",
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
                f"ConstrAI gauge: {name}",
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
        ``opentelemetry.metrics.get_meter("constrai")``.  The caller is
        responsible for configuring the exporter (OTLP, Jaeger, etc.).
    """

    def __init__(self, meter) -> None:
        try:
            from opentelemetry import metrics as otel_metrics  # noqa: F401
        except ImportError:
            raise ImportError(
                "opentelemetry-api is required for OTelMetrics. "
                "Install it with: pip install 'constrai[opentelemetry]'"
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
                description=f"ConstrAI counter: {name}",
            )
        return self._counters[name]

    def _get_histogram(self, name: str):
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                name=name,
                description=f"ConstrAI histogram: {name}",
                unit="s" if "latency" in name or "seconds" in name else "1",
            )
        return self._histograms[name]

    def _get_updown_counter(self, name: str):
        if name not in self._updown_counters:
            self._updown_counters[name] = self._meter.create_up_down_counter(
                name=name,
                description=f"ConstrAI gauge: {name}",
            )
        return self._updown_counters[name]

    def __repr__(self) -> str:
        return f"OTelMetrics(meter={self._meter!r})"
