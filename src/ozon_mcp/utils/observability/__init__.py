"""Observability surface: Prometheus metrics and the scrape endpoint."""

from ozon_mcp.utils.observability.metrics import (
    BROWSER_ACTIVE,
    COMPARISON_FALLBACKS,
    METRICS_PATH,
    SESSION_BOOTSTRAPS,
    UPSTREAM_LATENCY,
    UPSTREAM_REQUESTS,
    metrics_endpoint,
)

__all__ = [
    "BROWSER_ACTIVE",
    "COMPARISON_FALLBACKS",
    "METRICS_PATH",
    "SESSION_BOOTSTRAPS",
    "UPSTREAM_LATENCY",
    "UPSTREAM_REQUESTS",
    "metrics_endpoint",
]
