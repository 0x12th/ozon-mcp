from typing import Final

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

"""Prometheus instrumentation, self-contained.

Signals are taken at the transport seam rather than per tool: what actually
breaks in production is the Ozon side (antibot re-challenge, token expiry,
upstream 4xx/5xx), and that all funnels through one request path. Process-level
metrics come from prometheus_client's default collectors.
"""


METRICS_PATH: Final = "/metrics"

UPSTREAM_REQUESTS: Final = Counter(
    "ozon_mcp_upstream_requests_total",
    "Requests to Ozon, by backend and outcome.",
    ["backend", "outcome"],
)

UPSTREAM_LATENCY: Final = Histogram(
    "ozon_mcp_upstream_request_seconds",
    "Latency of requests to Ozon.",
    ["backend"],
)

SESSION_BOOTSTRAPS: Final = Counter(
    "ozon_mcp_session_bootstraps_total",
    "Browser bootstraps performed to clear the antibot and harvest a session.",
    ["reason"],
)

BROWSER_ACTIVE: Final = Gauge(
    "ozon_mcp_browser_active",
    "1 while a Chromium instance is held open, 0 when only HTTP is live.",
)

COMPARISON_FALLBACKS: Final = Counter(
    "ozon_mcp_comparison_fallbacks_total",
    "Comparison reads switched to the serial session, by reason.",
    ["reason"],
)

COMPARE_PRODUCTS_DURATION: Final = Histogram(
    "ozon_mcp_compare_products_duration_seconds",
    "Duration of compare_products calls in seconds.",
)

COMPARE_PRODUCTS_PHASE: Final = Histogram(
    "ozon_mcp_compare_products_phase_seconds",
    "Time spent in each comparison phase, including scheduling and local processing.",
    ["phase"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 60),
)


COMPARE_PRODUCTS_REVIEW_UPSTREAM_REQUESTS: Final = Histogram(
    "ozon_mcp_compare_products_review_upstream_requests",
    "HTTP attempts made by review reads in one comparison, including failed attempts.",
    buckets=(0, 1, 2, 5, 10, 20, 30, 40, 60, 100),
)

COMPARE_PRODUCTS_REVIEW_PAGES_SCANNED: Final = Histogram(
    "ozon_mcp_compare_products_review_pages_scanned",
    "Number of actual review pages parsed in one comparison (shared pages counted once).",
    buckets=(0, 1, 2, 5, 10, 20, 30, 40, 60, 100),
)

COMPARE_PRODUCTS_REVIEWS_RECEIVED: Final = Histogram(
    "ozon_mcp_compare_products_reviews_received",
    "Parsed review records on upstream pages before the requested depth is applied.",
    buckets=(0, 10, 30, 60, 150, 300, 600, 1200, 2400),
)

COMPARE_PRODUCTS_REVIEWS_MATCHED: Final = Histogram(
    "ozon_mcp_compare_products_reviews_matched",
    "Records matching their target SKU in sorted samples, before meaningful filtering and deduplication.",
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500, 1000),
)

COMPARE_PRODUCTS_REVIEWS_RETURNED: Final = Histogram(
    "ozon_mcp_compare_products_reviews_returned",
    "Meaningful attributable reviews retained after per-group deduplication.",
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500),
)

COMPARE_PRODUCTS_REVIEWS_DISCARDED_OTHER_SKU: Final = Histogram(
    "ozon_mcp_compare_products_reviews_discarded_other_sku",
    "Per-SKU attribution discards of other variants (shared pages may be considered for multiple SKUs).",
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500),
)

COMPARE_PRODUCTS_REVIEWS_WITHOUT_SKU: Final = Histogram(
    "ozon_mcp_compare_products_reviews_without_sku",
    "Per-SKU unattributed records with no numeric itemId (shared pages may be counted for multiple SKUs).",
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500),
)

COMPARE_PRODUCTS_REVIEW_SEMAPHORE_WAIT: Final = Histogram(
    "ozon_mcp_compare_products_review_semaphore_wait_seconds",
    "Sum of worker semaphore wait times per comparison (concurrent waits can overlap).",
    buckets=(0.01, 0.1, 0.5, 1, 2.5, 5, 10, 20, 40, 80),
)

COMPARE_PRODUCTS_REVIEW_READ: Final = Histogram(
    "ozon_mcp_compare_products_review_read_seconds",
    "Sum of review worker execution times per comparison, including parsing (workers overlap).",
    buckets=(0.01, 0.1, 0.5, 1, 2.5, 5, 10, 20, 40, 80),
)

COMPARE_PRODUCTS_RESPONSE_SIZE: Final = Histogram(
    "ozon_mcp_compare_products_response_bytes",
    "Size of serialized compare_products JSON responses in bytes.",
    buckets=(1024, 4096, 16384, 65536, 262144, 1048576, 4194304),
)

COMPARE_PRODUCTS_PRODUCT_COUNT: Final = Histogram(
    "ozon_mcp_compare_products_products",
    "Number of products returned by compare_products per call.",
    buckets=(0, 1, 2, 3, 5, 10, 15, 20),
)

COMPARE_PRODUCTS_REVIEW_GROUP_COUNT: Final = Histogram(
    "ozon_mcp_compare_products_review_groups",
    "Number of review groups returned by compare_products per call.",
    buckets=(0, 1, 2, 3, 5, 10, 15, 20),
)

COMPARE_PRODUCTS_INCOMPLETE_FRACTION: Final = Histogram(
    "ozon_mcp_compare_products_incomplete_fraction",
    "Fraction of compared products with errors or insufficient useful-review coverage; zero for no products.",
    buckets=(0, 0.1, 0.25, 0.5, 0.75, 1),
)


def metrics_endpoint(_request: Request) -> Response:
    """Scrape endpoint; sync so Starlette runs the blocking encode off the loop."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
