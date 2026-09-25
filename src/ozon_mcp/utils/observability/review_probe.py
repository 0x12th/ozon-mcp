"""Worker-local diagnostic counters for review reads (no request data)."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class ReviewProbeSnapshot:
    requests: int
    raw_reviews: int
    pages: int
    wait_seconds: float
    read_seconds: float


class ReviewProbe:
    """Aggregate diagnostic counts safely across workers sharing a probe."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = 0
        self._raw_reviews = 0
        self._pages = 0
        self._wait_seconds = 0.0
        self._read_seconds = 0.0

    def record_request(self) -> None:
        with self._lock:
            self._requests += 1

    def record_page(self, count: int) -> None:
        with self._lock:
            self._pages += 1
            self._raw_reviews += count

    def record_wait(self, seconds: float) -> None:
        with self._lock:
            self._wait_seconds += seconds

    def record_read(self, seconds: float) -> None:
        with self._lock:
            self._read_seconds += seconds

    def snapshot(self) -> ReviewProbeSnapshot:
        with self._lock:
            return ReviewProbeSnapshot(
                requests=self._requests,
                raw_reviews=self._raw_reviews,
                pages=self._pages,
                wait_seconds=self._wait_seconds,
                read_seconds=self._read_seconds,
            )


_ACTIVE_REVIEW_PROBE: ContextVar[ReviewProbe | None] = ContextVar("active_review_probe", default=None)


@contextmanager
def track_review_probe(probe: ReviewProbe | None) -> Iterator[None]:
    """Bind a probe to this worker for the duration of the block."""
    token = _ACTIVE_REVIEW_PROBE.set(probe)
    try:
        yield
    finally:
        _ACTIVE_REVIEW_PROBE.reset(token)


def active_review_probe() -> ReviewProbe | None:
    """Return the probe bound to this worker, if any."""
    return _ACTIVE_REVIEW_PROBE.get()


def _record_active_request() -> None:
    """Count one HTTP attempt only when a worker has opted into diagnostics."""
    probe = active_review_probe()
    if probe is not None:
        probe.record_request()
