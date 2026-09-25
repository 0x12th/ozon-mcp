"""Review diagnostics count actual requests without changing transport behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, override

import pytest
from curl_cffi import requests as curl_requests

from ozon_mcp.errors import UpstreamError
from ozon_mcp.session import transport
from ozon_mcp.utils.observability.review_probe import ReviewProbe, active_review_probe, track_review_probe


class FakeHTTP:
    def __init__(self, *answers: int | Exception) -> None:
        self.answers = iter(answers)
        self.calls = 0
        self.cookies = curl_requests.Cookies()
        self.closed = False

    def request(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(status_code=answer, text='{"widgetStates":{}}', headers={})

    def close(self) -> None:
        self.closed = True


class SerialSession(transport.OzonSession):
    def __init__(self, http: FakeHTTP) -> None:
        super().__init__()
        self._http = http
        self._headers = {}

    @override
    def _ensure_http(self) -> None:
        pass

    @override
    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        pass

    @override
    def save_state(self) -> None:
        pass


@pytest.mark.parametrize("first", [500, TimeoutError("offline")])
def test_serial_probe_counts_each_http_attempt(first: int | Exception) -> None:
    http = FakeHTTP(first, 200)
    probe = ReviewProbe()
    with track_review_probe(probe):
        assert SerialSession(http).fetch("/product/123/reviews/")["widgetStates"] == {}
    assert http.calls == 2
    assert probe.snapshot().requests == 2
    assert active_review_probe() is None


def test_serial_probe_counts_final_failure() -> None:
    http = FakeHTTP(*[TimeoutError("offline") for _ in range(3)])
    probe = ReviewProbe()
    with track_review_probe(probe), pytest.raises(UpstreamError):
        SerialSession(http).fetch("/product/123/reviews/")
    assert http.calls == probe.snapshot().requests == 3


def test_snapshot_probe_counts_failed_http_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP(TimeoutError("offline"))
    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_: http)
    probe = ReviewProbe()
    with track_review_probe(probe), pytest.raises(UpstreamError):
        transport.ReadSnapshot((), (), "chrome").fetch("/product/123/reviews/")
    assert http.calls == probe.snapshot().requests == 1
    assert http.closed


def test_snapshot_probe_counts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP(200)
    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_: http)
    probe = ReviewProbe()
    with track_review_probe(probe):
        assert transport.ReadSnapshot((), (), "chrome").fetch("/product/123/reviews/")["widgetStates"] == {}
    assert http.calls == probe.snapshot().requests == 1
    assert http.closed


def test_probe_context_is_nested_and_worker_local() -> None:
    outer, inner = ReviewProbe(), ReviewProbe()
    with track_review_probe(outer):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(active_review_probe).result() is None
            assert pool.submit(lambda: _record_in_worker(inner)).result() == 1
            assert pool.submit(active_review_probe).result() is None
        with track_review_probe(inner):
            assert active_review_probe() is inner
            with track_review_probe(None):
                assert active_review_probe() is None
            assert active_review_probe() is inner
        assert active_review_probe() is outer
    assert active_review_probe() is None


def _record_in_worker(probe: ReviewProbe) -> int:
    with track_review_probe(probe):
        probe.record_request()
        return probe.snapshot().requests


def test_probe_snapshot_is_immutable_and_aggregates_counts() -> None:
    probe = ReviewProbe()
    probe.record_request()
    probe.record_page(30)
    probe.record_page(2)
    probe.record_wait(0.25)
    probe.record_read(1.5)
    captured = probe.snapshot()
    assert (captured.requests, captured.raw_reviews, captured.pages) == (1, 32, 2)
    assert (captured.wait_seconds, captured.read_seconds) == (0.25, 1.5)
    with pytest.raises(FrozenInstanceError):
        captured.requests = 9
    probe.record_page(1)
    assert captured.raw_reviews == 32
    assert probe.snapshot().raw_reviews == 33
