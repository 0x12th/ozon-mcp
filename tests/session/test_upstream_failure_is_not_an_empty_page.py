"""A failed request must not look like an empty account.

Every parser turns a page with no widgets into an empty list, so returning
``{"widgetStates": {}}`` on a 502 or a timeout reported "you have no orders".
The tests drive the real request path with a stand-in for the HTTP session — the
seam the transport is built around — rather than patching its methods.
"""

from __future__ import annotations

import time
from typing import Any, override

import pytest

from ozon_mcp.errors import RateLimitedError, UpstreamError
from ozon_mcp.session.transport import OzonSession, comparison_read_limit
from ozon_mcp.utils.serde import dumps


class _Response:
    def __init__(self, status: int, text: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.text = text
        self.headers = headers or {}


class _Http:
    """Stands in for the curl_cffi session: hands back a scripted answer each call."""

    def __init__(self, *answers: _Response | Exception) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.timeouts: list[float] = []

    def request(self, *_args: Any, **kwargs: Any) -> _Response:
        self.timeouts.append(kwargs["timeout"])
        self.calls += 1
        answer = self.answers[min(self.calls, len(self.answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Session(OzonSession):
    """The transport with the browser side stubbed out."""

    def __init__(self, http: _Http) -> None:
        super().__init__()
        self._http = http
        self._headers = {"user-agent": "test"}
        self.rebootstraps = 0
        self.waits: list[tuple[int, float | None]] = []

    @override
    def _ensure_http(self) -> None:  # the browser is not part of these tests
        return

    @override
    def save_state(self) -> None:
        return

    @override
    def _rebootstrap(self) -> None:
        self.rebootstraps += 1

    @override
    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        self.waits.append((attempt, retry_after))


PAGE = dumps({"widgetStates": {"orderList-1": "{}"}})


def test_comparison_read_limit_bounds_http_and_retries_without_changing_writes() -> None:
    http = _Http(_Response(502, "unavailable"), _Response(200, PAGE))
    session = _Session(http)
    with comparison_read_limit(0.2):
        with pytest.raises(UpstreamError):
            session.fetch("/my/orderlist")
        assert session.action("favoriteCreateList", {})["widgetStates"]
    assert http.calls == 2
    assert http.timeouts[0] == pytest.approx(0.2)
    assert http.timeouts[1] > 0.2


def test_a_server_error_raises_instead_of_answering_empty() -> None:
    session = _Session(_Http(_Response(502, "<html>bad gateway</html>")))
    with pytest.raises(UpstreamError) as raised:
        session.fetch("/my/orderlist")
    assert raised.value.status == 502
    # The message has to be relayable: it says nothing was read.
    assert "not an empty account" in str(raised.value)


def test_a_transport_failure_raises_too() -> None:
    session = _Session(_Http(TimeoutError("connection timed out")))
    with pytest.raises(UpstreamError) as raised:
        session.fetch("/my/orderlist")
    assert raised.value.status == 0


def test_a_server_error_that_clears_is_retried_not_reported() -> None:
    session = _Session(_Http(_Response(500, "oops"), _Response(200, PAGE)))
    page = session.fetch("/my/orderlist")
    assert page["widgetStates"]
    assert session.waits == [(1, None)]


def test_a_rate_limit_honours_the_wait_ozon_asked_for() -> None:
    session = _Session(_Http(_Response(429, "slow down", {"Retry-After": "7"}), _Response(200, PAGE)))
    session.fetch("/my/orderlist")
    assert session.waits == [(1, pytest.approx(7.0))]


def test_a_rate_limit_that_does_not_clear_says_so() -> None:
    limited = _Response(429, "slow down", {"Retry-After": "3"})
    session = _Session(_Http(limited, limited, limited))
    with pytest.raises(RateLimitedError) as raised:
        session.fetch("/my/orderlist")
    assert raised.value.retry_after == pytest.approx(3.0)


def test_an_antibot_challenge_re_bootstraps_rather_than_waiting() -> None:
    session = _Session(_Http(_Response(403, '{"incidentId": "x"}'), _Response(200, PAGE)))
    session.fetch("/my/orderlist")
    assert session.rebootstraps == 1
    assert session.waits == []


def test_a_refused_action_is_passed_through_not_raised() -> None:
    # Ozon reports refused actions as JSON with a 4xx; the caller needs the
    # reason ("Пустое название вишлиста"), not an exception.
    body = dumps({"error": "Пустое название вишлиста"})
    session = _Session(_Http(_Response(400, body)))
    answer = session.action("favoriteCreateList", {"title": ""})
    assert answer["error"] == "Пустое название вишлиста"


def test_a_client_error_with_no_json_raises() -> None:
    # An HTML 404 carries nothing to act on, and would read as an empty page.
    session = _Session(_Http(_Response(404, "<html>not found</html>")))
    with pytest.raises(UpstreamError) as raised:
        session.fetch("/my/nope")
    assert raised.value.status == 404


@pytest.mark.parametrize(
    "text", ["<html>login</html>", "not JSON", "[]", '"hello"', '{"error":"denied"}', '{"errorForUser":"denied"}']
)
@pytest.mark.parametrize(("read", "args"), [("fetch", ("/my/orderlist",)), ("widget_state", ("state-1", "descriptor"))])
def test_successful_serial_reads_reject_non_pages(text: str, read: str, args: tuple[str, ...]) -> None:
    session = _Session(_Http(_Response(200, text)))
    with pytest.raises(UpstreamError) as raised:
        getattr(session, read)(*args)
    assert raised.value.status == 200
    assert "not an empty account" in str(raised.value)


def test_successful_serial_read_returns_page_with_status() -> None:
    session = _Session(_Http(_Response(200, PAGE)))
    assert session.fetch("/my/orderlist") == {"widgetStates": {"orderList-1": "{}"}, "_httpStatus": 200}


def test_direct_page_conversion_rejects_invalid_read_response() -> None:
    session = _Session(_Http())
    with pytest.raises(UpstreamError) as raised:
        session._page_from("<html>login</html>", 200, "composer", time.monotonic())
    assert raised.value.status == 200


def test_direct_get_request_rejects_error_envelope() -> None:
    session = _Session(_Http(_Response(200, '{"errorForUser":"denied"}')))
    with pytest.raises(UpstreamError) as raised:
        session._request("GET", "https://www.ozon.ru/my/orderlist")
    assert raised.value.status == 200


def test_client_error_json_is_not_a_read_page() -> None:
    session = _Session(_Http(_Response(400, '{"error":"denied"}')))
    with pytest.raises(UpstreamError) as raised:
        session.fetch("/my/orderlist")
    assert raised.value.status == 400


@pytest.mark.parametrize("method", ["action", "post_page"])
def test_write_response_semantics_remain_unchanged(method: str) -> None:
    session = _Session(_Http(_Response(200, "<html>not JSON</html>"), _Response(200, '{"error":"refused"}')))
    write = session.action if method == "action" else session.post_page
    assert write("favoriteCreateList", {"title": ""}) == {"_httpStatus": 200}
    assert write("favoriteCreateList", {"title": ""}) == {"error": "refused", "_httpStatus": 200}
