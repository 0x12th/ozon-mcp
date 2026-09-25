"""A parallel read must fail closed rather than discard an auth-token rotation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import requests as curl_requests

from ozon_mcp.errors import OzonError, UpstreamError
from ozon_mcp.session import transport


class RotatingHTTP:
    def __init__(self) -> None:
        self.cookies = curl_requests.Cookies()
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            status_code=200,
            text='{"widgetStates":{}}',
            headers={"set-cookie": "__Secure-access-token=rotated; Path=/"},
        )

    def close(self) -> None:
        self.closed = True


def test_auth_rotation_is_not_silently_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    cookies = curl_requests.Cookies()
    cookies.set("__Secure-user-id", "123", domain=".ozon.ru", secure=True)
    http = RotatingHTTP()
    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_: http)
    snapshot = transport.ReadSnapshot((), tuple(cookies.jar), "chrome")
    with pytest.raises(OzonError, match="authentication cookies"):
        snapshot.fetch("/product/123456/")
    assert http.closed


@pytest.mark.parametrize(
    "text", ["<html>login</html>", "not JSON", "[]", '"hello"', '{"error":"denied"}', '{"errorForUser":"denied"}']
)
@pytest.mark.parametrize(("read", "args"), [("fetch", ("/my/orderlist",)), ("widget_state", ("state-1", "descriptor"))])
def test_snapshot_reads_reject_non_pages(
    monkeypatch: pytest.MonkeyPatch, text: str, read: str, args: tuple[str, ...]
) -> None:
    class ReadHTTP(RotatingHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(status_code=200, text=text, headers={})

    http = ReadHTTP()
    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_: http)
    snapshot = transport.ReadSnapshot((), (), "chrome")
    with pytest.raises(UpstreamError) as raised:
        getattr(snapshot, read)(*args)
    assert raised.value.status == 200
    assert http.closed


def test_snapshot_read_returns_page_with_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class ReadHTTP(RotatingHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(status_code=200, text='{"widgetStates":{}}', headers={})

    http = ReadHTTP()
    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_: http)
    snapshot = transport.ReadSnapshot((), (), "chrome")
    assert snapshot.fetch("/my/orderlist") == {"widgetStates": {}, "_httpStatus": 200}
    assert http.closed
