"""A parallel read must fail closed rather than discard an auth-token rotation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import requests as curl_requests

from ozon_mcp.errors import OzonError
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
