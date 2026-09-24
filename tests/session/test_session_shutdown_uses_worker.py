"""Chromium must be closed on the same worker that owns its Playwright objects."""

from __future__ import annotations

import threading
from functools import cache
from types import SimpleNamespace

import pytest

from ozon_mcp import __main__ as entrypoint, dependencies


def test_close_only_existing_session_on_its_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    closed_on: list[int] = []

    @cache
    def session() -> SimpleNamespace:
        return SimpleNamespace(close=lambda: closed_on.append(threading.get_ident()))

    monkeypatch.setattr(dependencies, "get_session", session)
    dependencies.close_session()
    assert session.cache_info().currsize == 0

    session()
    worker_id = dependencies.get_executor().submit(threading.get_ident).result()
    dependencies.close_session()
    assert closed_on == [worker_id]
    assert session.cache_info().currsize == 0


def test_entrypoint_closes_session_if_server_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    def stop() -> None:
        raise RuntimeError("server stopped")

    monkeypatch.setattr(entrypoint, "get_settings", lambda: SimpleNamespace(transport="stdio"))
    monkeypatch.setattr(entrypoint.mcp, "run", stop)
    monkeypatch.setattr(entrypoint, "close_session", lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="server stopped"):
        entrypoint.main()
    assert closed == [True]
