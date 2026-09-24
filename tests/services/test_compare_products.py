"""Only the comparison contract and the concurrency/safety guarantees."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from ozon_mcp.models.catalog import (
    Characteristic,
    DeliveryEstimate,
    ProductCard,
    Review,
    Reviews,
    Tile,
    Variant,
    VariantOption,
)
from ozon_mcp.services import comparison

if TYPE_CHECKING:
    import pytest

    from support import FakeSession


def _catalog(monkeypatch: pytest.MonkeyPatch, count: int = 2) -> None:
    def search(**kwargs: object) -> list[Tile]:
        return [Tile(sku=str(i), title=f"Tile {i}", price="100 ₽") for i in range(1, count + 1)]

    def details(sku: str) -> ProductCard:
        return ProductCard(
            sku=sku,
            title=f"Card {sku}",
            price="90 ₽",
            price_regular="110 ₽",
            available=True,
            characteristics=[Characteristic(name="Вес", value="2 кг")],
            rating=4.5,
            reviews_count=100,
            photos=["gallery"],
            variants=[Variant(name="Цвет", options=[VariantOption(sku="9")])],
        )

    def delivery(sku: str) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku, delivery="Завтра", address="Дом")

    def reviews(sku: str, limit: int) -> Reviews:
        return Reviews(count=100, reviews=[Review(sku=sku, text="Удобно", photos=["photo"])])

    monkeypatch.setattr(comparison.catalog, "search", search)
    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)


async def test_compact_contract(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch, count=3)

    def delivery(sku: str) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku, delivery="Завтра" if sku == "1" else None)

    def reviews(sku: str, limit: int) -> Reviews:
        assert limit == 3
        return Reviews(reviews=[Review(sku=sku, text="Удобно", photos=["photo"])]) if sku == "1" else Reviews()

    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар", limit=2, reviews_limit=3)
    data = result.model_dump()
    assert [product["sku"] for product in data["products"]] == ["1", "2"]
    assert data["products"][0]["price"] == "90 ₽"
    assert data["products"][0]["characteristics"] == [{"name": "Вес", "value": "2 кг"}]
    assert data["products"][1]["delivery"]["delivery"] is None
    assert data["products"][1]["errors"] == []
    assert data["review_groups"][1]["reviews"] == []
    assert data["review_groups"][0]["reviews"][0]["sku"] == "1"
    assert [product["reviews_ref"] for product in data["products"]] == ["1", "2"]
    assert not {"photos", "variants", "distribution"} & data["products"][0].keys()
    assert "gallery" not in str(data)
    assert "photo" not in str(data)


async def test_independent_source_failures(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch)

    def details(sku: str) -> ProductCard:
        if sku == "1":
            raise RuntimeError("bad card")
        return ProductCard(sku=sku, title="ok")

    def reviews(sku: str, limit: int) -> Reviews:
        if sku == "2":
            raise RuntimeError("bad reviews")
        return Reviews(count=3)

    def delivery(sku: str) -> DeliveryEstimate:
        if sku == "2":
            raise RuntimeError("bad delivery")
        return DeliveryEstimate(sku=sku, delivery="Завтра")

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар")
    assert result.products[0].title == "Tile 1"  # search data survives a failed card
    assert result.products[0].delivery is not None
    assert result.products[0].errors == ["details: bad card"]
    assert result.products[1].delivery is None
    assert result.products[1].errors == ["delivery: bad delivery", "reviews: bad reviews"]
    assert result.review_groups[1].error == "bad reviews"


async def test_shared_reviews(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch)

    def details(sku: str) -> ProductCard:
        return ProductCard(
            sku=sku, variants=[Variant(name="Цвет", options=[VariantOption(sku="1"), VariantOption(sku="2")])]
        )

    reads: list[str] = []

    def reviews(sku: str, limit: int) -> Reviews:
        reads.append(sku)
        return Reviews(count=50, reviews=[Review(sku="2", text="наш вариант"), Review(sku="3", text="чужой")])

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар")
    assert reads == ["1"]
    assert [product.reviews_ref for product in result.products] == ["1", "1"]
    assert [review.sku for review in result.review_groups[0].reviews] == ["2"]


async def test_reads_actually_overlap_but_are_bounded(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch, count=12)
    snapshot = object()
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=snapshot))
    barrier = threading.Barrier(2, timeout=3)
    lock = threading.Lock()
    running = peak = 0

    def details(sku: str, *, session: object) -> ProductCard:
        nonlocal running, peak
        assert session is snapshot
        with lock:
            running += 1
            peak = max(peak, running)
        if sku in {"1", "2"}:
            barrier.wait()
        with lock:
            running -= 1
        return ProductCard(sku=sku)

    def delivery(sku: str, *, session: object) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku)

    def reviews(sku: str, limit: int, *, session: object) -> Reviews:
        return Reviews()

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар", limit=12)
    assert len(result.products) == 12
    assert 2 <= peak <= 6
