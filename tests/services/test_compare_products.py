"""Only the comparison contract and the concurrency/safety guarantees."""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from curl_cffi import requests as curl_requests
from prometheus_client import REGISTRY

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
from ozon_mcp.session import transport
from ozon_mcp.session.transport import SnapshotInvalidError
from ozon_mcp.utils.observability.review_probe import ReviewProbe, active_review_probe
from ozon_mcp.utils.serde import dumps
from support import page

if TYPE_CHECKING:
    from support import FakeSession


def _catalog(monkeypatch: pytest.MonkeyPatch, count: int = 2, reviews_limit: int = 10) -> None:
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

    def delivery(sku: str, *, allow_browser_fallback: bool = True) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku, delivery="Завтра", address="Дом")

    def reviews(sku: str, limit: int, sort: str = "useful") -> Reviews:
        return Reviews(count=100, reviews=[Review(sku=sku, text="Удобно", photos=["photo"])])

    monkeypatch.setattr(comparison.catalog, "search", search)
    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)

    def review_page(
        sku: str, sort: str, *, session: object = None, following: str | None = None
    ) -> tuple[Reviews, None]:
        answer = comparison.catalog.get_reviews(sku, limit=reviews_limit if sort == "useful" else 5, sort=sort)
        if (probe := active_review_probe()) is not None:
            probe.record_page(len(answer.reviews))
        return answer, None

    monkeypatch.setattr(comparison.catalog, "comparison_review_page", review_page)


async def test_compact_contract(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch, count=3, reviews_limit=3)

    def delivery(sku: str, *, allow_browser_fallback: bool = True) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku, delivery="Завтра" if sku == "1" else None)

    def reviews(sku: str, limit: int, sort: str = "useful") -> Reviews:
        assert limit == (3 if sort == "useful" else 5)
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

    def reviews(sku: str, limit: int, sort: str = "useful") -> Reviews:
        if sku == "2":
            raise RuntimeError("bad reviews")
        return Reviews(count=3)

    def delivery(sku: str, *, allow_browser_fallback: bool = True) -> DeliveryEstimate:
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
    assert result.products[1].errors == [
        "delivery: bad delivery",
        "reviews (useful): bad reviews",
        "reviews (worst): bad reviews",
    ]
    assert result.review_groups[1].error == "useful: bad reviews; worst: bad reviews"


async def test_shared_reviews(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch)

    def details(sku: str) -> ProductCard:
        return ProductCard(
            sku=sku, variants=[Variant(name="Цвет", options=[VariantOption(sku="1"), VariantOption(sku="2")])]
        )

    reads: list[str] = []

    def reviews(sku: str, limit: int, sort: str = "useful") -> Reviews:
        reads.append(f"{sku}:{sort}")
        return Reviews(
            count=50,
            reviews=[
                Review(sku="2", text="наш вариант", author="Автор", date="2026-09-01"),
                Review(sku="3", text="чужой"),
            ],
        )

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар")
    assert set(reads) == {"1:useful", "1:worst", "2:useful", "2:worst"}
    assert [product.reviews_ref for product in result.products] == ["1", "2"]
    assert result.review_groups[0].reviews == []
    assert [review.sku for review in result.review_groups[1].reviews] == ["2"]
    assert result.review_groups[0].coverage[0].matched == 0
    assert result.review_groups[1].coverage[0].matched == 1


async def test_shared_card_second_page_is_visible_after_sku_filter(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Shared cursor walk sees B even when A fills the first page."""
    sku_a, sku_b = "1000001", "2000002"
    get_reviews = comparison.catalog.get_reviews
    review_page = comparison.catalog.comparison_review_page
    _catalog(monkeypatch)
    monkeypatch.setattr(comparison.catalog, "get_reviews", get_reviews)
    monkeypatch.setattr(comparison.catalog, "comparison_review_page", review_page)
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=None))

    def details(sku: str) -> ProductCard:
        return ProductCard(
            sku=sku,
            variants=[Variant(name="Цвет", options=[VariantOption(sku=sku_a), VariantOption(sku=sku_b)])],
        )

    def listing(review_sku: str, size: int, *, next_page: bool) -> dict[str, object]:
        paging: dict[str, object] = {"total": 40}
        if next_page:
            paging["nextButton"] = "?page=2&page_key=NEXT"
        return page(
            webReviewProductScore={"totalScore": 4.5, "reviewsCount": 40},
            webListReviews={
                "reviews": [
                    {
                        "itemId": review_sku,
                        "author": {"firstName": f"Author {index}"},
                        "content": {"score": 5, "comment": f"review {review_sku}-{index}"},
                    }
                    for index in range(size)
                ],
                "paging": paging,
            },
        )

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    session.pages = {
        "page_key=NEXT": listing(sku_b, 10, next_page=False),
        "/reviews/": listing(sku_a, 30, next_page=True),
    }
    received_before = REGISTRY.get_sample_value("ozon_mcp_compare_products_reviews_received_sum") or 0
    discarded_before = REGISTRY.get_sample_value("ozon_mcp_compare_products_reviews_discarded_other_sku_sum") or 0
    with caplog.at_level(logging.INFO, logger="ozon_mcp"):
        result = await comparison.compare_products(skus=[sku_a, sku_b], reviews_limit=10)
    groups = {group.sku: group for group in result.review_groups}
    assert len(groups[sku_a].reviews) == 10
    assert [
        (coverage.sort, coverage.matched, coverage.complete, coverage.pages_scanned)
        for coverage in groups[sku_b].coverage
    ] == [
        ("useful", 10, True, 2),
        ("worst", 10, True, 2),
    ]
    assert len(groups[sku_b].reviews) == 10
    # One continuation per sort, shared by both SKUs, not a full walk per SKU.
    assert sum("page_key=NEXT" in url for url in session.fetched) == 2
    # A deeper direct read follows the cursor and confirms B reviews do exist.
    deeper = get_reviews(sku_b, limit=40)
    assert sum(review.sku == sku_b for review in deeper.reviews) == 10
    assert any("page_key=NEXT" in url for url in session.fetched)
    summaries = [record.message for record in caplog.records if record.message.startswith("tool=compare_products")]
    assert len(summaries) == 1
    assert "reviews_received=140" in summaries[0]
    assert "reviews_matched=80" in summaries[0]
    assert "reviews_returned=20" in summaries[0]
    assert "review_pages=6" in summaries[0]
    assert "reviews_without_sku=0" in summaries[0]
    assert "review_requests=0" in summaries[0]  # FakeSession makes no actual HTTP requests.
    assert "review_groups_diag=" not in summaries[0]
    assert groups[sku_b].coverage[0].meaningful == 10
    assert sku_a not in summaries[0]
    assert sku_b not in summaries[0]
    assert REGISTRY.get_sample_value("ozon_mcp_compare_products_reviews_received_sum") == received_before + 140
    discarded_after = REGISTRY.get_sample_value("ozon_mcp_compare_products_reviews_discarded_other_sku_sum")
    assert discarded_after is not None
    assert discarded_after == discarded_before + 80


async def test_review_read_probe_counts_one_real_transport_attempt_and_raw_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_data = page(
        webListReviews={
            "reviews": [{"itemId": "1000001", "content": {"comment": f"text {index}"}} for index in range(30)],
            "paging": {"total": 30},
        }
    )
    calls: list[str] = []

    class FakeHTTP:
        def __init__(self) -> None:
            self.cookies = curl_requests.Cookies()

        def request(self, method: str, url: str, **kwargs: object) -> SimpleNamespace:
            calls.append(url)
            return SimpleNamespace(status_code=200, text=dumps(page_data), headers={})

        def close(self) -> None:
            pass

    monkeypatch.setattr(transport.curl_requests, "Session", lambda **_kwargs: FakeHTTP())
    probe = ReviewProbe()
    snapshot = transport.ReadSnapshot((), (), "chrome")
    reviews, error = await comparison._read(
        comparison._ReadState(snapshot),
        lambda client: comparison.catalog.get_reviews("1000001", limit=10, session=client),
        lambda: (_ for _ in ()).throw(AssertionError("serial fallback")),
        probe=probe,
    )
    assert error is None
    assert reviews is not None
    assert reviews.fetched == 10
    assert len(calls) == 1
    measured = probe.snapshot()
    assert (measured.requests, measured.pages, measured.raw_reviews) == (1, 1, 30)
    assert measured.wait_seconds >= 0
    assert measured.read_seconds >= 0


def test_review_probe_separates_semaphore_wait_from_worker_read(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count().__next__

    class WaitingSemaphore:
        def __enter__(self) -> None:
            ticks()  # Deterministically advance the clock while waiting for a slot.

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(comparison, "_READ_SLOTS", WaitingSemaphore())
    monkeypatch.setattr(comparison.time, "perf_counter", ticks)
    probe = ReviewProbe()
    assert comparison._limited(lambda: "done", probe=probe) == "done"
    measured = probe.snapshot()
    assert measured.wait_seconds == 2
    assert measured.read_seconds == 1


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

    def delivery(sku: str, *, session: object, allow_browser_fallback: bool = True) -> DeliveryEstimate:
        return DeliveryEstimate(sku=sku)

    def reviews(sku: str, limit: int, sort: str = "useful", *, session: object) -> Reviews:
        return Reviews()

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products("товар", limit=12)
    assert len(result.products) == 12
    assert 2 <= peak <= 6


async def test_coverage_is_per_sku_even_with_identical_variants(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _catalog(monkeypatch)

    def details(sku: str) -> ProductCard:
        return ProductCard(
            sku=sku,
            characteristics=[Characteristic(name=f"Поле {i}", value="да") for i in range(13)],
            variants=[Variant(name="Цвет", options=[VariantOption(sku="1"), VariantOption(sku="2")])],
        )

    def reviews(sku: str, limit: int, sort: str) -> Reviews:
        return Reviews(count=100, reviews=[Review(sku="1", text="Розетка с заземлением"), Review(text="неизвестно")])

    monkeypatch.setattr(comparison.catalog, "product_details", details)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    for order in ("1", "2"), ("2", "1"):
        result = await comparison.compare_products(skus=list(order), reviews_limit=10)
        groups = {group.sku: group for group in result.review_groups}
        assert len(result.products[0].characteristics) == 13
        assert groups["1"].coverage[0].matched == 1
        assert groups["2"].coverage[0].matched == 0
        assert groups["2"].coverage[0].scanned == 2
        assert groups["2"].coverage[0].unattributed == 1
        assert not groups["2"].coverage[0].sufficient
        assert groups["2"].reviews == []


async def test_selected_skus_do_not_repeat_search_and_only_gaps_get_deeper_reads(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _catalog(monkeypatch)

    def no_search(**_kwargs: object) -> list[Tile]:
        raise AssertionError("searched")

    monkeypatch.setattr(comparison.catalog, "search", no_search)
    reads: list[tuple[str, str, int]] = []

    def reviews(sku: str, limit: int, sort: str) -> Reviews:
        reads.append((sku, sort, limit))
        count = limit if sku == "1" else 0
        return Reviews(count=50, reviews=[Review(sku=sku, text=str(i)) for i in range(count)])

    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    result = await comparison.compare_products(skus=["1", "2"], reviews_limit=3)
    assert [product.sku for product in result.products] == ["1", "2"]
    assert ("1", "useful", 10) in reads
    assert ("2", "useful", 10) in reads
    assert all(group.coverage[0].sort == "useful" for group in result.review_groups)


async def test_read_limit_is_shared_across_simultaneous_comparisons(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _catalog(monkeypatch, count=8)
    snapshot = object()
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=snapshot))
    lock = threading.Lock()
    running = peak = 0

    def details(sku: str, *, session: object) -> ProductCard:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.01)
        with lock:
            running -= 1
        return ProductCard(sku=sku)

    monkeypatch.setattr(comparison.catalog, "product_details", details)

    def delivery(sku: str, *, session: object, allow_browser_fallback: bool = True) -> DeliveryEstimate:
        assert session is snapshot
        return DeliveryEstimate(sku=sku)

    def reviews(sku: str, limit: int, sort: str, *, session: object) -> Reviews:
        assert session is snapshot
        return Reviews()

    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    monkeypatch.setattr(comparison.catalog, "get_reviews", reviews)
    await asyncio.gather(comparison.compare_products("товар"), comparison.compare_products("товар"))
    assert 2 <= peak <= 6


async def test_deadline_preserves_completed_data_and_marks_missing_reads(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _catalog(monkeypatch)
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=object()))
    monkeypatch.setattr(
        comparison, "get_settings", lambda: SimpleNamespace(comparison_timeout=0.01, request_timeout=0.01)
    )

    def details(sku: str, *, session: object) -> ProductCard:
        time.sleep(0.05)
        return ProductCard(sku=sku)

    monkeypatch.setattr(comparison.catalog, "product_details", details)

    def delivery(sku: str, *, session: object, allow_browser_fallback: bool = True) -> DeliveryEstimate:
        assert session is not None
        return DeliveryEstimate(sku=sku)

    monkeypatch.setattr(comparison.catalog, "delivery_estimate", delivery)
    result = await comparison.compare_products(skus=["1", "2"])
    assert [product.sku for product in result.products] == ["1", "2"]
    assert all(any("comparison deadline exceeded" in error for error in product.errors) for product in result.products)
    await asyncio.sleep(0.06)  # Allow already-running worker threads to release their process-wide slots.


async def test_snapshot_is_disabled_only_after_credential_failure() -> None:
    state = comparison._ReadState(object())

    def transient(client: object) -> str:
        raise RuntimeError("transient")

    assert await comparison._read(state, transient, lambda: "serial") == ("serial", None)
    assert not state.failed
    assert await comparison._read(state, lambda _client: "parallel", lambda: "serial") == ("parallel", None)

    def invalid(client: object) -> str:
        raise SnapshotInvalidError("rotated")

    assert await comparison._read(state, invalid, lambda: "serial") == ("serial", None)
    assert state.failed
    assert await comparison._read(state, lambda _client: "parallel", lambda: "serial") == ("serial", None)


async def test_explicit_skus_do_not_inherit_search_limit(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _catalog(monkeypatch)
    result = await comparison.compare_products(skus=[str(i) for i in range(12)], reviews_limit=1)
    assert len(result.products) == 12
    assert len(result.review_groups) == 12


async def test_phase_and_review_metrics_record_one_call_without_wall_clock(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    _catalog(monkeypatch)
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=None))
    monkeypatch.setattr(comparison, "_clock", itertools.count(1).__next__)
    phases = ("search", "snapshot", "details_delivery", "reviews", "serialize")

    def sample(name: str, labels: dict[str, str] | None = None) -> float:
        return REGISTRY.get_sample_value(name, labels) or 0

    before = {
        phase: (
            sample("ozon_mcp_compare_products_phase_seconds_count", {"phase": phase}),
            sample("ozon_mcp_compare_products_phase_seconds_sum", {"phase": phase}),
        )
        for phase in phases
    }
    reviews_before = sample("ozon_mcp_compare_products_reviews_received_sum")
    reviews_count_before = sample("ozon_mcp_compare_products_reviews_received_count")
    matched_before = sample("ozon_mcp_compare_products_reviews_matched_sum")
    products_before = sample("ozon_mcp_compare_products_products_count")
    groups_before = sample("ozon_mcp_compare_products_review_groups_count")
    incomplete_before = sample("ozon_mcp_compare_products_incomplete_fraction_count")

    with caplog.at_level(logging.INFO, logger="ozon_mcp"):
        result = await comparison.compare_products("товар")
    assert len(result.products) == len(result.review_groups) == 2
    for phase in phases:
        count, total = before[phase]
        assert sample("ozon_mcp_compare_products_phase_seconds_count", {"phase": phase}) == count + 1
        assert sample("ozon_mcp_compare_products_phase_seconds_sum", {"phase": phase}) == total + 1
    assert sample("ozon_mcp_compare_products_reviews_received_sum") == reviews_before + 4
    assert sample("ozon_mcp_compare_products_reviews_received_count") == reviews_count_before + 1
    assert sample("ozon_mcp_compare_products_reviews_matched_sum") == matched_before + 4
    assert REGISTRY.get_sample_value("ozon_mcp_compare_products_reviews_fetched_count") is None
    assert sample("ozon_mcp_compare_products_products_count") == products_before + 1
    assert sample("ozon_mcp_compare_products_review_groups_count") == groups_before + 1
    assert sample("ozon_mcp_compare_products_incomplete_fraction_count") == incomplete_before + 1
    summaries = [record.message for record in caplog.records if record.message.startswith("tool=compare_products")]
    assert len(summaries) == 1
    assert "total=" in summaries[0]
    assert "reviews_received=4 reviews_matched=4" in summaries[0]
    assert all(f"{phase}=1.000" in summaries[0] for phase in phases)
    assert "товар" not in summaries[0]


async def test_failed_search_still_records_phase_and_summary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(comparison, "_clock", itertools.count(1).__next__)

    def failed_search(**_kwargs: object) -> list[Tile]:
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(comparison.catalog, "search", failed_search)
    labels = {"phase": "search"}
    before = REGISTRY.get_sample_value("ozon_mcp_compare_products_phase_seconds_count", labels) or 0
    with caplog.at_level(logging.INFO, logger="ozon_mcp"), pytest.raises(RuntimeError, match="source unavailable"):
        await comparison.compare_products("товар")
    assert REGISTRY.get_sample_value("ozon_mcp_compare_products_phase_seconds_count", labels) == before + 1
    summaries = [record.message for record in caplog.records if record.message.startswith("tool=compare_products")]
    assert len(summaries) == 1
    assert "search=1.000 snapshot=n/a" in summaries[0]
    assert "товар" not in summaries[0]


def test_delivery_date_requires_explicit_year() -> None:
    assert comparison._delivery_date("Доставим 9 сентября 2026") == "2026-09-09"
    assert comparison._delivery_date("Доставим 9 сентября") is None
    assert comparison._delivery_date("Завтра") is None
    assert comparison._delivery_date("32 сентября 2026") is None
