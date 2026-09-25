"""SKU-aware comparison walks through bounded, shared review pages."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ozon_mcp.models.catalog import DeliveryEstimate, ProductCard, Review, Reviews, Variant, VariantOption
from ozon_mcp.services import comparison
from support import page

if TYPE_CHECKING:
    from support import FakeSession


def _listing(items: list[str | None], next_cursor: str | None = None) -> dict[str, object]:
    return page(
        webReviewProductScore={"totalScore": 4.5, "reviewsCount": 100},
        webListReviews={
            "reviews": [
                {
                    "itemId": sku,
                    "author": {"firstName": f"person-{i}"},
                    "content": {"comment": f"comment-{i}"},
                }
                for i, sku in enumerate(items)
            ],
            "paging": {"total": 100, **({"nextButton": f"?page=2&page_key={next_cursor}"} if next_cursor else {})},
        },
    )


def _setup(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    monkeypatch.setattr(comparison, "_snapshot", lambda: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        comparison.catalog,
        "product_details",
        lambda sku: ProductCard(
            sku=sku,
            variants=[Variant(name="Цвет", options=[VariantOption(sku="1000001"), VariantOption(sku="2000002")])],
        ),
    )
    monkeypatch.setattr(
        comparison.catalog,
        "delivery_estimate",
        lambda sku, *, allow_browser_fallback=True: DeliveryEstimate(sku=sku),  # ruff: ignore[unused-lambda-argument]
    )


async def test_budget_exhaustion_keeps_matches_and_marks_coverage_incomplete(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)
    session.pages = {
        "page_key=THIRD": _listing(["1000001"] * 28 + ["2000002"] * 2, "FOURTH"),
        "page_key=SECOND": _listing(["1000001"] * 30, "THIRD"),
        "/reviews/": _listing(["1000001"] * 30, "SECOND"),
    }
    result = await comparison.compare_products(skus=["2000002"], reviews_limit=10)
    coverage = result.review_groups[0].coverage[0]
    assert (coverage.sort, coverage.target, coverage.matched, coverage.pages_scanned, coverage.scanned) == (
        "useful",
        10,
        2,
        3,
        90,
    )
    assert not coverage.complete
    assert coverage.stop_reason == "budget"
    assert len(result.review_groups[0].reviews) >= 2  # Both sorted samples may contain the same unidentifiable reviews.
    assert not any("page_key=FOURTH" in url for url in session.fetched)


async def test_exact_limit_and_unattributed_reviews(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _setup(monkeypatch, session)
    session.pages = {"/reviews/": _listing([None] * 3 + ["2000002"] * 20)}
    result = await comparison.compare_products(skus=["2000002"], reviews_limit=10)
    group = result.review_groups[0]
    assert len(group.reviews) == 10
    assert group.coverage[0].matched == 20
    assert group.coverage[0].unattributed == 3
    assert group.coverage[0].complete
    assert group.coverage[0].stop_reason == "target_met"
    assert all(review.sku == "2000002" for review in group.reviews)


async def test_unattributed_only_is_not_evidence_for_sku(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    _setup(monkeypatch, session)
    session.pages = {"/reviews/": _listing([None] * 30)}
    group = (await comparison.compare_products(skus=["2000002"], reviews_limit=10)).review_groups[0]
    assert group.reviews == []
    assert group.coverage[0].unattributed == 30
    assert group.coverage[0].matched == 0
    assert not group.coverage[0].complete
    assert group.coverage[0].stop_reason == "source_exhausted"


async def test_variant_lists_alone_do_not_merge_distinct_review_feeds(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)
    session.pages = {
        "page_key=A_NEXT": _listing(["1000001"] * 10),
        "page_key=B_NEXT": _listing(["2000002"] * 10),
        "/product/1000001/reviews/": _listing(["1000001"] + [None] * 29, "A_NEXT"),
        "/product/2000002/reviews/": _listing(["2000002"] + [None] * 29, "B_NEXT"),
    }
    result = await comparison.compare_products(skus=["1000001", "2000002"], reviews_limit=10)
    assert all(group.coverage[0].complete for group in result.review_groups)
    assert sum("page_key=A_NEXT" in path for path in session.fetched) == 2
    assert sum("page_key=B_NEXT" in path for path in session.fetched) == 2


async def test_partial_upstream_failure_preserves_first_page(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)

    def fail() -> dict[str, object]:
        raise RuntimeError("upstream read failed")

    session.pages = {
        "page_key=SECOND": fail,
        "/reviews/": _listing(["2000002"] + ["1000001"] * 29, "SECOND"),
    }
    result = await comparison.compare_products(skus=["2000002"], reviews_limit=10)
    group = result.review_groups[0]
    assert group.coverage[0].matched == 1
    assert group.coverage[0].pages_scanned == 1
    assert not group.coverage[0].complete
    assert group.coverage[0].stop_reason == "error"
    assert len(group.reviews) >= 1
    assert group.error is not None
    assert "upstream read failed" in group.error
    assert any("upstream read failed" in error for error in result.products[0].errors)


@pytest.mark.parametrize(
    ("useful_count", "worst_count", "limit", "expected_useful", "expected_worst"),
    [
        (10, 5, 10, 7, 3),
        (10, 0, 10, 10, 0),
        (2, 5, 10, 2, 5),
        (10, 5, 1, 1, 0),
        (0, 5, 1, 0, 1),
        (10, 5, 2, 1, 1),
        (10, 5, 3, 2, 1),
        (10, 5, 4, 3, 1),
        (10, 5, 20, 10, 5),
    ],
)
async def test_review_display_quota_reallocates_unused_slots_without_changing_coverage(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
    useful_count: int,
    worst_count: int,
    limit: int,
    expected_useful: int,
    expected_worst: int,
) -> None:
    _setup(monkeypatch, session)

    def review_page(
        sku: str, sort: str, *, session: object = None, following: str | None = None
    ) -> tuple[Reviews, None]:
        size = useful_count if sort == "useful" else worst_count
        return Reviews(
            count=100,
            reviews=[
                Review(sku=sku, author=f"author-{sort}-{index}", date="2026-09-01", text=f"{sort}-{index}")
                for index in range(size)
            ],
        ), None

    monkeypatch.setattr(comparison.catalog, "comparison_review_page", review_page)
    result = await comparison.compare_products(skus=["2000002"], reviews_limit=limit)
    group = result.review_groups[0]
    assert len(group.reviews) <= limit
    assert sum(review.sort == "useful" for review in group.reviews) == expected_useful
    assert sum(review.sort == "worst" for review in group.reviews) == expected_worst
    useful, worst = group.coverage
    assert (useful.sort, useful.target, useful.meaningful) == ("useful", limit, useful_count)
    assert (worst.sort, worst.target, worst.meaningful) == ("worst", 5, worst_count)
    assert useful.complete is (useful_count >= limit)
    assert worst.complete is (worst_count >= 5)


async def test_reliable_identity_deduplicates_across_sorts_and_backfills(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)

    def review_page(
        sku: str, sort: str, *, session: object = None, following: str | None = None
    ) -> tuple[Reviews, None]:
        shared = [
            Review(sku=sku, author=f"shared-{index}", date="2026-09-01", text=f"shared-{index}") for index in range(3)
        ]
        distinct = [
            Review(sku=sku, author=f"{sort}-{index}", date="2026-09-01", text=f"{sort}-{index}")
            for index in range(7 if sort == "useful" else 2)
        ]
        return Reviews(count=100, reviews=shared + distinct), None

    monkeypatch.setattr(comparison.catalog, "comparison_review_page", review_page)
    group = (await comparison.compare_products(skus=["2000002"], reviews_limit=10)).review_groups[0]
    assert len(group.reviews) == 10
    assert [review.sort for review in group.reviews].count("worst") == 2
    assert [review.sort for review in group.reviews].count("useful") == 8
    assert group.coverage[0].meaningful == 10
    assert group.coverage[1].meaningful == 5
    assert group.coverage[0].complete
    assert group.coverage[1].complete
    assert len({review.text for review in group.reviews}) == 10


async def test_no_next_cursor_wins_over_page_budget_when_not_clipped(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)
    session.pages = {
        "page_key=THIRD": _listing(["1000001"] * 30),
        "page_key=SECOND": _listing(["1000001"] * 30, "THIRD"),
        "/reviews/": _listing(["1000001"] * 30, "SECOND"),
    }
    coverage = (await comparison.compare_products(skus=["2000002"], reviews_limit=10)).review_groups[0].coverage[0]
    assert (coverage.pages_scanned, coverage.scanned, coverage.stop_reason) == (3, 90, "source_exhausted")
    assert not coverage.complete


async def test_clipped_page_is_budget_even_without_next_cursor(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)
    session.pages = {"/reviews/": _listing(["1000001"] * 90 + ["2000002"] * 10)}
    coverage = (await comparison.compare_products(skus=["2000002"], reviews_limit=10)).review_groups[0].coverage[0]
    assert (coverage.pages_scanned, coverage.scanned, coverage.matched) == (1, 90, 0)
    assert coverage.stop_reason == "budget"
    assert not coverage.complete


async def test_target_met_takes_precedence_over_error_from_shared_walk(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)

    def fail() -> dict[str, object]:
        raise RuntimeError("upstream read failed")

    session.pages = {
        "page_key=SECOND": fail,
        "/reviews/": _listing(["1000001", "1000001", "2000002"], "SECOND"),
    }
    result = await comparison.compare_products(skus=["1000001", "2000002"], reviews_limit=2)
    first, second = result.review_groups
    assert (first.coverage[0].meaningful, first.coverage[0].stop_reason, first.coverage[0].complete) == (
        2,
        "target_met",
        True,
    )
    assert (second.coverage[0].meaningful, second.coverage[0].stop_reason, second.coverage[0].complete) == (
        1,
        "error",
        False,
    )
    assert first.error is not None  # The other SKU still required a failed next page.


async def test_repeated_cursor_is_error_not_source_exhaustion(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession
) -> None:
    _setup(monkeypatch, session)
    session.pages = {
        "page_key=LOOP": _listing(["1000001"] * 30, "LOOP"),
        "/reviews/": _listing(["1000001"] * 30, "LOOP"),
    }
    group = (await comparison.compare_products(skus=["2000002"], reviews_limit=10)).review_groups[0]
    assert group.coverage[0].stop_reason == "error"
    assert group.error is not None
    assert "cursor repeated" in group.error
