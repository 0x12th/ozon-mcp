"""Bounded, read-only comparison built on the catalog service, not on MCP calls."""

import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

from ozon_mcp.dependencies import run_blocking
from ozon_mcp.models.catalog import (
    ComparedProduct,
    ComparisonDelivery,
    ComparisonReview,
    ComparisonReviews,
    ProductCard,
    ProductComparison,
    Review,
    ReviewCoverage,
    Reviews,
    Tile,
)
from ozon_mcp.services import catalog
from ozon_mcp.session.transport import ReadSnapshot, SnapshotInvalidError, comparison_read_limit
from ozon_mcp.settings import get_settings
from ozon_mcp.utils.observability import COMPARISON_FALLBACKS
from ozon_mcp.utils.observability.metrics import (
    COMPARE_PRODUCTS_DURATION,
    COMPARE_PRODUCTS_INCOMPLETE_FRACTION,
    COMPARE_PRODUCTS_PHASE,
    COMPARE_PRODUCTS_PRODUCT_COUNT,
    COMPARE_PRODUCTS_RESPONSE_SIZE,
    COMPARE_PRODUCTS_REVIEW_GROUP_COUNT,
    COMPARE_PRODUCTS_REVIEW_PAGES_SCANNED,
    COMPARE_PRODUCTS_REVIEW_READ,
    COMPARE_PRODUCTS_REVIEW_SEMAPHORE_WAIT,
    COMPARE_PRODUCTS_REVIEW_UPSTREAM_REQUESTS,
    COMPARE_PRODUCTS_REVIEWS_DISCARDED_OTHER_SKU,
    COMPARE_PRODUCTS_REVIEWS_MATCHED,
    COMPARE_PRODUCTS_REVIEWS_RECEIVED,
    COMPARE_PRODUCTS_REVIEWS_RETURNED,
    COMPARE_PRODUCTS_REVIEWS_WITHOUT_SKU,
)
from ozon_mcp.utils.observability.review_probe import ReviewProbe, track_review_probe

logger = logging.getLogger("ozon_mcp")
_clock = time.perf_counter
_MAX_CONCURRENT = 6
# Acquired inside worker threads so cancellation cannot release a slot while HTTP is still running.
_READ_SLOTS = threading.BoundedSemaphore(_MAX_CONCURRENT)
_WORST_REVIEWS = 5
# Ozon currently returns 30 reviews per page. At most three pages (90 records)
# per sorted pool may be inspected, even when a rare variant has no matches.
_REVIEW_MAX_PAGES = 3
_REVIEW_MAX_SCANNED = 90
_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
_FULL_DATE = re.compile(r"(?<!\d)(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})(?!\d)", re.IGNORECASE)


@contextmanager
def _phase(
    name: Literal["search", "snapshot", "details_delivery", "reviews", "serialize"],
    durations: dict[str, float],
) -> Iterator[None]:
    started = _clock()
    try:
        yield
    finally:
        elapsed = _clock() - started
        durations[name] = elapsed
        COMPARE_PRODUCTS_PHASE.labels(phase=name).observe(elapsed)


def _delivery_date(line: str | None) -> str | None:
    """Never infer a year or a timezone from an ambiguous delivery label."""
    match = _FULL_DATE.search(line or "")
    if match is None:
        return None
    try:
        return date(int(match[3]), _MONTHS[match[2].lower()], int(match[1])).isoformat()
    except ValueError:
        return None


class _ReadState:
    def __init__(self, snapshot: ReadSnapshot | None) -> None:
        self.snapshot = snapshot
        self.failed = False


class _ReviewDiagnostics:
    """Per-call observations; SKUs are used for grouping but never exported to metrics or logs."""

    def __init__(self) -> None:
        self.samples: list[tuple[tuple[str, ...], ReviewProbe, Reviews | None]] = []


def _review_counts(result: ProductComparison, diagnostics: _ReviewDiagnostics) -> dict[str, int]:
    return {
        "matched": sum(coverage.matched for group in result.review_groups for coverage in group.coverage),
        "returned": sum(len(group.reviews) for group in result.review_groups),
        "without_sku": sum(coverage.unattributed for group in result.review_groups for coverage in group.coverage),
        "discarded_other_sku": sum(
            review.sku is not None and review.sku != sku
            for skus, _, page in diagnostics.samples
            if page is not None
            for sku in skus
            for review in page.reviews
        ),
    }


def _limited[T](work: Callable[[], T], *, probe: ReviewProbe | None = None) -> T:
    waiting_since = time.perf_counter() if probe is not None else 0.0
    with _READ_SLOTS:
        if probe is not None:
            probe.record_wait(time.perf_counter() - waiting_since)
        reading_since = time.perf_counter() if probe is not None else 0.0
        try:
            timeout = min(get_settings().request_timeout, max(0.1, get_settings().comparison_timeout / 3))
            with comparison_read_limit(timeout), track_review_probe(probe):
                return work()
        finally:
            if probe is not None:
                probe.record_read(time.perf_counter() - reading_since)


async def _snapshot() -> ReadSnapshot | None:
    """Prepare an isolated read client on the session thread after search."""
    try:
        return await run_blocking(lambda: catalog.get_session().snapshot_reads())
    except Exception:  # Serial reads remain available if snapshotting fails.
        COMPARISON_FALLBACKS.labels(reason="snapshot").inc()
        logger.warning("comparison snapshot failed; using serial reads", exc_info=True)
        return None


async def _read[T](
    state: _ReadState,
    parallel: Callable[[ReadSnapshot], T],
    serial: Callable[[], T],
    *,
    probe: ReviewProbe | None = None,
) -> tuple[T | None, Exception | None]:
    snapshot = state.snapshot
    if snapshot is not None and not state.failed:

        def isolated() -> T:
            if state.failed:
                raise SnapshotInvalidError("comparison snapshot disabled")
            return parallel(snapshot)

        try:
            return await asyncio.to_thread(lambda: _limited(isolated, probe=probe)), None
        except Exception as exc:  # Retry a failed isolated read with the original session.
            if isinstance(exc, SnapshotInvalidError):
                state.failed = True
            COMPARISON_FALLBACKS.labels(reason="read").inc()
            logger.warning("comparison parallel read failed; retrying on the session thread", exc_info=True)
    try:
        return await run_blocking(lambda: _limited(serial, probe=probe)), None
    except Exception as exc:  # ruff: ignore[blind-except] - one source must not discard other products
        return None, exc


def _apply_card(product: ComparedProduct, card: ProductCard | None) -> None:
    if card is None:
        return
    product.title = card.title or product.title
    product.price = card.price or product.price
    product.price_regular = card.price_regular or product.price_regular
    product.available = card.available
    product.characteristics = card.characteristics
    product.rating = card.rating
    product.reviews_count = card.reviews_count


def _select_products(tiles: list[Tile], limit: int) -> list[ComparedProduct]:
    selected: list[Tile] = []
    seen: set[str] = set()
    for tile in tiles:
        if tile.sku and tile.sku not in seen:
            selected.append(tile)
            seen.add(tile.sku)
        if len(selected) == limit:
            break
    return [
        ComparedProduct(
            sku=tile.sku, title=tile.title, url=tile.url, price=tile.price, price_regular=tile.price_regular
        )
        for tile in selected
        if tile.sku is not None
    ]


async def _bounded[T](
    reads: list[asyncio.Task[tuple[T | None, Exception | None]]], deadline: float
) -> list[tuple[T | None, Exception | None]]:
    """Keep completed reads when the comparison's budget expires."""
    if not reads:
        return []
    done, pending = await asyncio.wait(reads, timeout=max(0, deadline - time.monotonic()))
    for task in pending:
        task.cancel()
    return [task.result() if task in done else (None, TimeoutError("comparison deadline exceeded")) for task in reads]


@dataclass
class _ReviewSample:
    sku: str
    sort: Literal["useful", "worst"]
    target: int
    reviews: list[Review] = field(default_factory=list)
    score: float | None = None
    count: int | None = None
    cursor: str | None = None
    pages: int = 0
    scanned: int = 0
    clipped: bool = False
    error: Exception | None = None


def _meaningful(review: Review) -> bool:
    return bool(review.text or review.positive or review.negative)


def _append_page(sample: _ReviewSample, page: Reviews, cursor: str | None) -> None:
    remaining = max(0, _REVIEW_MAX_SCANNED - sample.scanned)
    sample.reviews.extend(page.reviews[:remaining])
    sample.clipped |= len(page.reviews) > remaining
    sample.scanned += min(len(page.reviews), remaining)
    sample.pages += 1
    sample.cursor = cursor
    sample.score = page.score if page.score is not None else sample.score
    sample.count = page.count if page.count is not None else sample.count


def _review_identity(review: Review) -> tuple[str | None, ...] | None:
    # Without author and date there is no reliable identity across sorted samples.
    if not review.author or not review.date:
        return None
    return (review.sku, review.author, review.date, review.text, review.positive, review.negative)


def _meaningful_count(sample: _ReviewSample) -> int:
    seen: set[tuple[str | None, ...]] = set()
    count = 0
    for review in sample.reviews:
        if review.sku != sample.sku or not _meaningful(review):
            continue
        key = _review_identity(review)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        count += 1
    return count


def _stop_reason(
    sample: _ReviewSample, meaningful: int
) -> Literal["target_met", "budget", "source_exhausted", "error"]:
    if meaningful >= sample.target:
        return "target_met"
    if sample.error is not None:
        return "error"
    if sample.clipped:
        return "budget"
    return "budget" if sample.cursor else "source_exhausted"


def _finish_sample(sample: _ReviewSample, group: ComparisonReviews) -> None:
    group.rating = sample.score if sample.score is not None else group.rating
    group.count = sample.count if sample.count is not None else group.count
    matched = sum(review.sku == sample.sku for review in sample.reviews)
    meaningful = _meaningful_count(sample)
    complete = meaningful >= sample.target
    group.coverage.append(
        ReviewCoverage(
            sort=sample.sort,
            scanned=sample.scanned,
            matched=matched,
            meaningful=meaningful,
            unattributed=sum(review.sku is None for review in sample.reviews),
            target=sample.target,
            pages_scanned=sample.pages,
            stop_reason=_stop_reason(sample, meaningful),
            complete=complete,
            sufficient=complete,
        )
    )
    if sample.error is not None:
        message = f"{sample.sort}: {sample.error}"
        group.error = f"{group.error}; {message}" if group.error else message


def _worst_quota(limit: int) -> int:
    # Reserve roughly 30% for complaints, including one slot for small samples.
    return min(_WORST_REVIEWS, max(1, limit * 3 // 10)) if limit > 1 else 0


def _take_reviews(
    sample: _ReviewSample,
    group: ComparisonReviews,
    seen: set[tuple[str | None, ...]],
    used: set[int],
    limit: int,
) -> None:
    for position, review in enumerate(sample.reviews):
        if len(group.reviews) >= limit:
            break
        if position in used or review.sku != sample.sku or not _meaningful(review):
            continue
        key = _review_identity(review)
        if key is not None and key in seen:
            continue
        used.add(position)
        if key is not None:
            seen.add(key)
        group.reviews.append(
            ComparisonReview(
                sort=sample.sort,
                sku=review.sku,
                variant=review.variant,
                score=review.score,
                positive=review.positive,
                negative=review.negative,
                text=review.text,
            )
        )


def _same_pool_key(sample: _ReviewSample) -> tuple[object, ...] | None:
    # Identical variant lists on product cards do not prove a shared review feed.
    # Require a nonempty, identical actual first page AND identical pagination cursor.
    if not sample.reviews or sample.error is not None:
        return None
    return (
        sample.sort,
        sample.cursor,
        sample.count,
        tuple(review.model_dump_json() for review in sample.reviews),
    )


async def _review_page(
    state: _ReadState, sku: str, sort: str, following: str | None, probe: ReviewProbe
) -> tuple[tuple[Reviews, str | None] | None, Exception | None]:
    return await _read(
        state,
        lambda client: catalog.comparison_review_page(sku, sort, session=client, following=following),
        lambda: catalog.comparison_review_page(sku, sort, following=following),
        probe=probe,
    )


async def _initial_review_pages(
    samples: list[_ReviewSample], state: _ReadState, deadline: float, diagnostics: _ReviewDiagnostics
) -> None:
    probes = [ReviewProbe() for _ in samples]
    first = await _bounded(
        [
            asyncio.create_task(_review_page(state, sample.sku, sample.sort, None, probe))
            for sample, probe in zip(samples, probes, strict=True)
        ],
        deadline,
    )
    for sample, probe, (response, error) in zip(samples, probes, first, strict=True):
        diagnostics.samples.append(((sample.sku,), probe, response[0] if response else None))
        if response is not None:
            _append_page(sample, *response)
        sample.error = error


def _shared_walks(samples: list[_ReviewSample]) -> list[list[_ReviewSample]]:
    # Only pages demonstrably identical on the wire share subsequent cursor walks.
    walks: list[list[_ReviewSample]] = []
    keys: dict[tuple[object, ...], list[_ReviewSample]] = {}
    for sample in samples:
        key = _same_pool_key(sample)
        if key is not None and key in keys:
            keys[key].append(sample)
        else:
            walk = [sample]
            walks.append(walk)
            if key is not None:
                keys[key] = walk
    return walks


def _targets_met(walk: list[_ReviewSample]) -> bool:
    return all(_meaningful_count(sample) >= sample.target for sample in walk)


async def _walk_pages(walk: list[_ReviewSample], state: _ReadState, diagnostics: _ReviewDiagnostics) -> None:
    owner = walk[0]
    visited: set[str] = set()
    while owner.cursor and owner.pages < _REVIEW_MAX_PAGES and owner.scanned < _REVIEW_MAX_SCANNED:
        cursor = owner.cursor
        if _targets_met(walk):
            break
        if cursor in visited:
            for sample in walk:
                sample.error = RuntimeError("review pagination cursor repeated")
            break
        visited.add(cursor)
        probe = ReviewProbe()
        response, error = await _review_page(state, owner.sku, owner.sort, cursor, probe)
        page = response[0] if response else None
        diagnostics.samples.append((tuple(sample.sku for sample in walk), probe, page))
        if error is not None:
            for sample in walk:
                sample.error = error
            break
        if response is None:
            break
        for sample in walk:
            _append_page(sample, *response)
        if not page or not page.reviews:
            break


async def _continue_review_walk(
    walk: list[_ReviewSample], state: _ReadState, diagnostics: _ReviewDiagnostics
) -> tuple[None, Exception | None]:
    try:
        await _walk_pages(walk, state, diagnostics)
    except Exception as exc:  # ruff: ignore[blind-except] - preserve data from earlier pages
        return None, exc
    else:
        return None, None


def _finalize_reviews(
    products: list[ComparedProduct], samples: list[_ReviewSample], reviews_limit: int
) -> list[ComparisonReviews]:
    groups = {product.sku: ComparisonReviews(sku=product.sku) for product in products}
    by_sku = {product.sku: product for product in products}
    by_sort: dict[tuple[str, str], _ReviewSample] = {}
    for sample in samples:
        _finish_sample(sample, groups[sample.sku])
        by_sort[sample.sku, sample.sort] = sample
        if sample.error is not None:
            by_sku[sample.sku].errors.append(f"reviews ({sample.sort}): {sample.error}")
    useful_quota = reviews_limit - _worst_quota(reviews_limit)
    for product in products:
        group = groups[product.sku]
        useful = by_sort[product.sku, "useful"]
        worst = by_sort[product.sku, "worst"]
        seen: set[tuple[str | None, ...]] = set()
        used_useful: set[int] = set()
        used_worst: set[int] = set()
        _take_reviews(useful, group, seen, used_useful, useful_quota)
        _take_reviews(worst, group, seen, used_worst, reviews_limit)
        _take_reviews(useful, group, seen, used_useful, reviews_limit)
        _take_reviews(worst, group, seen, used_worst, reviews_limit)
        product.reviews_ref = product.sku
    return list(groups.values())


async def _collect_reviews(
    products: list[ComparedProduct],
    state: _ReadState,
    reviews_limit: int,
    deadline: float,
    diagnostics: _ReviewDiagnostics,
) -> list[ComparisonReviews]:
    samples = [
        _ReviewSample(product.sku, sort, reviews_limit if sort == "useful" else _WORST_REVIEWS)
        for product in products
        for sort in ("useful", "worst")
    ]
    await _initial_review_pages(samples, state, deadline, diagnostics)
    walks = _shared_walks(samples)
    results = await _bounded(
        [asyncio.create_task(_continue_review_walk(walk, state, diagnostics)) for walk in walks], deadline
    )
    for walk, (_, error) in zip(walks, results, strict=True):
        if error is not None:
            for sample in walk:
                sample.error = error
    return _finalize_reviews(products, samples, reviews_limit)


async def _enrich_cards(products: list[ComparedProduct], state: _ReadState, deadline: float) -> None:
    cards = [
        asyncio.create_task(
            _read(
                state,
                lambda client, sku=product.sku: catalog.product_details(sku, session=client),
                lambda sku=product.sku: catalog.product_details(sku),
            )
        )
        for product in products
    ]
    deliveries = [
        asyncio.create_task(
            _read(
                state,
                lambda client, sku=product.sku: catalog.delivery_estimate(
                    sku, session=client, allow_browser_fallback=False
                ),
                lambda sku=product.sku: catalog.delivery_estimate(sku, allow_browser_fallback=False),
            )
        )
        for product in products
    ]
    card_results = await _bounded(cards, deadline)
    delivery_results = await _bounded(deliveries, deadline)
    for product, (card, details_error), (delivery, delivery_error) in zip(
        products, card_results, delivery_results, strict=True
    ):
        if details_error:
            product.errors.append(f"details: {details_error}")

        _apply_card(product, card)
        if delivery_error:
            product.errors.append(f"delivery: {delivery_error}")
        if delivery:
            product.delivery = ComparisonDelivery(
                delivery=delivery.delivery,
                date=_delivery_date(delivery.delivery),
                address=delivery.address,
                source=delivery.source,
            )


async def _select_source(
    query: str | None,
    limit: int,
    sort: str,
    skus: list[str] | None,
    category: str | None,
    filters: dict[str, str] | None,
    deadline: float,
) -> list[ComparedProduct]:
    if skus is not None:
        return [ComparedProduct(sku=sku) for sku in dict.fromkeys(skus)]
    tiles = await asyncio.wait_for(
        run_blocking(
            lambda: _limited(
                lambda: catalog.search(query=query, category=category, filters=filters, sort=sort, limit=limit)
            )
        ),
        timeout=max(0.001, deadline - time.monotonic()),
    )
    return _select_products(tiles, limit)


async def _take_snapshot(products: list[ComparedProduct], deadline: float) -> ReadSnapshot | None:
    if not products or time.monotonic() >= deadline:
        return None
    try:
        return await asyncio.wait_for(_snapshot(), timeout=max(0.001, deadline - time.monotonic()))
    except TimeoutError:
        COMPARISON_FALLBACKS.labels(reason="snapshot_timeout").inc()
        return None


async def _compare(
    query: str | None,
    limit: int,
    reviews_limit: int,
    sort: str,
    skus: list[str] | None,
    category: str | None,
    filters: dict[str, str] | None,
    *,
    durations: dict[str, float],
    diagnostics: _ReviewDiagnostics,
) -> ProductComparison:
    deadline = time.monotonic() + get_settings().comparison_timeout
    with _phase("search", durations):
        products = await _select_source(query, limit, sort, skus, category, filters, deadline)
    answer = ProductComparison(query=query, fetched_at=datetime.now(UTC).isoformat(), products=products)
    with _phase("snapshot", durations):
        state = _ReadState(await _take_snapshot(products, deadline))
    with _phase("details_delivery", durations):
        await _enrich_cards(products, state, deadline)

    reviews_started = _clock()
    answer.review_groups = await _collect_reviews(products, state, reviews_limit, deadline, diagnostics)
    durations["reviews"] = _clock() - reviews_started
    COMPARE_PRODUCTS_PHASE.labels(phase="reviews").observe(durations["reviews"])
    return answer


async def compare_products(
    query: str | None = None,
    limit: int = 10,
    reviews_limit: int = 10,
    sort: str = "popular",
    *,
    skus: list[str] | None = None,
    category: str | None = None,
    filters: dict[str, str] | None = None,
) -> ProductComparison:
    """Compare search hits or selected SKUs, retaining independent failures."""
    if skus is None and not (query or category):
        raise ValueError("provide query/category or skus")
    if skus is not None and (query is not None or category is not None or filters is not None):
        raise ValueError("skus cannot be combined with search inputs")
    if skus is not None and not 1 <= len(skus) <= 20:
        raise ValueError("skus must contain 1–20 items")
    started = _clock()
    call_id = uuid.uuid4().hex[:12]
    diagnostics = _ReviewDiagnostics()
    durations: dict[str, float] = {}
    counts: dict[str, int | float | None] = {
        "products": None,
        "review_groups": None,
        "reviews_received": None,
        "reviews_matched": None,
        "reviews_returned": None,
        "reviews_discarded_other_sku": None,
        "reviews_without_sku": None,
        "review_requests": None,
        "review_pages": None,
        "review_wait": None,
        "review_read": None,
        "incomplete": None,
    }
    try:
        result = await _compare(
            query,
            limit,
            reviews_limit,
            sort,
            skus,
            category,
            filters,
            durations=durations,
            diagnostics=diagnostics,
        )
        with _phase("serialize", durations):
            COMPARE_PRODUCTS_RESPONSE_SIZE.observe(len(result.model_dump_json().encode("utf-8")))
            review_counts = _review_counts(result, diagnostics)
            physical = [probe.snapshot() for _, probe, _ in diagnostics.samples]
            counts["reviews_received"] = sum(item.raw_reviews for item in physical)
            counts["reviews_matched"] = review_counts["matched"]
            counts["reviews_returned"] = review_counts["returned"]
            counts["reviews_discarded_other_sku"] = review_counts["discarded_other_sku"]
            counts["reviews_without_sku"] = review_counts["without_sku"]
            counts["review_requests"] = sum(item.requests for item in physical)
            counts["review_pages"] = sum(item.pages for item in physical)
            counts["review_wait"] = sum(item.wait_seconds for item in physical)
            counts["review_read"] = sum(item.read_seconds for item in physical)
            counts["products"] = len(result.products)
            counts["review_groups"] = len(result.review_groups)

            counts["incomplete"] = (
                sum(
                    bool(product.errors)
                    or any(
                        not coverage.sufficient
                        for group in result.review_groups
                        if group.sku == product.sku
                        for coverage in group.coverage
                        if coverage.sort == "useful"
                    )
                    for product in result.products
                )
                / len(result.products)
                if result.products
                else 0
            )
            COMPARE_PRODUCTS_PRODUCT_COUNT.observe(counts["products"])
            COMPARE_PRODUCTS_REVIEW_GROUP_COUNT.observe(counts["review_groups"])
            COMPARE_PRODUCTS_REVIEWS_RECEIVED.observe(counts["reviews_received"])
            COMPARE_PRODUCTS_REVIEWS_MATCHED.observe(counts["reviews_matched"])
            COMPARE_PRODUCTS_REVIEWS_RETURNED.observe(counts["reviews_returned"])
            COMPARE_PRODUCTS_REVIEWS_DISCARDED_OTHER_SKU.observe(counts["reviews_discarded_other_sku"])
            COMPARE_PRODUCTS_REVIEWS_WITHOUT_SKU.observe(counts["reviews_without_sku"])
            COMPARE_PRODUCTS_REVIEW_UPSTREAM_REQUESTS.observe(counts["review_requests"])
            COMPARE_PRODUCTS_REVIEW_PAGES_SCANNED.observe(counts["review_pages"])
            COMPARE_PRODUCTS_REVIEW_SEMAPHORE_WAIT.observe(counts["review_wait"])
            COMPARE_PRODUCTS_REVIEW_READ.observe(counts["review_read"])
            COMPARE_PRODUCTS_INCOMPLETE_FRACTION.observe(counts["incomplete"])
        return result
    finally:
        elapsed = _clock() - started
        COMPARE_PRODUCTS_DURATION.observe(elapsed)
        logger.info(
            "tool=compare_products call_id=%s total=%.3f search=%s snapshot=%s "
            "details_delivery=%s reviews=%s serialize=%s products=%s review_groups=%s "
            "reviews_received=%s reviews_matched=%s reviews_returned=%s "
            "reviews_discarded_other_sku=%s reviews_without_sku=%s review_requests=%s "
            "review_pages=%s review_wait=%s review_read=%s incomplete=%s",
            call_id,
            elapsed,
            *(
                f"{durations[name]:.3f}" if name in durations else "n/a"
                for name in ("search", "snapshot", "details_delivery", "reviews", "serialize")
            ),
            counts["products"],
            counts["review_groups"],
            counts["reviews_received"],
            counts["reviews_matched"],
            counts["reviews_returned"],
            counts["reviews_discarded_other_sku"],
            counts["reviews_without_sku"],
            counts["review_requests"],
            counts["review_pages"],
            counts["review_wait"],
            counts["review_read"],
            counts["incomplete"],
        )
