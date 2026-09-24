"""Bounded, read-only comparison built on the catalog service, not on MCP calls."""

import asyncio
import logging
from collections.abc import Callable

from ozon_mcp.dependencies import run_blocking
from ozon_mcp.models.catalog import (
    ComparedProduct,
    ComparisonDelivery,
    ComparisonReview,
    ComparisonReviews,
    ProductCard,
    ProductComparison,
    Tile,
)
from ozon_mcp.services import catalog
from ozon_mcp.session.transport import ReadSnapshot
from ozon_mcp.utils.observability import COMPARISON_FALLBACKS

logger = logging.getLogger("ozon_mcp")
_MAX_CONCURRENT = 6
_MAX_CHARACTERISTICS = 12


def _review_key(card: ProductCard | None, sku: str) -> frozenset[str]:
    """Only identical, explicit variant sets qualify as the same card."""
    if card is None:
        return frozenset({sku})
    options = {option.sku for group in card.variants for option in group.options if option.sku}
    return frozenset(options | {sku})


async def _snapshot() -> ReadSnapshot | None:
    """Prepare an isolated read client on the session thread after search."""
    try:
        return await run_blocking(lambda: catalog.get_session().snapshot_reads())
    except Exception:  # Serial reads remain available if snapshotting fails.
        COMPARISON_FALLBACKS.labels(reason="snapshot").inc()
        logger.warning("comparison snapshot failed; using serial reads", exc_info=True)
        return None


async def _read[T](
    semaphore: asyncio.Semaphore,
    snapshot: ReadSnapshot | None,
    parallel: Callable[[ReadSnapshot], T],
    serial: Callable[[], T],
) -> tuple[T | None, Exception | None]:
    async with semaphore:
        if snapshot is not None:
            try:
                return await asyncio.to_thread(parallel, snapshot), None
            except Exception:  # Retry a failed isolated read with the original session.
                COMPARISON_FALLBACKS.labels(reason="read").inc()
                logger.warning("comparison parallel read failed; retrying on the session thread", exc_info=True)
        try:
            return await run_blocking(serial), None
        except Exception as exc:  # ruff: ignore[blind-except] - one source must not discard other products
            return None, exc


def _apply_card(product: ComparedProduct, card: ProductCard | None) -> None:
    if card is None:
        return
    product.title = card.title or product.title
    product.price = card.price or product.price
    product.price_regular = card.price_regular or product.price_regular
    product.available = card.available
    product.characteristics = card.characteristics[:_MAX_CHARACTERISTICS]
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


async def compare_products(
    query: str, limit: int = 10, reviews_limit: int = 10, sort: str = "popular"
) -> ProductComparison:
    """Compare search hits, retaining independent failures and shared review pools."""
    tiles = await run_blocking(lambda: catalog.search(query=query, sort=sort, limit=limit))
    products = _select_products(tiles, limit)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    snapshot = await _snapshot() if products else None

    cards_and_delivery = await asyncio.gather(
        *(
            asyncio.gather(
                _read(
                    semaphore,
                    snapshot,
                    lambda client, sku=product.sku: catalog.product_details(sku, session=client),
                    lambda sku=product.sku: catalog.product_details(sku),
                ),
                _read(
                    semaphore,
                    snapshot,
                    lambda client, sku=product.sku: catalog.delivery_estimate(sku, session=client),
                    lambda sku=product.sku: catalog.delivery_estimate(sku),
                ),
            )
            for product in products
        )
    )
    cards: list[ProductCard | None] = []
    for product, ((card, details_error), (delivery, delivery_error)) in zip(products, cards_and_delivery, strict=True):
        cards.append(card)
        if details_error:
            product.errors.append(f"details: {details_error}")
        _apply_card(product, card)
        if delivery_error:
            product.errors.append(f"delivery: {delivery_error}")
        if delivery:
            product.delivery = ComparisonDelivery(
                delivery=delivery.delivery, address=delivery.address, source=delivery.source
            )

    # Identical explicit variant sets identify one shared card. Never infer a
    # match from equal ratings, names, or a review's human-readable variant.
    groups: dict[frozenset[str], list[ComparedProduct]] = {}
    for product, card in zip(products, cards, strict=True):
        key = _review_key(card, product.sku)
        groups.setdefault(key, []).append(product)
    representatives = [members[0].sku for members in groups.values()]
    results = await asyncio.gather(
        *(
            _read(
                semaphore,
                snapshot,
                lambda client, sku=sku: catalog.get_reviews(sku, limit=reviews_limit, session=client),
                lambda sku=sku: catalog.get_reviews(sku, limit=reviews_limit),
            )
            for sku in representatives
        )
    )
    review_groups: list[ComparisonReviews] = []
    for members, sku, (reviews, error) in zip(groups.values(), representatives, results, strict=True):
        member_skus = {product.sku for product in members}
        for product in members:
            product.reviews_ref = sku
            if error:
                product.errors.append(f"reviews: {error}")
        review_groups.append(
            ComparisonReviews(
                sku=sku,
                rating=reviews.score if reviews else None,
                count=reviews.count if reviews else None,
                reviews=[
                    ComparisonReview(
                        sku=review.sku,
                        variant=review.variant,
                        score=review.score,
                        positive=review.positive,
                        negative=review.negative,
                        text=review.text,
                    )
                    for review in (reviews.reviews if reviews else [])
                    if (review.text or review.positive or review.negative)
                    and (review.sku is None or review.sku in member_skus)
                ],
                error=str(error) if error else None,
            )
        )
    return ProductComparison(query=query, products=products, review_groups=review_groups)
