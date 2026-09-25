"""The storefront: search, product cards, reviews, delivery estimates."""

from typing import Annotated

from pydantic import Field

from ozon_mcp.dependencies import run_blocking
from ozon_mcp.mcp_server import mcp
from ozon_mcp.models.catalog import (
    Cheaper,
    DeliveryEstimate,
    Description,
    ProductCard,
    ProductComparison,
    Reviews,
    SearchFilter,
    Tile,
)
from ozon_mcp.services import catalog, comparison
from ozon_mcp.utils.annotations import Limit, Page, ReviewSort, SearchSort, SkuOrUrl


@mcp.tool()
async def search(
    query: Annotated[str | None, Field(description="Search text. Give this, a category, or both.")] = None,
    category: Annotated[
        str | None,
        Field(description='A category slug from a category URL, e.g. "produkty-dlya-doma-9200".'),
    ] = None,
    sort: SearchSort = "popular",
    page: Page = 1,
    filters: Annotated[
        dict[str, str] | None,
        Field(
            description="Facets from get_search_filters(): {key: option_value} for a checkbox or category, "
            '{key: "min;max"} for a range, e.g. {"currency_price": "200;600"}.'
        ),
    ] = None,
    limit: Limit = 36,
) -> list[Tile]:
    """Storefront search → product tiles (sku/title/price/url). Give a text
    query, a category slug ("produkty-dlya-doma-9200"), or both.
    limit is depth, not page size: pages are walked until there are that many
    results, so raise it when looking for the cheapest — a page holds a few
    dozen and the cheapest lot is often further down. sort="cheap" ranks on the
    payable price (Ozon's own order goes by a different figure).
    Ozon's text search is literal about words: a lot whose own title omits the
    brand does not come back for a query that includes it, so search the model
    ("Basilisk V3 X HyperSpeed") rather than brand-plus-model, and confirm what a
    lot actually is with product_details() — a tile's title is the seller's
    wording and may name no model at all.
    A tile is not a card: for variants, characteristics, photos and stock call
    product_details() with its sku. For comparing several products, prefer
    compare_products(query) instead of search() plus per-hit reads.
    filters comes from get_search_filters() — {key: option_value} for a
    checkbox or category facet, {key: "min;max"} for a range, e.g.
    {"currency_price": "200;600"}. Narrowing by price is a filter, not a sort.
    """
    return await run_blocking(lambda: catalog.search(query, category, sort, page, filters, limit))


@mcp.tool()
async def compare_products(
    query: Annotated[str | None, Field(description="Search phrase; omit when skus are supplied.")] = None,
    limit: Annotated[int, Field(ge=1, le=20, description="Maximum search hits to compare (1–20).")] = 10,
    reviews_limit: Annotated[
        int, Field(ge=1, le=20, description="Target meaningful useful reviews per SKU (1–20); may be incomplete.")
    ] = 10,
    sort: SearchSort = "popular",
    skus: Annotated[
        list[str] | None, Field(min_length=1, max_length=20, description="Selected SKUs; exclusive with search inputs.")
    ] = None,
    category: Annotated[str | None, Field(description="Category slug for search; not available with skus.")] = None,
    filters: Annotated[
        dict[str, str] | None, Field(description="Search facets as in search(filters=...); not with skus.")
    ] = None,
) -> ProductComparison:
    """Prefer this for comparing several products: search with query/category,
    or pass selected skus to enrich finalists without searching again. Provide
    exactly one input mode. Search facets match search(filters=...). One read-only
    call gathers prices, characteristics, delivery and reviews for the hits.
    Reuse its results; do not fetch the same data again
    with search(), product_details(), delivery_estimate() or get_reviews().
    Review samples are per SKU; coverage reports scanned, attributable and
    meaningful counts for useful and worst sorts. Unattributed reviews are not
    evidence about a specific SKU. Review sku is only set when Ozon supplies itemId.
    Individual failures are listed in each product's errors. Use individual
    tools only for missing fields, contradictory results or deeper finalist
    checks, not as a second pass over every hit.
    """
    if skus is not None and (query is not None or category is not None or filters is not None):
        raise ValueError("skus cannot be combined with search inputs")
    if skus is None and not (query or category):
        raise ValueError("provide query/category or skus")
    return await comparison.compare_products(
        query, limit, reviews_limit, sort, skus=skus, category=category, filters=filters
    )


@mcp.tool()
async def get_search_filters(
    query: Annotated[str, Field(description="The same search text whose facets you want.")],
) -> list[SearchFilter]:
    """Facets available for a query: {name, key, type, options|range}. Apply them
    with search(filters={key: value}) — checkbox/category value = option.value,
    range value = "min;max". Flow: search → get_search_filters → search(filters).
    """
    return await run_blocking(lambda: catalog.get_search_filters(query))


@mcp.tool()
async def product_details(
    sku_or_url: SkuOrUrl,
    with_description: Annotated[
        bool,
        Field(description="Also fetch the description text and its images (one extra request)."),
    ] = False,
    with_reviews: Annotated[
        bool,
        Field(description="Also fetch the reviews (one extra request)."),
    ] = False,
) -> ProductCard:
    """Product card: title, three prices, variants, characteristics, gallery photos.
    Money as Ozon prints it: `price` is «С банками» — what the account actually
    pays and the figure to compare lots on — `price_regular` is «С другими
    банками», `price_old` the struck-through comparison. `available` says whether
    it is on sale at all. `cheaper_offers` / `cheaper_from` are Ozon's own count
    and lowest price for other offers of this product; find_cheaper() lists them.
    Each variant carries its own sku, price and availability — that sku is what
    goes into the cart, and for apparel it is the only one that will add.
    with_description / with_reviews fetch those too (separate requests each);
    get_description() and get_reviews() do the same on their own. When comparing
    several products, start with compare_products(); read individual cards only
    for gaps, contradictions or finalists, not for data already returned.
    """
    return await run_blocking(
        lambda: catalog.product_details(sku_or_url, with_description=with_description, with_reviews=with_reviews)
    )


@mcp.tool()
async def get_reviews(sku_or_url: SkuOrUrl, limit: Limit = 30, sort: ReviewSort = "useful") -> Reviews:
    """Rating and reviews: `score` (e.g. 4.9), `count` — Ozon's own total — the
    `distribution` per star, and `limit` reviews with `fetched` saying how many
    came back. count and fetched are different numbers: reporting the second as
    the first turns 155 847 reviews into 30.
    Reviews arrive thirty at a time and pages are walked to meet `limit`. There
    is no filter by star, so the one-star ones are reached with sort="worst" plus
    enough depth.
    Each review keeps «Достоинства» and «Недостатки» apart from the comment,
    carries its useful votes, and names the variant it is about — a card's
    reviews cover its sizes and colours, so some are about a different one.
    For several products, start with compare_products(); fetch separate reviews
    only for gaps, contradictions or finalists needing more depth.
    """
    return await run_blocking(lambda: catalog.get_reviews(sku_or_url, limit, sort))


@mcp.tool()
async def get_description(sku_or_url: SkuOrUrl) -> Description:
    """Product description text plus the images embedded in it."""
    return await run_blocking(lambda: catalog.get_description(sku_or_url))


@mcp.tool()
async def delivery_estimate(sku_or_url: SkuOrUrl) -> DeliveryEstimate:
    """When a product would arrive, to which of the account's addresses, and
    from which warehouse ("Завтра, 2 сентября" / "ул. Данилова, 17" / "Со
    склада Ozon"). The date is relative to that address, so quote both.
    This is per product, before ordering; for an existing order the dates are
    in list_orders(), and for an order being formed in get_checkout(). For
    several products, start with compare_products(); fetch a separate estimate
    only for a gap, contradiction or finalist, not for every compared hit.
    """
    return await run_blocking(lambda: catalog.delivery_estimate(sku_or_url))


@mcp.tool()
async def find_cheaper(sku_or_url: SkuOrUrl, limit: Limit = 10) -> Cheaper:
    """The cheapest lots of the same thing, ranked by payable price — top `limit`
    (default 10).
    Looks in both places Ozon keeps them: its own «Есть дешевле или быстрее»
    offers for this exact product, and a price-sorted search by the card's
    title, which reaches the same product listed separately. Entries carry
    `seller` and `delivery` when they came from the offers list, and offers have
    no title — confirm the model with product_details() before quoting one.
    Raises when the base price cannot be read, rather than answering "nothing is
    cheaper" for a product it failed to price.
    """
    return await run_blocking(lambda: catalog.find_cheaper(sku_or_url, limit))
