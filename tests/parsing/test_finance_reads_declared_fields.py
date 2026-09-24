"""The balance and the points come from the tiles that state them.

Both used to be found by regex — the balance as the first money-looking string
in the widget, the points through a 250-character window over every widget on
the page — so a second priced tile would have quietly replaced the answer.
"""

from __future__ import annotations

from ozon_mcp.parsing.finance import parse_finance


def _page() -> dict[str, object]:
    cashback = (
        '{"title": {"text": "Кэшбэк"}, "subtitle": {"text": "1 200 ₽"},'
        ' "trackingInfo": {"click": {"actionType": "ClickActionCardsCashback"}}}'
    )
    balance = (
        '{"title": {"text": "Ozon Карта"}, "subtitle": {"text": "415,64 ₽"},'
        ' "trackingInfo": {"click": {"actionType": "ClickActionCardsOzonBank_BankBalanceCard"},'
        ' "view": {"actionType": "ViewActionCardsOzonBank"}}}'
    )
    return {
        "widgetStates": {
            "actionCards-1": f'{{"cards": [{cashback}, {balance}]}}',
            "menu-2": (
                '{"sections": [{"title": "Личная информация", "items": ['
                '{"title": "Баллы за отзывы", "action": {"link": "/my/reviews/promo"}},'
                '{"title": "Баллы и бонусы", "action": {"link": "/my/points"},'
                ' "notification": {"type": "text", "text": {"text": "6 633"}}}]}]}'
            ),
        }
    }


def test_balance_comes_from_the_balance_tile() -> None:
    # The cashback tile is priced too, and Ozon puts it first.
    finances = parse_finance(_page())
    assert finances.ozon_card_balance == "415,64 ₽"
    assert finances.points == "6633"


def test_nothing_declared_reads_as_nothing() -> None:
    finances = parse_finance({"widgetStates": {"actionCards-1": '{"cards": []}', "menu-2": "{}"}})
    assert finances.ozon_card_balance is None
    assert finances.points is None
