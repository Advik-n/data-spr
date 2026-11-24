"""Policy-derived normalized-price fixtures, independent of competition ID labels."""
from decimal import Decimal

import pytest

from procurement.baseline import compile_policy, evaluate

BOOK = compile_policy()


@pytest.mark.parametrize(("message", "quantity", "price"), [
    ("₹15,900 per 100 shipments.", 20, "159"),
    ("Rs. 1,884 per dozen shipments.", 12, "157"),
    ("₹620 in total for all 4 shipments.", 4, "155"),
    ("Between ₹159 and ₹155 per shipment.", 1, "159"),
    ("₹176 per shipment, plus 1 free shipment for every 10 you pay for.", 10, "160"),
    ("₹200 per shipment, with a further 25% off for orders of 10 shipments or more.", 10, "150"),
    ("₹200 per shipment, with a further 25% off for orders of 10 shipments or more.", 9, "200"),
    ("A 25% discount on our list price of ₹200 per shipment.", 10, "150"),
    ("₹177 including GST per shipment.", 1, "150"),
    ("₹150 excl. GST per shipment.", 1, "150"),
    ("One hundred fifty-nine rupees per shipment.".lower(), 1, "159"),
    ("₹0.159k per shipment.", 1, "159"),
    ("₹0.00159 lakh per shipment.", 1, "159"),
])
def test_normalized_offer(message, quantity, price):
    row = dict(thread_id="fixture", supplier="Redwood Couriers", category="courier",
               date="2026-09-01", quantity=str(quantity), round="1", supplier_message=message)
    trace = evaluate(row, BOOK)
    assert trace["price"] == Decimal(price)
    assert trace["label"] == ("ACCEPT:ACCEPTABLE_PRICE:" if Decimal(price) <= 160 else "COUNTER:COUNTER_OFFER:160")
