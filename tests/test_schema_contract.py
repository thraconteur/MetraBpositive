from src.core.schema import (
    BBox,
    Declaration,
    Finding,
    Outcome,
    PresenceState,
    ScanResult,
    Severity,
)


def test_core_schema_exports_expected_contract():
    assert Outcome.VIOLATION.value == "VIOLATION"
    assert PresenceState.CONFIRMED_ABSENT.value == "CONFIRMED_ABSENT"
    assert Severity.CRITICAL.value == "CRITICAL"
    assert Severity("major") is Severity.MAJOR
    assert Severity("MAJOR") is Severity.MAJOR

    b = BBox(10, 20, 30, 40)
    assert b.x2 == 40
    assert b.y2 == 60
    assert b.cx == 25.0
    assert b.cy == 40.0

    d = Declaration(field_id="retail_sale_price", raw_text="Rs. 120", present=True)
    assert d.field_id == "retail_sale_price"

    f = Finding(rule_id="retail_sale_price.required_phrasing", outcome=Outcome.VIOLATION)
    assert f.outcome is Outcome.VIOLATION

    res = ScanResult()
    assert res.declaration("missing_field") is None
