import datetime as dt

import pytest

from dates import resolve_date

TODAY = dt.date(2026, 10, 2)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("i want an appointment for tuesday", dt.date(2026, 10, 6)),
        ("next tuesday", dt.date(2026, 10, 6)),
        ("10 am on oct 6th", dt.date(2026, 10, 6)),
        ("what about the 15th", dt.date(2026, 10, 15)),
        ("no 6th", dt.date(2026, 10, 6)),
        ("the 1st", dt.date(2026, 11, 1)),
        ("tomorrow", dt.date(2026, 10, 3)),
        ("january 5", dt.date(2027, 1, 5)),
        ("2026-10-20", dt.date(2026, 10, 20)),
        ("10 am", None),
        ("may i book something", None),
        ("yes", None),
    ],
)
def test_resolve_date(text, expected):
    assert resolve_date(text, TODAY) == expected