import pandas as pd
import pytest

from src.fiscal import compustat_fyear, compustat_fyear_series

CASES = [
    ("2016-01-02", 2015),
    ("2011-01-01", 2010),
    ("2014-02-01", 2013),
    ("2016-06-04", 2015),
    ("2016-06-30", 2016),
]


@pytest.mark.parametrize("date,expected", CASES)
def test_compustat_fyear(date, expected):
    assert compustat_fyear(date) == expected


def test_compustat_fyear_series_matches_scalar():
    s = pd.Series([d for d, _ in CASES])
    assert list(compustat_fyear_series(s)) == [e for _, e in CASES]


from src.fiscal import period_to_fyear  # noqa: E402


def tenks(*dates):
    return pd.DataFrame({"report_date": pd.to_datetime(list(dates))})


DEC = tenks("2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31")
JAN = tenks("2012-01-28", "2013-02-02", "2014-02-01", "2015-01-31", "2016-01-30")


@pytest.mark.parametrize(
    "company,period_end,expected",
    [
        (DEC, "2014-03-31", 2014),
        (DEC, "2013-12-31", 2013),
        (JAN, "2014-02-01", 2013),
        (JAN, "2014-05-03", 2014),
        (DEC, "2009-12-31", 2009),  # first 10-K is for 2011-12-31: roll forward
        (tenks("2014-12-31", "2016-12-31"), "2015-03-31", 2015),  # no 10-K for 2015
    ],
)
def test_period_to_fyear(company, period_end, expected):
    assert period_to_fyear(period_end, company) == expected


def test_period_to_fyear_bad_input():
    assert period_to_fyear(None, DEC) is None
    assert period_to_fyear("2014-03-31", tenks()) is None
