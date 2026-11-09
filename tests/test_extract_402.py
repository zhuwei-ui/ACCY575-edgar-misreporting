import pandas as pd

from src.extract_402 import keep_filed_fyears, regex_extract


def tenks(rows):
    return pd.DataFrame(rows, columns=["report_date", "filing_date", "fyear"]).astype(
        {"report_date": "datetime64[ns]", "filing_date": "datetime64[ns]"})


K = tenks([("2013-12-31", "2014-03-01", 2013), ("2014-12-31", "2015-03-01", 2014)])


def test_keep_filed_fyears_drops_years_whose_10k_came_later():
    # 8-K on 2014-11-10 about FY2013 and Q1-Q3 2014: FY2014's 10-K not filed yet
    assert keep_filed_fyears([2013, 2014], K, "2014-11-10") == [2013]
    assert keep_filed_fyears([2013, 2014], K, "2015-04-01") == [2013, 2014]
    assert keep_filed_fyears([2012], K, "2015-04-01") == []  # no 10-K for 2012 at all


def test_regex_extract_apple_style_text():
    k = tenks([(f"{y}-09-30", f"{y}-12-01", y) for y in range(2002, 2008)])
    text = ("financial statements for the fiscal years ended 2003, 2004 and 2005 and periods "
            "commencing on September 29, 2002 should no longer be relied upon; an investigation")
    out = regex_extract(text, k)
    assert out["restated_fyears"] == [2003, 2004, 2005]
    assert out["irregularity"] is True
