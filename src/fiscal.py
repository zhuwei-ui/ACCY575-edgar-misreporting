"""Fiscal-year rules shared by every script.

`compustat_fyear` (§5) maps a fiscal year-END date to a Compustat-style fiscal year.
`period_to_fyear` (§8a) is added later and maps any period-end date (quarter or year) to the
fiscal year of the company's 10-K that contains it.
"""

from datetime import timedelta

import pandas as pd


def compustat_fyear(report_date) -> int:
    """Compustat convention with a 7-day shift for 52/53-week years.

    d = report_date - 7 days; fyear = d.year if d.month >= 6 else d.year - 1.
    """
    d = pd.Timestamp(report_date) - timedelta(days=7)
    return d.year if d.month >= 6 else d.year - 1


def compustat_fyear_series(dates: pd.Series) -> pd.Series:
    """Vectorized compustat_fyear; NaT stays missing."""
    d = pd.to_datetime(dates) - pd.Timedelta(days=7)
    return (d.dt.year - (d.dt.month < 6).astype("Int64")).astype("Int64")


def period_to_fyear(period_end, company_10ks: pd.DataFrame) -> int | None:
    """§8a — map any period-end date (quarter or year) to the company's fiscal year.

    `company_10ks`: EVERY original 10-K the company filed (from filings.parquet, not only the
    2011+ panel rows), with a `report_date` column (`fyear` is computed with §5's rule if absent).

    Rule: take the 10-K with the earliest report_date on or after period_end - 10 days, provided
    that report_date is no more than 380 days after period_end, and return its fyear. Otherwise
    roll forward to the next year-end on or after period_end - 10 days in the fiscal year-end month
    of the company's nearest 10-K, and apply §5's rule. Returns None if the date is unusable or
    the company has no 10-K with a report_date.
    """
    try:
        pe = pd.Timestamp(period_end)
    except (ValueError, TypeError):
        return None
    if pd.isna(pe):
        return None
    k = company_10ks.dropna(subset=["report_date"])
    if k.empty:
        return None
    rd = pd.to_datetime(k["report_date"])
    fy = k["fyear"] if "fyear" in k else compustat_fyear_series(rd)
    floor = pe - timedelta(days=10)
    after = rd >= floor
    if after.any():
        i = rd[after].idxmin()
        if (rd[i] - pe).days <= 380:
            return int(fy[i])
    # roll forward in the fiscal year-end month of the nearest 10-K
    nearest = rd[(rd - pe).abs().idxmin()]
    fye_month = (nearest - timedelta(days=7)).month
    ye = pd.Timestamp(year=floor.year, month=fye_month, day=1) + pd.offsets.MonthEnd(0)
    if ye < floor:
        ye = pd.Timestamp(year=floor.year + 1, month=fye_month, day=1) + pd.offsets.MonthEnd(0)
    return compustat_fyear(ye)


def load_company_10ks(path: str = "data/interim/filings.parquet") -> dict[str, pd.DataFrame]:
    """Every original 10-K per CIK, with report_date and §5 fyear, for period_to_fyear."""
    f = pd.read_parquet(path, columns=["cik", "form", "reportDate", "filing_date", "accession_number"])
    k = f[(f["form"] == "10-K") & f["reportDate"].notna()].rename(columns={"reportDate": "report_date"})
    k = k.assign(fyear=compustat_fyear_series(k["report_date"]))
    return {cik: g.reset_index(drop=True) for cik, g in k.groupby("cik")}
