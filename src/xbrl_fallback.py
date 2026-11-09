"""§6 step 8 — fill financials from a filing's own XBRL for 10-Ks missing from the FSDS.

Targets: panel 10-Ks with isXBRL == 1 in filings.parquet whose accession appears in no quarter's
sub.txt (under any CIK). Combined-filing subsidiaries whose parent has FSDS numbers are *not*
targets: their numbers belong to the parent, as in §6 step 1.

For each target, Filing(...).xbrl() facts are filtered to: us-gaap concepts, USD units, no
dimensions; instants within 7 days of report_date (current) or report_date - 1 year (lag);
durations of 350-380 days ending within 7 days of those dates. A filing's own XBRL is as first
reported by construction. Filtered facts are cached per accession in data/raw/xbrl_facts/ so the
step is resumable; failures go to data/interim/xbrl_failures.csv.
"""

import csv
import re
from pathlib import Path

import src.sec  # noqa: F401  (must precede any edgar import)
import pandas as pd
from edgar import Filing

from src.build_financials import ALL_TAGS, compute_vars

FILINGS = Path("data/interim/filings.parquet")
CACHE = Path("data/raw/xbrl_facts")
FAILURES = Path("data/interim/xbrl_failures.csv")
USD = re.compile(r"^(u_)?(iso4217[_:]?)?usd$", re.I)
TOL = pd.Timedelta(days=7)


def targets(panel: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_parquet(FILINGS, columns=["cik", "accession_number", "isXBRL"])
    m = panel.merge(f, left_on=["cik", "accession"], right_on=["cik", "accession_number"], how="left")
    return m[(m["isXBRL"] == 1) & ~m["accession"].isin(sub["adsh"])].reset_index(drop=True)


def filter_facts(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    df = df[df["concept"].str.startswith("us-gaap:") & ~df["is_dimensioned"].astype(bool)
            & df["unit_ref"].fillna("").str.match(USD) & df["numeric_value"].notna()].copy()
    df["tag"] = df["concept"].str.removeprefix("us-gaap:")
    df = df[df["tag"].isin(ALL_TAGS)]
    lag_date = report_date - pd.DateOffset(years=1)
    inst = pd.to_datetime(df.get("period_instant"), errors="coerce")
    start = pd.to_datetime(df.get("period_start"), errors="coerce")
    end = pd.to_datetime(df.get("period_end"), errors="coerce")
    is_inst = df["period_type"] == "instant"
    full_year = (end - start).dt.days.between(350, 380)
    d = inst.where(is_inst, end)
    ok = is_inst | full_year
    cur = ok & ((d - report_date).abs() <= TOL)
    lag = ok & ((d - lag_date).abs() <= TOL)
    df["which"] = None
    df.loc[cur, "which"] = "cur"
    df.loc[lag & ~cur, "which"] = "lag"
    df["qtrs"] = is_inst.map({True: "0", False: "4"})
    df = df[df["which"].notna()]
    return df[["tag", "which", "qtrs", "numeric_value"]].rename(columns={"numeric_value": "value"})


def fetch(row) -> pd.DataFrame:
    path = CACHE / f"{row.accession}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fl = Filing(cik=int(row.cik), company="", form="10-K",
                filing_date=row.filing_date.strftime("%Y-%m-%d"), accession_no=row.accession)
    x = fl.xbrl()
    if x is None:
        raise ValueError("no XBRL returned")
    facts = filter_facts(x.facts.to_dataframe(), pd.Timestamp(row.report_date))
    facts["adsh"] = row.accession
    tmp = path.with_suffix(".tmp")
    facts.to_parquet(tmp, index=False)
    tmp.rename(path)
    return facts


def fill_from_xbrl(panel: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    t = targets(panel, sub)
    print(f"\nStep 8: {len(t):,} isXBRL 10-Ks missing from every FSDS quarter", flush=True)
    longs, fails = [], []
    for i, row in enumerate(t.itertuples(), 1):
        try:
            longs.append(fetch(row))
        except Exception as e:
            fails.append({"cik": row.cik, "accession": row.accession, "error": repr(e)})
        if i % 50 == 0 or i == len(t):
            print(f"  {i}/{len(t)} done, {len(fails)} failed", flush=True)
    if fails:
        with FAILURES.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cik", "accession", "error"])
            w.writeheader()
            w.writerows(fails)
        print(f"  failures logged to {FAILURES}")
    if not longs:
        return pd.DataFrame()
    long = pd.concat(longs, ignore_index=True)
    # duplicate facts (same tag/period reported twice): keep one; log real conflicts
    conflicts = long.groupby(["adsh", "tag", "which"])["value"].nunique()
    if (conflicts > 1).any():
        print(f"  {(conflicts > 1).sum()} (accession, tag, period) with conflicting XBRL values; kept the first")
    long = long.drop_duplicates(["adsh", "tag", "which"])
    vals = compute_vars(long).reset_index().rename(columns={"adsh": "accession"})
    has_any = vals.drop(columns="accession").select_dtypes("number").notna().any(axis=1)
    vals = vals[has_any]
    return t[["cik", "fyear", "accession"]].merge(vals, on="accession")
