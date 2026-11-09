"""§6d — compare at / sale / ni for random FSDS rows against each filing's own XBRL."""

import sys

import src.sec  # noqa: F401
import pandas as pd
from edgar import Filing

from src.xbrl_fallback import filter_facts
from src.build_financials import compute_vars

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
fin = pd.read_parquet("data/interim/financials.parquet")
panel = pd.read_parquet("data/interim/tenk_panel.parquet")
s = fin[fin.fin_source == "fsds"].sample(n, random_state=575).merge(panel, on=["cik", "fyear", "accession"])
for r in s.itertuples():
    x = Filing(cik=int(r.cik), company="", form="10-K", filing_date=r.filing_date.strftime("%Y-%m-%d"),
               accession_no=r.accession).xbrl()
    long = filter_facts(x.facts.to_dataframe(), r.report_date).assign(adsh=r.accession)
    v = compute_vars(long.drop_duplicates(["adsh", "tag", "which"])).iloc[0]
    for var in ["at", "sale", "ni"]:
        mine, theirs = getattr(r, var), v.get(var)
        ok = (pd.isna(mine) and pd.isna(theirs)) or mine == theirs
        print(f"{r.cik:>8} {r.fyear} {r.accession} {var:4} panel={mine:>18,.0f} xbrl={theirs:>18,.0f} {'OK' if ok else 'MISMATCH'}")
    print(f"   https://www.sec.gov/Archives/edgar/data/{int(r.cik)}/{r.accession.replace('-', '')}/{r.primary_document}")
