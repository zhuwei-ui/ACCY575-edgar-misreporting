"""§7b — count Item 4.02 8-Ks in filings.parquet and cross-check against EDGAR full-text search."""

import src.sec  # noqa: F401
import pandas as pd
from edgar import search_filings

f = pd.read_parquet("data/interim/filings.parquet", columns=["cik", "form", "items", "filing_date"])
is_402 = f["form"].isin(["8-K", "8-K/A"]) & f["items"].fillna("").str.split(",").apply(lambda xs: "4.02" in xs)
mine = f[is_402].groupby(f["filing_date"].dt.year).size()

rows = []
for y in range(2011, 2026):
    try:
        r = search_filings(forms="8-K", items="4.02", start_date=f"{y}-01-01", end_date=f"{y}-12-31", limit=100)
        total = r.total
    except Exception as e:
        total = f"error: {e!r}"
    rows.append({"year": y, "panel_filter": int(mine.get(y, 0)), "full_text_search": total})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("data/interim/402_count_check.csv", index=False)
