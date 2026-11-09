"""§8a — draw 30 events at random (seed 575) for hand labeling; dump their first-filing sections
with the company's 10-K report dates to data/interim/gold_review.txt for reading."""

from pathlib import Path

import pandas as pd

from src.fiscal import load_company_10ks

ev = pd.read_parquet("data/interim/events_402.parquet")
s = ev.sample(30, random_state=575).sort_values("event_id")
s[["event_id", "cik", "first_filing_date", "n_filings"]].to_csv("data/interim/gold_sample.csv", index=False)
tenks = load_company_10ks()
out = []
for e in s.itertuples():
    p = Path(f"data/raw/402_sections/{e.accessions[0]}.txt")
    k = tenks.get(e.cik)
    rds = ", ".join(f"{d:%Y-%m-%d}→{fy}" for d, fy in zip(k["report_date"], k["fyear"])) if k is not None else "none"
    out.append(f"#### {e.event_id}  cik={e.cik}  first={e.first_filing_date:%Y-%m-%d}  filings={e.n_filings}  "
               f"acc={e.accessions[0]}\n10-Ks (report_date→fyear): {rds}\n\n"
               + (p.read_text(encoding='utf-8') if p.exists() else "[SECTION NOT YET DOWNLOADED]") + "\n")
Path("data/interim/gold_review.txt").write_text("\n".join(out), encoding="utf-8")
print(s[["event_id", "first_filing_date", "n_filings"]].to_string(index=False))
