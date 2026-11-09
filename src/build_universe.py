"""§4 — The filing universe.

Reads EDGAR's quarterly form index (via edgartools) for every quarter from 2011 Q1 through
2023 Q4, keeps 10-K and 10-K/A rows, and writes data/interim/tenk_index.parquet with columns
(cik, company, form, filing_date, accession_number). One request per quarter.

Run: uv run python -m src.build_universe
"""

import csv
from pathlib import Path

import src.sec  # noqa: F401  (must precede any edgar import)
import pandas as pd
from edgar import get_filings

OUT = Path("data/interim/tenk_index.parquet")
FAILURES = Path("data/interim/universe_failures.csv")
COLS = ["cik", "company", "form", "filing_date", "accession_number"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames, failures = [], []
    for y in range(2011, 2024):
        for q in range(1, 5):
            try:
                df = get_filings(year=y, quarter=q, form=["10-K", "10-K/A"]).to_pandas()
                frames.append(df[COLS])
                print(f"{y}Q{q}: {len(df):,} rows", flush=True)
            except Exception as e:  # keep going; log and report
                failures.append({"year": y, "quarter": q, "error": repr(e)})
                print(f"{y}Q{q}: FAILED {e!r}", flush=True)

    if failures:
        with FAILURES.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["year", "quarter", "error"])
            w.writeheader()
            w.writerows(failures)
        raise SystemExit(f"{len(failures)} quarters failed; see {FAILURES}. Re-run before using the index.")

    idx = pd.concat(frames, ignore_index=True)
    idx["cik"] = idx["cik"].astype(int).astype(str)  # string, no leading zeros
    idx["filing_date"] = pd.to_datetime(idx["filing_date"])
    idx = idx[idx["form"].isin(["10-K", "10-K/A"])]
    idx.to_parquet(OUT, index=False)

    print(f"\nWrote {len(idx):,} rows to {OUT}")
    print("\nRows by form and filing year:")
    print(pd.crosstab(idx["filing_date"].dt.year, idx["form"]))

    # Checks against the walkthrough's 2016 numbers
    y16 = idx[idx["filing_date"].dt.year == 2016]
    k = y16[y16["form"] == "10-K"]
    print("\n2016 check (expected: 7,589 / 7,259 / 1,175 / 7,373 / 96)")
    print(f"  10-K rows:                {len(k):,}")
    print(f"  10-K distinct accessions: {k['accession_number'].nunique():,}")
    print(f"  10-K/A rows:              {(y16['form'] == '10-K/A').sum():,}")
    print(f"  distinct CIKs with 10-K:  {k['cik'].nunique():,}")
    print(f"  CIKs with 2+ 10-Ks:       {(k.groupby('cik')['accession_number'].nunique() >= 2).sum():,}")


if __name__ == "__main__":
    main()
