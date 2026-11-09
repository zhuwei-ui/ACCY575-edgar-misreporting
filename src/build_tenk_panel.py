"""§5 — The CIK-year 10-K panel.

1. For each CIK in data/interim/tenk_index.parquet, save Company(cik).get_filings().to_pandas()
   (plus `sic`) to data/raw/submissions/{cik}.parquet. Existing files are skipped, so the job is
   resumable. Failures go to data/interim/submissions_failures.csv.
2. Concatenate into data/interim/filings.parquet (all forms; reused in §7).
3. Drop SIC 6189 (asset-backed issuers).
4. Keep original 10-Ks, compute fyear with the Compustat rule (src.fiscal), keep 2011-2021.
5. Earliest-filed 10-K per (cik, fyear); the rest go to data/interim/tenk_duplicates.parquet.
6. n_amendments = 10-K/As for the same reportDate; combined = accession under >1 CIK;
   late_filer = filed more than 365 days after report_date.
7. Validate with src.schemas.tenk_panel_schema and write data/interim/tenk_panel.parquet.

Run: uv run python -u -m src.build_tenk_panel [--limit N] [--skip-download]
"""

import argparse
import csv
import time
from pathlib import Path

import src.sec  # noqa: F401  (must precede any edgar import)
import pandas as pd
from edgar import Company

from src.fiscal import compustat_fyear_series
from src.schemas import tenk_panel_schema

INDEX = Path("data/interim/tenk_index.parquet")
SUBS = Path("data/raw/submissions")
FAILURES = Path("data/interim/submissions_failures.csv")
FILINGS = Path("data/interim/filings.parquet")
DUPES = Path("data/interim/tenk_duplicates.parquet")
OUT = Path("data/interim/tenk_panel.parquet")


def download(ciks: list[str]) -> None:
    SUBS.mkdir(parents=True, exist_ok=True)
    todo = [c for c in ciks if not (SUBS / f"{c}.parquet").exists()]
    print(f"{len(ciks):,} CIKs; {len(ciks) - len(todo):,} already saved; {len(todo):,} to fetch", flush=True)
    new_file = not FAILURES.exists()
    t0 = time.time()
    with FAILURES.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["cik", "error", "time"])
        for i, cik in enumerate(todo, 1):
            try:
                c = Company(int(cik))
                df = c.get_filings().to_pandas()
                df["sic"] = str(c.data.sic) if c.data.sic is not None else None
                df["filing_date"] = pd.to_datetime(df["filing_date"])
                tmp = SUBS / f"{cik}.parquet.tmp"
                df.to_parquet(tmp, index=False)
                tmp.rename(SUBS / f"{cik}.parquet")
            except Exception as e:
                w.writerow([cik, repr(e), pd.Timestamp.now().isoformat()])
                fh.flush()
            if i % 250 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  {i:,}/{len(todo):,} fetched, {el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left", flush=True)


def build(ciks: list[str]) -> None:
    parts = []
    for cik in ciks:
        p = SUBS / f"{cik}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d.insert(0, "cik", cik)
            parts.append(d)
    f = pd.concat(parts, ignore_index=True)
    f["reportDate"] = pd.to_datetime(f["reportDate"].replace("", None), errors="coerce")
    f.to_parquet(FILINGS, index=False)
    print(f"\nfilings.parquet: {len(f):,} filings for {f['cik'].nunique():,} CIKs", flush=True)

    abs_ciks = f.loc[f["sic"] == "6189", "cik"].unique()
    print(f"Dropped {len(abs_ciks):,} asset-backed issuers (SIC 6189)")
    f = f[f["sic"] != "6189"]

    k = f[f["form"] == "10-K"].copy()
    no_rd = k["reportDate"].isna().sum()
    if no_rd:
        print(f"Dropped {no_rd:,} original 10-Ks with no reportDate")
    k = k[k["reportDate"].notna()]
    k["fyear"] = compustat_fyear_series(k["reportDate"])
    k = k[k["fyear"].between(2011, 2021)]

    k = k.sort_values(["cik", "fyear", "filing_date", "accession_number"])
    dup = k.duplicated(["cik", "fyear"], keep="first")
    k[dup].to_parquet(DUPES, index=False)
    print(f"Duplicate (cik, fyear) 10-Ks dropped (kept earliest filed): {dup.sum():,} -> {DUPES}")
    k = k[~dup]

    amend = (
        f[f["form"] == "10-K/A"].groupby(["cik", "reportDate"]).size().rename("n_amendments").reset_index()
    )
    k = k.merge(amend, on=["cik", "reportDate"], how="left")
    k["n_amendments"] = k["n_amendments"].fillna(0).astype(int)
    k["combined"] = k.groupby("accession_number")["cik"].transform("nunique") > 1

    panel = pd.DataFrame(
        {
            "cik": k["cik"].astype(str),
            "fyear": k["fyear"].astype(int),
            "report_date": k["reportDate"].astype("datetime64[ns]"),
            "filing_date": pd.to_datetime(k["filing_date"]).astype("datetime64[ns]"),
            "accession": k["accession_number"],
            "primary_document": k["primaryDocument"].replace("", None),
            "sic": k["sic"],
            "n_amendments": k["n_amendments"],
            "combined": k["combined"].astype(bool),
        }
    )
    panel["late_filer"] = (panel["filing_date"] - panel["report_date"]).dt.days > 365
    panel = tenk_panel_schema.validate(panel)
    panel.to_parquet(OUT, index=False)

    print(f"\nWrote {len(panel):,} CIK-years to {OUT}")
    print("\nRows per fyear:")
    print(panel["fyear"].value_counts().sort_index().to_string())
    print(f"\ncombined rows: {panel['combined'].sum():,}")
    print(f"late_filer rows: {panel['late_filer'].sum():,}")
    print(f"rows with n_amendments > 0: {(panel['n_amendments'] > 0).sum():,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N CIKs (testing)")
    ap.add_argument("--skip-download", action="store_true", help="build from files already saved")
    args = ap.parse_args()

    ciks = sorted(pd.read_parquet(INDEX, columns=["cik"])["cik"].unique(), key=int)
    if args.limit:
        ciks = ciks[: args.limit]
    if not args.skip_download:
        download(ciks)
    build(ciks)


if __name__ == "__main__":
    main()
