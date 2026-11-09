"""§7c — Item 4.02 filings -> events, and the Item 4.02 section text for every filing.

1. From data/interim/filings.parquet keep 8-K / 8-K/A whose `items` include 4.02, filed between
   2011-01-01 and the download date, for CIKs in tenk_panel.
2. Events per CIK, in filing-date order: an 8-K/A filed within 365 days of the start of the
   company's most recent event joins it; anything else (an original 8-K, or an 8-K/A that joins
   nothing) starts a new event.
3. For every filing: Filing(...).obj()["Item 4.02"], cleaned (step 4), saved to
   data/raw/402_sections/{accession}.txt. Existing files are skipped; failures are logged.
4. Fixes, logged to data/interim/402_section_fixes.csv:
   - "fallback": section None/empty, or only the heading came back (body < 50 chars) -> search
     the filing's full text (.obj().text()) for the 4.02 Non-Reliance title and take what follows.
   - "not_found": the fallback found no title either; an empty file is saved.
   - "cut": the section ran on into a later item heading (start of a line) or the SIGNATURE block.
   - "stacked_heading": the 4.02 heading was immediately followed by another item heading (e.g.
     "Item 8.01 Other Events.") with the disclosure underneath; the text under it is kept, up to
     the next heading that follows real body text. Review these when labeling.

Writes data/interim/events_402.parquet.
Run: uv run python -u -m src.build_402_events [--limit N]
"""

import argparse
import csv
import re
import time
import warnings
from pathlib import Path

import src.sec  # noqa: F401  (must precede any edgar import)
import pandas as pd
from edgar import Filing

warnings.filterwarnings("ignore", category=FutureWarning)

DOWNLOAD_DATE = pd.Timestamp("2026-11-07")
FILINGS = Path("data/interim/filings.parquet")
PANEL = Path("data/interim/tenk_panel.parquet")
SECTIONS = Path("data/raw/402_sections")
FIXES = Path("data/interim/402_section_fixes.csv")
FAILURES = Path("data/interim/402_failures.csv")
OUT = Path("data/interim/events_402.parquet")

TITLE = re.compile(r"4\.02\s*[.:\s–—-]*Non-?\s?Reliance", re.I)
HEADING_ONLY = re.compile(
    r"^\s*(Item\s+)?4\.02\s*[.:\s–—-]*Non-?\s?Reliance\s+on\s+Previously\s+Issued\s+Financial\s+Statements"
    r"(\s+or\s+a\s+Related\s+Audit\s+Report\s+or\s+Completed\s+Interim\s+Review)?\.?",
    re.I,
)
# a later item heading at the start of a line ("Item 9.01 Financial Statements and Exhibits"),
# words separated by any whitespace (headings sometimes break across lines), or SIGNATURE(S)
LATER_ITEM = re.compile(r"(?m)^[ \t]*Item\s+(?!4\.02)(\d{1,2}\.\d{2})\.?[ \t:–—-]*\s*[A-Z][A-Za-z]+(\s+[A-Za-z]+)*")
SIGNATURE = re.compile(r"(?m)^[ \t]*SIGNATURES?\b")


def find_402(f: pd.DataFrame, panel_ciks: set[str]) -> pd.DataFrame:
    is_402 = f["form"].isin(["8-K", "8-K/A"]) & f["items"].fillna("").str.split(",").apply(lambda xs: "4.02" in xs)
    x = f[is_402 & f["filing_date"].between("2011-01-01", DOWNLOAD_DATE) & f["cik"].isin(panel_ciks)]
    return x[["cik", "form", "filing_date", "accession_number"]].sort_values(
        ["cik", "filing_date", "accession_number"]).reset_index(drop=True)


def group_events(x: pd.DataFrame) -> pd.DataFrame:
    events = []
    for cik, g in x.groupby("cik", sort=True):
        cur = None
        for r in g.itertuples():
            joins = (cur is not None and r.form == "8-K/A"
                     and (r.filing_date - cur["first_filing_date"]).days <= 365)
            if joins:
                cur["accessions"].append(r.accession_number)
                cur["forms"].append(r.form)
                cur["filing_dates"].append(r.filing_date)
            else:
                cur = {"cik": cik, "first_filing_date": r.filing_date, "accessions": [r.accession_number],
                       "forms": [r.form], "filing_dates": [r.filing_date]}
                events.append(cur)
    ev = pd.DataFrame(events)
    ev.insert(1, "event_id", ev["cik"] + "-" + ev.groupby("cik").cumcount().add(1).astype(str))
    ev["n_filings"] = ev["accessions"].str.len()
    return ev


def body_len(section: str) -> int:
    return len(HEADING_ONLY.sub("", section, count=1).strip())


def cut_runon(section: str) -> tuple[str, list[str]]:
    """Cut at the first later-item heading or SIGNATURE that follows real body text.

    Some filers stack headings ("Item 4.02 Non-Reliance ..." directly followed by "Item 8.01
    Other Events.") and put the 4.02 disclosure under the second heading. A heading with no
    body text before it is skipped (logged as "stacked_heading") rather than cut at.
    """
    start = 40  # skip the section's own heading
    cuts = sorted({m.start() for p in (LATER_ITEM, SIGNATURE) for m in p.finditer(section, start)})
    fixes = []
    for p in cuts:
        if body_len(section[:p]) >= 50:
            return section[:p].rstrip(), fixes + ["cut"]
        if "stacked_heading" not in fixes:
            fixes.append("stacked_heading")
    return section, fixes


def extract_section(filing) -> tuple[str, list[str]]:
    fixes = []
    obj = filing.obj()
    section = obj["Item 4.02"]
    if not section or body_len(section) < 50:
        fixes.append("fallback")
        text = obj.text() or ""
        m = TITLE.search(text)
        # skip matches that are just an entry in a list of headings: need real body after it
        while m and body_len(text[m.start(): m.start() + 600]) < 50:
            m = TITLE.search(text, m.end())
        if not m:
            return "", fixes + ["not_found"]
        section = text[m.start():]
    section, cut_fixes = cut_runon(section)
    fixes += cut_fixes
    if body_len(section) < 50:
        fixes.append("not_found")
    return section, fixes


def download_sections(x: pd.DataFrame) -> None:
    SECTIONS.mkdir(parents=True, exist_ok=True)
    todo = x[[not (SECTIONS / f"{a}.txt").exists() for a in x["accession_number"]]]
    print(f"{len(x):,} filings; {len(x) - len(todo):,} already saved; {len(todo):,} to fetch", flush=True)
    new_fix, new_fail = not FIXES.exists(), not FAILURES.exists()
    t0 = time.time()
    with FIXES.open("a", newline="") as ffix, FAILURES.open("a", newline="") as ffail:
        wfix, wfail = csv.writer(ffix), csv.writer(ffail)
        if new_fix:
            wfix.writerow(["accession", "fix"])
        if new_fail:
            wfail.writerow(["cik", "accession", "error"])
        for i, r in enumerate(todo.itertuples(), 1):
            try:
                fl = Filing(cik=int(r.cik), company="", form=r.form,
                            filing_date=r.filing_date.strftime("%Y-%m-%d"), accession_no=r.accession_number)
                section, fixes = extract_section(fl)
                tmp = SECTIONS / f"{r.accession_number}.txt.tmp"
                tmp.write_text(section, encoding="utf-8")
                tmp.rename(SECTIONS / f"{r.accession_number}.txt")
                for fx in fixes:
                    wfix.writerow([r.accession_number, fx])
            except Exception as e:
                wfail.writerow([r.cik, r.accession_number, repr(e)])
            ffix.flush()
            ffail.flush()
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i:,}/{len(todo):,} done, {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left",
                      flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    panel_ciks = set(pd.read_parquet(PANEL, columns=["cik"])["cik"])
    f = pd.read_parquet(FILINGS, columns=["cik", "form", "items", "filing_date", "accession_number"])
    x = find_402(f, panel_ciks)
    ev = group_events(x)
    ev.to_parquet(OUT, index=False)
    print(f"{len(x):,} Item 4.02 filings -> {len(ev):,} events for {ev['cik'].nunique():,} CIKs")
    print("\nEvents per year (first filing):")
    print(ev["first_filing_date"].dt.year.value_counts().sort_index().to_string())

    todo = x if not args.limit else x.head(args.limit)
    download_sections(todo)

    fixes = pd.read_csv(FIXES)
    print("\nSections needing each fix:")
    print(fixes["fix"].value_counts().to_string())
    if FAILURES.exists():
        print(f"Failures: {len(pd.read_csv(FAILURES)):,} (see {FAILURES})")


if __name__ == "__main__":
    main()
