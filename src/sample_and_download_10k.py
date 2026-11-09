"""§8b — case-control sample of CIK-years, then the primary 10-K document for each, as clean text.

1. Join tenk_panel to labels, sort by (cik, fyear), draw u = default_rng(575).random(n) once for
   every CIK-year, random20 = u < 0.2. Keep every labeled CIK-year (sampled_as="restated", w=1) and
   every unlabeled CIK-year with random20 (sampled_as="control", w=5).
   -> data/interim/text_sample.parquet
2. Download only the primary document, parse with edgartools' HTMLParser (drops the inline-XBRL
   header), NFKC-normalize, strip box-drawing characters, collapse whitespace, and save gzip UTF-8 to
   data/raw/tenk_text/{accession}.txt.gz (temp file + rename, so a crash never leaves a partial file).
3. Resumable: existing outputs are skipped. Progress every 100 documents. Failures ->
   data/interim/tenk_text_failures.csv.

Run: uv run python -m src.sample_and_download_10k [--limit N] [--sample-only]
"""

import argparse
import gzip
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import src.sec  # noqa: F401  (loads .env before edgar)
from edgar.documents import HTMLParser, ParserConfig
from edgar.httprequests import download_file

PANEL = Path("data/interim/tenk_panel.parquet")
LABELS = Path("data/interim/labels.parquet")
SAMPLE = Path("data/interim/text_sample.parquet")
TEXT_DIR = Path("data/raw/tenk_text")
FAILURES = Path("data/interim/tenk_text_failures.csv")

BOX = re.compile(r"[─-╿▀-▟]")


def build_sample() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL)
    labels = pd.read_parquet(LABELS, columns=["cik", "fyear", "restated"])
    t = panel.merge(labels, on=["cik", "fyear"], how="left").sort_values(["cik", "fyear"]).reset_index(drop=True)
    u = np.random.default_rng(575).random(len(t))
    t["random20"] = u < 0.2
    restated = t["restated"].eq(1)
    t["sampled_as"] = np.select([restated, t["random20"]], ["restated", "control"], default=None)
    t["w"] = np.select([restated, t["random20"]], [1, 5], default=0)
    s = t[t["sampled_as"].notna()][["cik", "fyear", "accession", "primary_document", "sampled_as", "w", "random20"]]
    s = s.reset_index(drop=True)
    s.to_parquet(SAMPLE, index=False)
    print(f"panel CIK-years {len(t):,}; labeled {restated.sum():,}; random20 {t['random20'].sum():,}")
    print(f"sample {len(s):,} CIK-years ({s['accession'].nunique():,} unique accessions): "
          f"{s['sampled_as'].value_counts().to_dict()}")
    return s


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = BOX.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_text(cik: str, accession: str, primary_document: str) -> str:
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
           f"{accession.replace('-', '')}/{primary_document}")
    raw = download_file(url, as_text=True)
    if not raw:
        raise ValueError("empty download")
    if primary_document.lower().endswith(".txt"):
        return clean(raw)
    return clean(HTMLParser(ParserConfig(form="10-K")).parse(raw).text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    s = build_sample()
    if args.sample_only:
        return
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    docs = s.drop_duplicates("accession")
    if args.limit:
        docs = docs.head(args.limit)
    todo = [r for r in docs.itertuples() if not (TEXT_DIR / f"{r.accession}.txt.gz").exists()]
    print(f"{len(docs) - len(todo):,} already done; {len(todo):,} to download", flush=True)

    # keep only failures that are still missing (a re-run retries them; successes leave the log)
    failures = [f for f in (pd.read_csv(FAILURES).to_dict("records") if FAILURES.exists() else [])
                if not (TEXT_DIR / f"{f['accession']}.txt.gz").exists()]
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        out = TEXT_DIR / f"{r.accession}.txt.gz"
        try:
            text = fetch_text(r.cik, r.accession, r.primary_document)
            if len(text) < 1000:
                raise ValueError(f"text too short ({len(text)} chars)")
            tmp = out.with_suffix(".gz.part")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, out)
            failures = [f for f in failures if f["accession"] != r.accession]
        except Exception as exc:  # noqa: BLE001
            failures.append({"accession": r.accession, "cik": r.cik, "primary_document": r.primary_document,
                             "error": repr(exc)[:500], "time": pd.Timestamp.now().isoformat()})
            pd.DataFrame(failures).to_csv(FAILURES, index=False)
        if i == len(todo):
            pd.DataFrame(failures, columns=["accession", "cik", "primary_document", "error", "time"]).to_csv(FAILURES, index=False)
        if i % 100 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"done {i:,} / {len(todo):,}; remaining {len(todo) - i:,}; elapsed {el / 60:.1f} min; "
                  f"ETA {el / i * (len(todo) - i) / 60:.1f} min; failures {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
