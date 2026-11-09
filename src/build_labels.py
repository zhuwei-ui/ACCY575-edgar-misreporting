"""§8a — restatement labels, one row per (cik, fyear) declared unreliable by an Item 4.02.

Extractor: llm_extract (claude-haiku-4-5). On the 30-event gold set it scored exact match 1.00 vs
0.90 for the regex baseline (see README). Responses come from the cache filled by
src.llm_batch_402; anything not cached is sent live (cached afterwards).

Rules
- Each event is read from the saved section of its FIRST filing. If that section is empty (Item
  4.02 text not found), the event's next filing with a non-empty section is used instead, and
  `section_source` records it.
- restated = 1 for every fiscal year whose annual statements, or any interim period inside it, are
  declared unreliable (the walkthrough's rule). The owner's stricter rule, used for the gold set,
  only counts years whose original 10-K was filed before the disclosure; it equals
  `restated == 1 & ~revealed_before_10k` in the final panel (§9), so no information is lost.
- A CIK-year hit by several events keeps the earliest reveal_date, irregularity = any, and
  irregularity_reveal_date = earliest first_filing_date among events that flag an irregularity.
- A disclosure cannot withdraw a period that had not ended yet: period ends after the event's
  first_filing_date (typos, or future filings mentioned in passing) are dropped before mapping to
  fiscal years, as are unparseable dates. Dropped periods, and years more than 15 years before the
  disclosure, are logged to data/interim/labels_implausible.csv.
- Only fyear 2011-2021 is written.

Run: uv run python -m src.build_labels
"""

from pathlib import Path

import pandas as pd

from src.extract_402 import llm_extract
from src.fiscal import load_company_10ks, period_to_fyear

EVENTS = Path("data/interim/events_402.parquet")
SECTIONS = Path("data/raw/402_sections")
OUT = Path("data/interim/labels.parquet")
EVENT_OUT = Path("data/interim/event_extractions.parquet")
IMPLAUSIBLE = Path("data/interim/labels_implausible.csv")
FAILURES = Path("data/interim/labels_failures.csv")


def section_for(accessions) -> tuple[str, str | None]:
    for acc in accessions:
        p = SECTIONS / f"{acc}.txt"
        if p.exists():
            t = p.read_text(encoding="utf-8")
            if t.strip():
                return t, acc
    return "", None


def main() -> None:
    ev = pd.read_parquet(EVENTS)
    tenks = load_company_10ks()
    empty = pd.DataFrame({"report_date": []})

    rows, fails = [], []
    for e in ev.itertuples():
        text, acc = section_for(list(e.accessions))
        if not text:
            fails.append({"event_id": e.event_id, "error": "no non-empty section in any filing"})
            continue
        try:
            out = llm_extract(text, tenks.get(e.cik, empty))
        except Exception as exc:  # noqa: BLE001
            fails.append({"event_id": e.event_id, "error": repr(exc)})
            continue
        k = tenks.get(e.cik, empty)
        ok_ends, dropped = [], []
        for pe in out["period_ends"]:
            d = pd.to_datetime(pe, errors="coerce")
            (ok_ends if pd.notna(d) and d <= e.first_filing_date else dropped).append(pe)
        fyears = sorted({fy for fy in (period_to_fyear(pe, k) for pe in ok_ends) if fy is not None})
        rows.append({"event_id": e.event_id, "cik": e.cik, "first_filing_date": e.first_filing_date,
                     "section_source": acc, "fallback_section": acc != e.accessions[0],
                     "restated_fyears": fyears, "period_ends": ok_ends, "dropped_period_ends": dropped,
                     "irregularity": out["irregularity"],
                     "irregularity_evidence": out["irregularity_evidence"]})
    x = pd.DataFrame(rows)
    x.to_parquet(EVENT_OUT, index=False)
    pd.DataFrame(fails, columns=["event_id", "error"]).to_csv(FAILURES, index=False)
    print(f"events extracted: {len(x)} of {len(ev)} ({x['fallback_section'].sum()} from a later filing); "
          f"failures: {len(fails)} -> {FAILURES}")
    print(f"events with no restated year: {(x['restated_fyears'].str.len() == 0).sum()}")

    long = x.explode("restated_fyears").dropna(subset=["restated_fyears"])
    long = long.rename(columns={"restated_fyears": "fyear"}).astype({"fyear": int})
    yr = pd.to_datetime(long["first_filing_date"]).dt.year
    bad = long[long["fyear"] < yr - 15]
    drops = x[x["dropped_period_ends"].str.len() > 0]
    pd.concat([bad[["event_id", "cik", "fyear", "first_filing_date", "period_ends"]].assign(reason="> 15 years old"),
               drops[["event_id", "cik", "first_filing_date", "dropped_period_ends"]].assign(reason="period ends after disclosure")]
              ).to_csv(IMPLAUSIBLE, index=False)
    print(f"logged -> {IMPLAUSIBLE}: {len(drops)} events with periods after the disclosure "
          f"({drops['dropped_period_ends'].str.len().sum()} periods dropped); {len(bad)} years > 15 years old")

    long = long.drop(bad.index)
    long["irr_date"] = long["first_filing_date"].where(long["irregularity"])
    lab = (long.groupby(["cik", "fyear"])
               .agg(reveal_date=("first_filing_date", "min"),
                    irregularity=("irregularity", "any"),
                    irregularity_reveal_date=("irr_date", "min"),
                    event_ids=("event_id", lambda s: sorted(set(s))))
               .reset_index())
    lab.insert(2, "restated", 1)
    lab["n_events"] = lab["event_ids"].str.len()
    lab = lab[lab["fyear"].between(2011, 2021)].reset_index(drop=True)

    assert not lab.duplicated(["cik", "fyear"]).any()
    assert (lab["irregularity_reveal_date"].isna() | (lab["irregularity_reveal_date"] >= lab["reveal_date"])).all()
    assert (lab["irregularity"] == lab["irregularity_reveal_date"].notna()).all()
    lab.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(lab)} restated CIK-years, {lab['cik'].nunique()} CIKs, "
          f"{lab['irregularity'].sum()} irregularity")
    print(lab.groupby("fyear").agg(restated=("restated", "size"), irregularity=("irregularity", "sum"),
                                   multi_event=("n_events", lambda s: (s > 1).sum())).to_string())


if __name__ == "__main__":
    main()
