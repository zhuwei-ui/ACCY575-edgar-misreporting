"""§8a — score the regex and LLM extractors against data/labels/402_gold.csv.

For each gold event, both extractors run on the saved section of the event's FIRST filing, and
their fiscal years are then filtered by keep_filed_fyears (only years whose original 10-K was filed
before the 4.02 disclosure count as restated). Events whose saved section is empty are skipped.
Reports exact-match rate, year-level precision and recall, and irregularity accuracy, and writes
every disagreement to data/interim/extractor_disagreements.csv for reading.

Run: uv run python -m src.score_extractors [--no-llm]
"""

import argparse
from pathlib import Path

import pandas as pd

from src.extract_402 import keep_filed_fyears, llm_extract, regex_extract
from src.fiscal import load_company_10ks

GOLD = Path("data/labels/402_gold.csv")
EVENTS = Path("data/interim/events_402.parquet")
SECTIONS = Path("data/raw/402_sections")
OUT = Path("data/interim/extractor_disagreements.csv")


def parse_years(s) -> set[int]:
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return {int(x) for x in str(s).split(";") if x.strip()}


def score(rows: list[dict], name: str) -> dict:
    tp = sum(len(r["pred"] & r["gold"]) for r in rows)
    fp = sum(len(r["pred"] - r["gold"]) for r in rows)
    fn = sum(len(r["gold"] - r["pred"]) for r in rows)
    return {
        "extractor": name,
        "n": len(rows),
        "exact_match": sum(r["pred"] == r["gold"] for r in rows) / len(rows),
        "year_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "year_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "irregularity_accuracy": sum(r["irr_pred"] == r["irr_gold"] for r in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    gold = pd.read_csv(GOLD, dtype={"event_id": str})
    ev = pd.read_parquet(EVENTS).set_index("event_id")
    tenks = load_company_10ks()
    extractors = {"regex": regex_extract} | ({} if args.no_llm else {"llm": llm_extract})

    results, disagreements = [], []
    for name, fn in extractors.items():
        rows = []
        for g in gold.itertuples():
            e = ev.loc[g.event_id]
            text = (SECTIONS / f"{e['accessions'][0]}.txt").read_text(encoding="utf-8")
            if not text.strip():
                if name == "regex":
                    print(f"skipped {g.event_id}: empty section")
                continue
            k = tenks.get(e["cik"], pd.DataFrame({"report_date": []}))
            out = fn(text, k)
            pred = set(keep_filed_fyears(out["restated_fyears"], k, e["first_filing_date"]))
            r = {"event_id": g.event_id, "gold": parse_years(g.restated_fyears), "pred": pred,
                 "irr_gold": str(g.irregularity).strip().lower() in ("true", "1"), "irr_pred": out["irregularity"]}
            rows.append(r)
            if r["gold"] != r["pred"] or r["irr_gold"] != r["irr_pred"]:
                disagreements.append({"extractor": name, "event_id": g.event_id,
                                      "gold": sorted(r["gold"]), "pred": sorted(r["pred"]),
                                      "pred_before_rule": out["restated_fyears"],
                                      "irr_gold": r["irr_gold"], "irr_pred": r["irr_pred"],
                                      "period_ends": out.get("period_ends")})
        results.append(score(rows, name))

    print(pd.DataFrame(results).round(3).to_string(index=False))
    pd.DataFrame(disagreements).to_csv(OUT, index=False)
    print(f"\n{len(disagreements)} disagreements written to {OUT}")


if __name__ == "__main__":
    main()
