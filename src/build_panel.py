"""§9 — assemble and validate data/final/panel.parquet, one row per CIK-year (fyear 2011-2021).

tenk_panel <- financials on (cik, fyear, accession) <- labels on (cik, fyear) <- text_sample on
(cik, fyear). Adds revealed_before_10k (reveal_date <= filing_date; False when unrestated),
blank_check (sic 6770, SPACs), and text_path where the downloaded text exists.

Negative total assets (2 shell-company filer errors as of this build: CIK 1404212 FY2012 at = -17,
CIK 1625285 FY2018 at = -13) are set to missing so the schema's at >= 0 check holds; they are printed.

The owner's stricter restated definition (gold set rule: the year's 10-K was filed before the
disclosure) is `restated == 1 & ~revealed_before_10k`.

Run: uv run python -m src.build_panel
"""

from pathlib import Path

import pandas as pd

from src.schemas import panel_schema

TEXT_DIR = Path("data/raw/tenk_text")
OUT = Path("data/final/panel.parquet")


def main() -> None:
    panel = pd.read_parquet("data/interim/tenk_panel.parquet")
    fin = pd.read_parquet("data/interim/financials.parquet")
    lab = pd.read_parquet("data/interim/labels.parquet")
    samp = pd.read_parquet("data/interim/text_sample.parquet")

    df = panel.merge(fin, on=["cik", "fyear", "accession"], how="left", validate="1:1")
    df = df.merge(lab.drop(columns=["event_ids"]), on=["cik", "fyear"], how="left", validate="1:1")
    neg = df["at"] < 0
    if neg.any():
        print("setting negative at to missing:\n", df.loc[neg, ["cik", "fyear", "accession", "at"]].to_string(), "\n")
        df.loc[neg, "at"] = float("nan")
    df["restated"] = df["restated"].fillna(0).astype(int)
    df["irregularity"] = df["irregularity"].fillna(False).astype(bool)
    df["n_events"] = df["n_events"].fillna(0).astype(int)
    df = df.merge(samp[["cik", "fyear", "sampled_as", "w", "random20"]], on=["cik", "fyear"],
                  how="left", validate="1:1")
    # every random20 == True row is in text_sample (as restated or control), so unsampled rows are False
    df["random20"] = df["random20"].fillna(False).astype(bool)
    df["w"] = df["w"].astype(float)

    have = {p.name.removesuffix(".txt.gz") for p in TEXT_DIR.glob("*.txt.gz")} if TEXT_DIR.exists() else set()
    df["text_path"] = [str(TEXT_DIR / f"{a}.txt.gz") if (s is not None and not pd.isna(s) and a in have) else None
                       for a, s in zip(df["accession"], df["sampled_as"])]
    df["revealed_before_10k"] = (df["reveal_date"] <= df["filing_date"]).fillna(False).astype(bool)
    df["blank_check"] = df["sic"].eq("6770")

    df = panel_schema.validate(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} rows x {df.shape[1]} columns\n")

    by = df.groupby("fyear").agg(
        cik_years=("cik", "size"),
        at_share=("at", lambda s: round(s.notna().mean(), 3)),
        restated=("restated", "sum"),
        revealed_before_10k=("revealed_before_10k", "sum"),
        irregularity=("irregularity", "sum"),
        blank_check=("blank_check", "sum"),
        restated_blank_check=("blank_check", lambda s: (s & df.loc[s.index, "restated"].eq(1)).sum()),
        with_text=("text_path", lambda s: s.notna().sum()),
    )
    by.loc["total"] = by.sum()
    by.loc["total", "at_share"] = round(df["at"].notna().mean(), 3)
    print(by.to_string())
    r = df[df["restated"] == 1]
    print(f"\nrestated CIK-years with text: {r['text_path'].notna().mean():.3f}; "
          f"owner's stricter rule (restated & not revealed_before_10k): {(~r['revealed_before_10k']).sum():,}")


if __name__ == "__main__":
    main()
