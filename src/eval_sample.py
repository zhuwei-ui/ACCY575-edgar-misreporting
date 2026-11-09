"""M9 §4 — fix the evaluation sample once, before any model sees it.

From data/final/panel.parquet and data/features/financial.parquet, keeps CIK-years that, in order:
  1. have text (text_path not null), and so a sampling weight w;
  2. have every §4 financial variable non-missing (rows dropped in features_financial for
     at <= 0 or at_lag <= 0 count here);
  3. are not revealed_before_10k (the 10-K was written after the restatement was public);
  4. are not financial firms: numeric sic outside 6000-6799. Rows with no sic are kept and counted.

Writes data/features/eval_sample.parquet and prints drops by cause and fyear, survivors per year,
and the §3 SPAC checks (blank_check must be zero; restated 2020-2021 CIK-years the exclusions
removed; restated rows with reveal_date in the 2021-04-12..2021-06-30 warrant wave).

Run: uv run python -m src.eval_sample
"""

from pathlib import Path

import pandas as pd

from src.features_financial import FEATURES

PANEL = Path("data/final/panel.parquet")
FIN = Path("data/features/financial.parquet")
OUT = Path("data/features/eval_sample.parquet")

COLS = ["cik", "fyear", "accession", "filing_date", "sic", "blank_check", "w", "random20", "restated",
        "reveal_date", "irregularity", "irregularity_reveal_date", "text_path"]
WARRANT_WAVE = (pd.Timestamp("2021-04-12"), pd.Timestamp("2021-06-30"))


def build(panel: pd.DataFrame, fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eval sample, panel with a `drop_cause` column: first failed filter, or None)."""
    df = panel.merge(fin, on=["cik", "fyear"], how="left", validate="1:1")
    sic_num = pd.to_numeric(df["sic"], errors="coerce")
    tests = [
        ("no_text", df["text_path"].isna()),
        ("missing_financial", df[FEATURES].isna().any(axis=1)),
        ("revealed_before_10k", df["revealed_before_10k"]),
        ("financial_firm", sic_num.between(6000, 6799)),
    ]
    cause = pd.Series(None, index=df.index, dtype=object)
    for name, failed in tests:
        cause = cause.where(cause.notna() | ~failed, name)
    df["drop_cause"] = cause
    keep = df[cause.isna()].copy()
    assert keep["w"].notna().all()
    return keep[COLS + FEATURES].reset_index(drop=True), df


def main() -> None:
    panel = pd.read_parquet(PANEL)
    fin = pd.read_parquet(FIN)
    sample, df = build(panel, fin)

    print("rows dropped by first failed filter (all rows / restated rows):")
    t = df.assign(n=1).pivot_table(index="drop_cause", values=["n", "restated"], aggfunc="sum", dropna=False)
    print(t.reindex(["no_text", "missing_financial", "revealed_before_10k", "financial_firm"]).to_string(), "\n")

    print("rows dropped by cause and fyear (rows with text only):")
    has_text = df[df["text_path"].notna() & df["drop_cause"].notna()]
    print(pd.crosstab(has_text["drop_cause"], has_text["fyear"]).to_string(), "\n")

    print("restated rows dropped by cause and fyear:")
    r = df[(df["restated"] == 1) & df["drop_cause"].notna()]
    print(pd.crosstab(r["drop_cause"], r["fyear"]).to_string(), "\n")

    per_year = sample.groupby("fyear").agg(rows=("cik", "size"), restated=("restated", "sum"),
                                           irregularity=("irregularity", "sum"),
                                           weighted_rows=("w", "sum"))
    print("survivors per fyear:\n", per_year.to_string(), "\n")

    no_sic = sample["sic"].isna() | pd.to_numeric(sample["sic"], errors="coerce").isna()
    print(f"kept rows with no usable sic: {int(no_sic.sum())} (restated {int(sample.loc[no_sic, 'restated'].sum())})")
    print(f"blank_check rows in sample: {int(sample['blank_check'].sum())}")
    assert sample["blank_check"].sum() == 0, "blank-check shells must not be in the evaluation sample"

    r2021 = df[(df["restated"] == 1) & df["fyear"].isin([2020, 2021]) & df["text_path"].notna()]
    removed = r2021[r2021["drop_cause"].notna()]
    print(f"restated fiscal 2020-2021 CIK-years with text: {len(r2021)}; removed {len(removed)} "
          f"(blank_check {int(removed['blank_check'].sum())}); by cause: "
          f"{removed['drop_cause'].value_counts().to_dict()}")
    bc = removed[removed["blank_check"]]
    print(f"  blank-check restated rows removed, by cause: {bc['drop_cause'].value_counts().to_dict()}")

    wave = sample["reveal_date"].between(*WARRANT_WAVE)
    print(f"restated rows in sample with reveal_date {WARRANT_WAVE[0].date()}..{WARRANT_WAVE[1].date()}: "
          f"{int(wave.sum())}, by fyear {sample.loc[wave, 'fyear'].value_counts().sort_index().to_dict()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(sample):,} rows, {int(sample['restated'].sum())} restated, "
          f"{int(sample['irregularity'].sum())} irregularity")


if __name__ == "__main__":
    main()
