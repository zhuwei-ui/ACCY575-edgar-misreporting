"""§6 — Financial statement items from XBRL, as first reported.

Source: SEC Financial Statement Data Sets (data/raw/fsds/{y}q{q}.zip), selected by accession,
so every number is the number *that 10-K* reported (no look-ahead from later restatements).

Filters (per quarterly ZIP, read in memory with zipfile, all columns as strings):
  sub: adsh is a 10-K accession in tenk_panel AND sub.cik == that panel row's cik
  num: uom == USD, empty segments and coreg, version != adsh (standard tags only)
  instant  items: qtrs == 0;  duration items: qtrs == 4
  current: ddate == period;  lag (*_lag): ddate == period - 12 months (month-end)

Tags are walked per period (current and lag separately): first tag in the list with a value.
sstk and dltis take the largest value across all their tags.

Missing-value policy: if `at` (current) is present, rect, invt, pstk, sstk, dltis (current and
lag) not found after trying every tag are set to 0, and listed in `zero_filled`. Everything
else stays NaN. Coverage is printed before zero-filling.

Step 8: panel 10-Ks with isXBRL == 1 that appear in no quarter's sub.txt are filled from the
filing's own XBRL (fin_source = "xbrl"); see src/xbrl_fallback.py.

period_mismatch: True when the FSDS `period` (the filing's balance-sheet date) is more than 31
days from the filer-entered report_date that §5 used to assign fyear. Decision (2026-11-07): keep
§5's fyear as is (walkthrough rule) and flag these rows instead of correcting them; about 186
rows, of which ~40 would move to a different fyear. Module 9 can drop or inspect them.

Tag-list changes (record every one here):
  - none yet; lists are exactly the walkthrough's §6c table.

Run: uv run python -u -m src.build_financials [--skip-xbrl]
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("data/interim/tenk_panel.parquet")
FSDS = Path("data/raw/fsds")
LONG = Path("data/interim/fsds_values.parquet")  # filtered long table (cache for re-runs)
SUBS_OUT = Path("data/interim/fsds_sub.parquet")
OUT = Path("data/interim/financials.parquet")

INSTANT = {
    "at": ["Assets"],
    "che": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ],
    "rect": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
        "AccountsAndOtherReceivablesNetCurrent",
    ],
    "invt": ["InventoryNet"],
    "ppent": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "lt": ["Liabilities"],
    "pstk": ["PreferredStockValue"],
}
LT_DERIVE = {
    "lse": ["LiabilitiesAndStockholdersEquity"],
    "seq": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "temp": [
        "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
        "TemporaryEquityCarryingAmountAttributableToParent",
    ],
}
DURATION = {
    "sale": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "ni": ["NetIncomeLoss", "ProfitLoss"],
    "pi": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "fincf": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ],
    "sstk": [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
        "ProceedsFromIssuanceOrSaleOfEquity",
        "ProceedsFromIssuanceInitialPublicOffering",
        "ProceedsFromIssuanceOfPrivatePlacement",
        "ProceedsFromIssuanceOfPreferredStockAndPreferenceStock",
    ],
    "dltis": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromIssuanceOfSecuredDebt",
        "ProceedsFromIssuanceOfUnsecuredDebt",
        "ProceedsFromConvertibleDebt",
        "ProceedsFromNotesPayable",
        "ProceedsFromBankDebt",
        "ProceedsFromLongTermLinesOfCredit",
        "ProceedsFromLinesOfCredit",
        "ProceedsFromRelatedPartyDebt",
    ],
}
PI_DERIVE = {
    "pi_dom": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "pi_for": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"],
}
MAX_VARS = {"sstk", "dltis"}
ZERO_FILL = ["rect", "invt", "pstk", "sstk", "dltis"]
VARS = ["at", "che", "rect", "invt", "ppent", "lt", "pstk", "sale", "ni", "pi", "fincf", "sstk", "dltis"]

INSTANT_TAGS = {t for d in (INSTANT, LT_DERIVE) for ts in d.values() for t in ts}
DURATION_TAGS = {t for d in (DURATION, PI_DERIVE) for ts in d.values() for t in ts}
ALL_TAGS = INSTANT_TAGS | DURATION_TAGS


# --------------------------------------------------------------------------- extraction
def read_quarter(zpath: Path, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sub rows kept, filtered num rows with which='cur'/'lag') for one quarterly ZIP."""
    with zipfile.ZipFile(zpath) as z:
        sub = pd.read_csv(io.BytesIO(z.read("sub.txt")), sep="\t", dtype=str, keep_default_na=False,
                          usecols=["adsh", "cik", "form", "period", "sic", "filed"])
        sub = sub.merge(keys, left_on=["adsh", "cik"], right_on=["accession", "cik"], how="inner")
        sub = sub.drop_duplicates("adsh")
        if sub.empty:
            return sub, pd.DataFrame()
        adshs = set(sub["adsh"])
        parts = []
        with z.open("num.txt") as fh:
            reader = pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False, chunksize=2_000_000,
                                 usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments",
                                          "coreg", "value"], quoting=3, encoding="utf-8",
                                 encoding_errors="replace")
            for ch in reader:
                ch = ch[ch["adsh"].isin(adshs) & ch["tag"].isin(ALL_TAGS)]
                ch = ch[(ch["uom"] == "USD") & (ch["segments"] == "") & (ch["coreg"] == "")
                        & (ch["version"] != ch["adsh"]) & ch["qtrs"].isin(["0", "4"])]
                parts.append(ch)
    num = pd.concat(parts, ignore_index=True)
    # instant tags only at qtrs 0, duration tags only at qtrs 4
    num = num[((num["qtrs"] == "0") & num["tag"].isin(INSTANT_TAGS))
              | ((num["qtrs"] == "4") & num["tag"].isin(DURATION_TAGS))]
    num = num.merge(sub[["adsh", "period"]], on="adsh")
    period = pd.to_datetime(num["period"], format="%Y%m%d")
    lag = (period - pd.DateOffset(months=12)) + pd.offsets.MonthEnd(0)
    ddate = pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce")
    num["which"] = np.select([ddate == period, ddate == lag], ["cur", "lag"], default="")
    num = num[num["which"] != ""]
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num[num["value"].notna()]
    return sub, num[["adsh", "tag", "ddate", "qtrs", "which", "value"]]


def extract_all(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = panel[["cik", "accession"]].drop_duplicates()
    subs, nums = [], []
    for zpath in sorted(FSDS.glob("*.zip")):
        sub, num = read_quarter(zpath, keys)
        print(f"  {zpath.name}: {len(sub):,} panel 10-Ks, {len(num):,} values", flush=True)
        subs.append(sub)
        nums.append(num)
    sub = pd.concat(subs, ignore_index=True).drop_duplicates("adsh")
    num = pd.concat(nums, ignore_index=True).drop_duplicates(["adsh", "tag", "ddate", "qtrs", "value"])
    dup = num.duplicated(["adsh", "tag", "ddate", "qtrs"], keep=False)
    assert not dup.any(), f"{dup.sum()} rows violate one value per (adsh, tag, ddate, qtrs):\n{num[dup].head(20)}"
    return sub, num


# --------------------------------------------------------------------------- variables
def first_tag(wide: pd.DataFrame, tags: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=wide.index)
    for t in tags:
        if t in wide:
            out = out.fillna(wide[t])
    return out


def max_tag(wide: pd.DataFrame, tags: list[str]) -> pd.Series:
    cols = [t for t in tags if t in wide]
    return wide[cols].max(axis=1) if cols else pd.Series(np.nan, index=wide.index)


def compute_vars(long: pd.DataFrame) -> pd.DataFrame:
    """long: (adsh, tag, which, value) -> one row per adsh with VARS and VARS_lag."""
    out = []
    for which, suffix in [("cur", ""), ("lag", "_lag")]:
        w = long[long["which"] == which].pivot(index="adsh", columns="tag", values="value")
        v = pd.DataFrame(index=w.index)
        for var, tags in {**INSTANT, **DURATION}.items():
            v[var] = max_tag(w, tags) if var in MAX_VARS else first_tag(w, tags)
        # lt fallback: L&SE - equity (incl. NCI, else parent) - temporary equity (if present)
        derived = first_tag(w, LT_DERIVE["lse"]) - first_tag(w, LT_DERIVE["seq"]) - first_tag(w, LT_DERIVE["temp"]).fillna(0)
        v["lt_derived"] = v["lt"].isna() & derived.notna()
        v["lt"] = v["lt"].fillna(derived)
        # pi fallback: domestic + foreign, only when both present
        v["pi"] = v["pi"].fillna(first_tag(w, PI_DERIVE["pi_dom"]) + first_tag(w, PI_DERIVE["pi_for"]))
        out.append(v.add_suffix(suffix))
    return pd.concat(out, axis=1)


def apply_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    has_at = df["at"].notna()
    filled = pd.Series("", index=df.index)
    for var in ZERO_FILL:
        for col in (var, f"{var}_lag"):
            m = has_at & df[col].isna()
            df.loc[m, col] = 0.0
            filled[m] = filled[m] + col + ";"
    df["zero_filled"] = filled.str.rstrip(";").replace("", None)
    return df


def coverage(df: pd.DataFrame, title: str) -> None:
    cols = VARS
    print(f"\n{title}: share non-missing (before zero-filling)")
    print(df[cols].notna().mean().round(3).to_string())
    print("\nby fyear:")
    print(df.groupby("fyear")[cols].apply(lambda g: g.notna().mean()).round(2).to_string())


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xbrl", action="store_true", help="skip step 8 (filing XBRL fallback)")
    ap.add_argument("--reextract", action="store_true", help="re-read the ZIPs even if cached")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL)
    if LONG.exists() and SUBS_OUT.exists() and not args.reextract:
        num, sub = pd.read_parquet(LONG), pd.read_parquet(SUBS_OUT)
        print(f"Using cached {LONG} ({len(num):,} values)")
    else:
        print("Reading FSDS ZIPs...", flush=True)
        sub, num = extract_all(panel)
        num.to_parquet(LONG, index=False)
        sub.to_parquet(SUBS_OUT, index=False)

    vals = compute_vars(num).reset_index().rename(columns={"adsh": "accession"})
    vals["fsds_period"] = vals["accession"].map(sub.set_index("adsh")["period"])
    fin = panel[["cik", "fyear", "accession"]].merge(
        sub[["adsh", "cik"]].rename(columns={"adsh": "accession"}), on=["cik", "accession"]
    ).merge(vals, on="accession", how="left")
    fin["fin_source"] = "fsds"

    if not args.skip_xbrl:
        from src.xbrl_fallback import fill_from_xbrl

        xb = fill_from_xbrl(panel, sub)
        if len(xb):
            xb["fin_source"] = "xbrl"
            fin = pd.concat([fin, xb], ignore_index=True)
        print(f"\nFilled {len(xb):,} 10-Ks from their own XBRL (step 8)")

    fin["lt_derived"] = fin["lt_derived"].fillna(False).astype(bool)
    fin["lt_derived_lag"] = fin["lt_derived_lag"].fillna(False).astype(bool)
    coverage(fin, f"All {len(fin):,} 10-Ks with financials")
    no_row = len(panel) - len(fin)
    print(f"\nPanel 10-Ks with no financials row at all: {no_row:,} of {len(panel):,}")
    print("  of which combined filings:", (~panel.set_index(["cik", "accession"]).index.isin(
        fin.set_index(["cik", "accession"]).index) & panel["combined"]).sum())

    rd = fin.merge(panel[["cik", "accession", "report_date"]], on=["cik", "accession"], how="left")["report_date"]
    fp = pd.to_datetime(fin["fsds_period"], format="%Y%m%d", errors="coerce")
    fin["period_mismatch"] = ((rd.values - fp).dt.days.abs() > 31).fillna(False).astype(bool)
    print(f"period_mismatch rows (FSDS period >31 days from report_date): {fin['period_mismatch'].sum():,}")

    fin = apply_zero_fill(fin)
    assert not fin.duplicated(["cik", "fyear", "accession"]).any()
    fin.to_parquet(OUT, index=False)
    print(f"\nWrote {len(fin):,} rows to {OUT}")


if __name__ == "__main__":
    main()
