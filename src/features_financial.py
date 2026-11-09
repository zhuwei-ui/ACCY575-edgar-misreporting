"""M9 §4 — financial predictors (Dechow et al. 2011 subset buildable from XBRL), keyed by (cik, fyear).

Reads data/final/panel.parquet, writes data/features/financial.parquet. Rows with at <= 0 or
at_lag <= 0 are dropped first (every ratio divides by assets); rows with missing at stay, with
missing features. With avg_at = (at + at_lag) / 2:

    log_at        = log(at)
    rsst_acc      = (noac - noac_lag) / avg_at,  noac = at - che - lt - pstk
                    (Richardson et al. 2005 ΔWC + ΔNCO + ΔFIN; see tests/test_features_financial.py)
    ch_rec        = (rect - rect_lag) / avg_at
    ch_inv        = (invt - invt_lag) / avg_at
    soft_assets   = (at - ppent - che) / at
    ch_cash_sales = cs / cs_lag - 1,  cs = sale - (rect - rect_lag); cs_lag is the same company's
                    cs in fyear - 1 (exactly), else sale_lag (the fallback); missing when cs_lag <= 0
    ch_roa        = (pi - pi_lag) / avg_at  (paper's Appendix B: change in pretax income / avg assets)
    issue         = 1 if sstk > 0 or dltis > 0 else 0
    tot_financing = fincf / avg_at

No winsorizing here (§7 does it inside each window). Prints the share missing per variable and
which inputs were missing.

Run: uv run python -m src.features_financial
"""

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("data/final/panel.parquet")
OUT = Path("data/features/financial.parquet")

FEATURES = ["log_at", "rsst_acc", "ch_rec", "ch_inv", "soft_assets", "ch_cash_sales", "ch_roa",
            "issue", "tot_financing"]

# Raw inputs behind each feature, for the missingness report.
INPUTS = {
    "log_at": ["at"],
    "rsst_acc": ["at", "che", "lt", "pstk", "at_lag", "che_lag", "lt_lag", "pstk_lag"],
    "ch_rec": ["rect", "rect_lag", "at", "at_lag"],
    "ch_inv": ["invt", "invt_lag", "at", "at_lag"],
    "soft_assets": ["at", "ppent", "che"],
    "ch_cash_sales": ["sale", "rect", "rect_lag", "cs_lag"],
    "ch_roa": ["pi", "pi_lag", "at", "at_lag"],
    "issue": ["sstk", "dltis"],
    "tot_financing": ["fincf", "at", "at_lag"],
}


def noac(at, che, lt, pstk):
    """Net operating assets in the reduced form: at - che - lt - pstk."""
    return at - che - lt - pstk


def compute(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (features keyed by cik, fyear, plus diagnostic columns cs_lag/cs_lag_source; stats)."""
    assert not panel.duplicated(["cik", "fyear"]).any(), "panel must be one row per (cik, fyear)"
    nonpos = (panel["at"] <= 0) | (panel["at_lag"] <= 0)
    stats = {"dropped_nonpos_at": int(nonpos.sum()),
             "dropped_at_le0": int((panel["at"] <= 0).sum()),
             "dropped_at_lag_le0": int((panel["at_lag"] <= 0).sum())}
    df = panel.loc[~nonpos].sort_values(["cik", "fyear"]).reset_index(drop=True).copy()

    avg_at = (df["at"] + df["at_lag"]) / 2
    out = df[["cik", "fyear"]].copy()
    out["log_at"] = np.log(df["at"])
    out["rsst_acc"] = (noac(df["at"], df["che"], df["lt"], df["pstk"])
                       - noac(df["at_lag"], df["che_lag"], df["lt_lag"], df["pstk_lag"])) / avg_at
    out["ch_rec"] = (df["rect"] - df["rect_lag"]) / avg_at
    out["ch_inv"] = (df["invt"] - df["invt_lag"]) / avg_at
    out["soft_assets"] = (df["at"] - df["ppent"] - df["che"]) / df["at"]

    cs = df["sale"] - (df["rect"] - df["rect_lag"])
    prev_cs = cs.groupby(df["cik"]).shift()
    prev_fyear = df["fyear"].groupby(df["cik"]).shift()
    cs_prior = prev_cs.where(prev_fyear == df["fyear"] - 1)
    use_fallback = cs_prior.isna()
    cs_lag = cs_prior.where(~use_fallback, df["sale_lag"])
    out["cs_lag"] = cs_lag
    out["cs_lag_source"] = np.where(~use_fallback, "prior_row",
                                    np.where(df["sale_lag"].notna(), "sale_lag", "none"))
    out["ch_cash_sales"] = (cs / cs_lag - 1).where(cs_lag > 0)

    out["ch_roa"] = (df["pi"] - df["pi_lag"]) / avg_at
    sstk_pos, dltis_pos = df["sstk"] > 0, df["dltis"] > 0
    issue = (sstk_pos | dltis_pos).astype(float)
    # Unknown only when no input is positive and at least one is missing.
    issue[~(sstk_pos | dltis_pos) & (df["sstk"].isna() | df["dltis"].isna())] = np.nan
    out["issue"] = issue
    out["tot_financing"] = df["fincf"] / avg_at

    has_cs = cs.notna()
    stats["cs_lag_prior_row"] = int((~use_fallback & has_cs).sum())
    stats["cs_lag_fallback_sale_lag"] = int((use_fallback & df["sale_lag"].notna() & has_cs).sum())
    stats["cs_lag_le0"] = int((has_cs & (cs_lag <= 0)).sum())
    stats["n_rows"] = len(out)
    return out, stats


def missing_report(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Per feature: share missing, and among missing rows the share with each input missing."""
    df = panel.merge(feats[["cik", "fyear", "cs_lag"]], on=["cik", "fyear"], how="inner")
    df = df.merge(feats[["cik", "fyear"] + FEATURES], on=["cik", "fyear"], how="inner")
    rows = []
    for f in FEATURES:
        miss = df[f].isna()
        causes = {c: round(df.loc[miss, c].isna().mean(), 3) for c in INPUTS[f]} if miss.any() else {}
        if f == "ch_cash_sales" and miss.any():
            causes["cs_lag<=0"] = round((df.loc[miss, "cs_lag"] <= 0).mean(), 3)
        top = ", ".join(f"{k} {v:.0%}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1]) if v > 0)
        rows.append({"feature": f, "share_missing": round(miss.mean(), 3), "n_missing": int(miss.sum()),
                     "inputs_missing_among_missing_rows": top})
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(PANEL)
    feats, stats = compute(panel)
    print(f"dropped {stats['dropped_nonpos_at']} rows with at <= 0 or at_lag <= 0 "
          f"(at <= 0: {stats['dropped_at_le0']}, at_lag <= 0: {stats['dropped_at_lag_le0']})")
    n_cs = stats["cs_lag_prior_row"] + stats["cs_lag_fallback_sale_lag"]
    print(f"cs_lag for rows with cs: prior-fyear row {stats['cs_lag_prior_row']:,}, "
          f"fallback sale_lag {stats['cs_lag_fallback_sale_lag']:,} "
          f"({stats['cs_lag_fallback_sale_lag'] / max(n_cs, 1):.1%}); cs_lag <= 0: {stats['cs_lag_le0']:,}")
    fb = feats[feats["cs_lag_source"] == "sale_lag"].groupby("fyear").size()
    print("fallback rows by fyear:", fb.to_dict(), "\n")

    with pd.option_context("display.width", 200, "display.max_colwidth", 120):
        print(missing_report(panel, feats).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats[["cik", "fyear"] + FEATURES].to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(feats):,} rows")


if __name__ == "__main__":
    main()
