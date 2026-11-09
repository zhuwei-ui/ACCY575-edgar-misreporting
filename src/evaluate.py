"""M9 §7 — rolling out-of-sample logistic models, weighted pooled AUC, cluster-bootstrap differences.

Everything (training and testing) is restricted to data/features/eval_sample.parquet. For each
outcome (restated, irregularity), test year T in 2016-2021 and model spec:
  1. training rows: fyear in T-5..T-1 and filing_date <= T-06-30; test rows: fyear == T;
  2. winsorize every continuous financial and style variable at the w-weighted 1st/99th percentiles
     of the training rows, applied to both;
  3. standardize every feature with the training rows' w-weighted mean and SD;
  4. LogisticRegression(C=1.0, max_iter=5000) with sample_weight=w; score the test rows.
Training label: outcome == 1 and its reveal date <= cutoff. Test label: outcome == 1 (eventual truth).
Topic proportions for both come from the window's LDA (data/features/topics{tag}_{T}.parquet).

Outputs (suffix --out-tag for robustness runs): results/m9/auc{tag}.csv (pooled and per-year
weighted AUCs), results/m9/auc_diff{tag}.csv (cluster-bootstrap by CIK, 1,000 resamples, seed 575),
results/m9/coefs{tag}.csv, and data/features/oos_scores{tag}.parquet.

Run: uv run python -m src.evaluate [--topics-tag _df0025] [--out-tag _df0025] [--drop-warrant-wave]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features_financial import FEATURES as FIN_ALL
from src.features_style import STYLE
from src.topics_rolling import TEST_YEARS, topic_cols, window_of

FEAT = Path("data/features")
RES = Path("results/m9")
FIN = FIN_ALL
FIN_CONT = [f for f in FIN if f != "issue"]
TOPIC = topic_cols()
MODELS = {
    "FIN": FIN, "STYLE": STYLE, "TOPIC": TOPIC,
    "FIN+STYLE": FIN + STYLE, "FIN+TOPIC": FIN + TOPIC, "STYLE+TOPIC": STYLE + TOPIC,
    "FIN+STYLE+TOPIC": FIN + STYLE + TOPIC,
}
WINSORIZE = set(FIN_CONT) | set(STYLE)
PAIRS = [("FIN", "FIN+TOPIC"), ("STYLE", "STYLE+TOPIC"), ("FIN", "TOPIC"), ("FIN+STYLE", "FIN+STYLE+TOPIC")]
OUTCOMES = {"restated": ("restated", "reveal_date"), "irregularity": ("irregularity", "irregularity_reveal_date")}
WARRANT_WAVE = (pd.Timestamp("2021-04-12"), pd.Timestamp("2021-06-30"))


def wpercentile(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    return np.percentile(x, q, weights=w, method="inverted_cdf")


def prepare(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Winsorize (training caps) and standardize (training moments), both w-weighted."""
    w = train["w"].to_numpy()
    Xtr, Xte = train[feats].to_numpy(float).copy(), test[feats].to_numpy(float).copy()
    for j, f in enumerate(feats):
        if f in WINSORIZE:
            lo, hi = wpercentile(Xtr[:, j], w, [1, 99])
            Xtr[:, j] = np.clip(Xtr[:, j], lo, hi)
            Xte[:, j] = np.clip(Xte[:, j], lo, hi)
    mean = np.average(Xtr, axis=0, weights=w)
    sd = np.sqrt(np.average((Xtr - mean) ** 2, axis=0, weights=w))
    sd[sd == 0] = 1.0  # a constant feature stays constant (0) after centering
    return (Xtr - mean) / sd, (Xte - mean) / sd


def weighted_auc(y, s, w) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def load(topics_tag: str, drop_wave: bool) -> tuple[pd.DataFrame, dict]:
    sample = pd.read_parquet(FEAT / "eval_sample.parquet")
    style = pd.read_parquet(FEAT / "style.parquet")
    df = sample.merge(style, on=["cik", "fyear"], how="left", validate="1:1")
    assert df[FIN + STYLE].notna().all().all(), "missing financial or style values in eval sample"
    if drop_wave:
        wave = df["reveal_date"].between(*WARRANT_WAVE)
        print(f"dropping {int(wave.sum())} restated rows revealed in the warrant wave")
        df = df[~wave].reset_index(drop=True)
    topics = {T: pd.read_parquet(FEAT / f"topics{topics_tag}_{T}.parquet") for T in TEST_YEARS}
    return df, topics


def run(df: pd.DataFrame, topics: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    scores, coefs, notes = [], [], []
    for outcome, (ycol, rcol) in OUTCOMES.items():
        for T in TEST_YEARS:
            years, cutoff = window_of(T)
            tp = topics[T]
            train = df[df["fyear"].isin(years) & (df["filing_date"] <= cutoff)]
            train = train.merge(tp[tp["role"] == "window"].drop(columns="role"), on=["cik", "fyear"],
                                how="left", validate="1:1")
            test = df[df["fyear"] == T].merge(tp[tp["role"] == "test"].drop(columns="role"),
                                              on=["cik", "fyear"], how="left", validate="1:1")
            assert train[TOPIC].notna().all().all() and test[TOPIC].notna().all().all()
            y_tr = ((train[ycol].astype(int) == 1) & (train[rcol] <= cutoff)).astype(int).to_numpy()
            y_te = (test[ycol].astype(int) == 1).astype(int).to_numpy()
            if y_tr.sum() == 0:
                notes.append(f"{outcome} {T}: no training positives, window skipped")
                continue
            keys = None
            for name, feats in MODELS.items():
                Xtr, Xte = prepare(train, test, feats)
                lr = LogisticRegression(C=1.0, max_iter=5000)
                lr.fit(Xtr, y_tr, sample_weight=train["w"].to_numpy())
                s = lr.decision_function(Xte)
                k = set(zip(test["cik"], test["fyear"]))
                assert keys is None or k == keys, "models must score identical CIK-years"
                keys = k
                scores.append(pd.DataFrame({"outcome": outcome, "model": name, "test_year": T,
                                            "cik": test["cik"].to_numpy(), "fyear": test["fyear"].to_numpy(),
                                            "y": y_te, "w": test["w"].to_numpy(), "score": s}))
                coefs.append(pd.DataFrame({"outcome": outcome, "model": name, "test_year": T,
                                           "feature": ["intercept"] + feats,
                                           "coef": np.r_[lr.intercept_, lr.coef_.ravel()]}))
            notes.append(f"{outcome} {T}: train {len(train)} rows, {int(y_tr.sum())} positives "
                         f"(of {int((train[ycol].astype(int) == 1).sum())} eventual); "
                         f"test {len(test)} rows, {int(y_te.sum())} positives")
    return pd.concat(scores, ignore_index=True), pd.concat(coefs, ignore_index=True), notes


def auc_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outcome, model), g in scores.groupby(["outcome", "model"], sort=False):
        rows.append({"outcome": outcome, "model": model, "test_year": "pooled", "n": len(g),
                     "n_pos": int(g["y"].sum()), "auc": weighted_auc(g["y"], g["score"], g["w"])})
        for T, gy in g.groupby("test_year"):
            rows.append({"outcome": outcome, "model": model, "test_year": str(T), "n": len(gy),
                         "n_pos": int(gy["y"].sum()), "auc": weighted_auc(gy["y"], gy["score"], gy["w"])})
    return pd.DataFrame(rows)


def bootstrap_diffs(scores: pd.DataFrame, n_boot: int = 1000, seed: int = 575) -> pd.DataFrame:
    """Cluster bootstrap by CIK: a CIK drawn m times enters with weight m * w (same as duplicating rows)."""
    rows = []
    for outcome, g in scores.groupby("outcome", sort=False):
        wide = g.pivot_table(index=["cik", "fyear"], columns="model", values="score").reset_index()
        base = g[g["model"] == "FIN"][["cik", "fyear", "y", "w"]]
        wide = wide.merge(base, on=["cik", "fyear"], validate="1:1")
        ciks, inv = np.unique(wide["cik"].to_numpy(), return_inverse=True)
        y, w = wide["y"].to_numpy(), wide["w"].to_numpy()
        rng = np.random.default_rng(seed)
        draws = [np.bincount(rng.integers(0, len(ciks), len(ciks)), minlength=len(ciks)) for _ in range(n_boot)]
        for a, b in PAIRS:
            sa, sb = wide[a].to_numpy(), wide[b].to_numpy()
            point = weighted_auc(y, sb, w) - weighted_auc(y, sa, w)
            diffs = []
            for m in draws:
                wb = w * m[inv]
                keep = wb > 0
                yk = y[keep]
                if yk.min() == yk.max():
                    continue
                diffs.append(roc_auc_score(yk, sb[keep], sample_weight=wb[keep])
                             - roc_auc_score(yk, sa[keep], sample_weight=wb[keep]))
            diffs = np.array(diffs)
            rows.append({"outcome": outcome, "comparison": f"{b} - {a}", "auc_a": weighted_auc(y, sa, w),
                         "auc_b": weighted_auc(y, sb, w), "diff": point,
                         "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                         "share_boot_le0": float((diffs <= 0).mean()), "n_resamples": len(diffs),
                         "n_cik": len(ciks), "n_rows": len(wide), "n_pos": int(y.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-tag", default="")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--drop-warrant-wave", action="store_true")
    args = ap.parse_args()

    df, topics = load(args.topics_tag, args.drop_warrant_wave)
    scores, coefs, notes = run(df, topics)
    print("\n".join(notes), "\n")
    auc = auc_table(scores)
    diffs = bootstrap_diffs(scores)

    RES.mkdir(parents=True, exist_ok=True)
    auc.to_csv(RES / f"auc{args.out_tag}.csv", index=False)
    diffs.to_csv(RES / f"auc_diff{args.out_tag}.csv", index=False)
    coefs.to_csv(RES / f"coefs{args.out_tag}.csv", index=False)
    scores.to_parquet(FEAT / f"oos_scores{args.out_tag}.parquet", index=False)

    pooled = auc[auc["test_year"] == "pooled"].pivot(index="model", columns="outcome", values="auc")
    print("pooled weighted AUC:\n", pooled.reindex(list(MODELS)).round(3).to_string(), "\n")
    yearly = auc[auc["test_year"] != "pooled"].pivot_table(index=["outcome", "model"], columns="test_year",
                                                          values="auc", sort=False)
    print("per-year weighted AUC (NaN = undefined, one class only):\n", yearly.round(3).to_string(), "\n")
    with pd.option_context("display.width", 200):
        print(diffs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
