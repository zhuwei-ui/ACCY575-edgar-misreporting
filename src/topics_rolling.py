"""M9 §6b-c — LDA topic proportions in rolling five-year windows.

For each test fiscal year T in 2016-2021 (window = fiscal years T-5..T-1, cutoff = T-06-30):
  1. Fit rows: DTM rows with fyear in the window, filing_date <= cutoff, and random20 (a 20% sample
     drawn without looking at labels; see walkthrough §6c for why not every row).
  2. Window stopwords, from the fit rows only (§6b):
       (1) the most frequent words, removed in order of frequency until --head-share (0.60) of all
           word occurrences in the window are gone;
       (2) words occurring fewer than --min-count-per-doc x n_fit_docs times (default 0.0275: the
           paper's 1,100 occurrences over roughly 40,000 10-Ks per window);
       (3) words in fewer than --min-df-share of the fit documents (default 0.01; the paper's 100
           filings is about 0.0025 of its windows, used as a robustness check).
  3. Fit LatentDirichletAllocation with the paper's settings except learning_offset (10, not 1024),
     on the fit rows shuffled with a seeded RNG.
  4. Transform every window row and every test-year row (same columns). transform rows sum to 1.

Writes, per window: data/features/lda{tag}/lda_{T}.joblib (model), vocab_{T}.txt (kept columns),
data/features/topics{tag}_{T}.parquet (cik, fyear, role in {window, test}, t00..t30), and
results/m9/topics{tag}/top_words_{T}.csv (top 15 words per topic, with mean share).

Run: uv run python -m src.topics_rolling [--windows 2016 2017] [--tag _df0025 --min-df-share 0.0025]
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation

FEAT = Path("data/features")
TEST_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
N_TOPICS = 31


def window_of(test_year: int) -> tuple[list[int], pd.Timestamp]:
    return list(range(test_year - 5, test_year)), pd.Timestamp(f"{test_year}-06-30")


def topic_cols(k: int = N_TOPICS) -> list[str]:
    return [f"t{i:02d}" for i in range(k)]


def window_stopwords(X: sp.csr_matrix, head_share: float, min_count_per_doc: float,
                     min_df_share: float) -> tuple[np.ndarray, dict]:
    """Return (boolean mask of kept columns, counts of words removed by each rule), from X only."""
    n_docs = X.shape[0]
    counts = np.asarray(X.sum(axis=0)).ravel()
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    present = counts > 0
    order = np.argsort(-counts, kind="stable")
    cum = np.cumsum(counts[order])
    # remove the head until at least head_share of all occurrences are gone
    n_head = int(np.searchsorted(cum, head_share * counts.sum(), side="left")) + 1
    head = np.zeros_like(present)
    head[order[:n_head]] = True
    rare_count = present & (counts < min_count_per_doc * n_docs)
    rare_df = present & (df < min_df_share * n_docs)
    keep = present & ~head & ~rare_count & ~rare_df
    info = {"n_docs": n_docs, "words_present": int(present.sum()), "head_removed": int(head.sum()),
            "head_share_removed": float(counts[head].sum() / counts.sum()),
            "min_count": float(min_count_per_doc * n_docs), "rare_count_removed": int(rare_count.sum()),
            "min_df": float(min_df_share * n_docs), "rare_df_removed": int(rare_df.sum()),
            "kept": int(keep.sum())}
    return keep, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="*", default=TEST_YEARS, help="test years to run")
    ap.add_argument("--tag", default="", help="suffix for output names, for robustness runs")
    ap.add_argument("--head-share", type=float, default=0.60)
    ap.add_argument("--min-count-per-doc", type=float, default=0.0275)
    ap.add_argument("--min-df-share", type=float, default=0.01)
    ap.add_argument("--n-topics", type=int, default=N_TOPICS)
    args = ap.parse_args()

    X = sp.load_npz(FEAT / "dtm.npz").tocsr()
    vocab = np.array((FEAT / "dtm_vocab.txt").read_text().splitlines())
    rows = pd.read_parquet(FEAT / "dtm_rows.parquet")
    sample = pd.read_parquet(FEAT / "eval_sample.parquet", columns=["cik", "fyear", "filing_date", "random20"])
    assert (rows[["cik", "fyear"]].values == sample[["cik", "fyear"]].values).all(), "DTM rows out of order"
    assert X.shape == (len(rows), len(vocab))

    model_dir = FEAT / f"lda{args.tag}"
    words_dir = Path("results/m9") / f"topics{args.tag}"
    model_dir.mkdir(parents=True, exist_ok=True)
    words_dir.mkdir(parents=True, exist_ok=True)
    cols = topic_cols(args.n_topics)
    log = []

    for T in args.windows:
        t0 = time.time()
        years, cutoff = window_of(T)
        in_win = sample["fyear"].isin(years).to_numpy()
        late = in_win & (sample["filing_date"] > cutoff).to_numpy()
        fit_mask = in_win & ~late & sample["random20"].to_numpy()
        test_mask = (sample["fyear"] == T).to_numpy()
        fit_idx = np.flatnonzero(fit_mask)
        keep, info = window_stopwords(X[fit_idx], args.head_share, args.min_count_per_doc, args.min_df_share)
        kept_cols = np.flatnonzero(keep)
        rng = np.random.default_rng(575)
        shuffled = rng.permutation(fit_idx)
        Xfit = X[shuffled][:, kept_cols]

        lda = LatentDirichletAllocation(
            n_components=args.n_topics, learning_method="online", batch_size=100, learning_offset=10,
            learning_decay=0.7, doc_topic_prior=0.05, topic_word_prior=0.05, max_iter=10,
            random_state=575, n_jobs=1)
        lda.fit(Xfit)
        perplexity = lda.perplexity(Xfit)

        out_idx = np.flatnonzero(in_win | test_mask)
        theta = lda.transform(X[out_idx][:, kept_cols])
        props = pd.DataFrame(theta, columns=cols)
        props.insert(0, "role", np.where(test_mask[out_idx], "test", "window"))
        props.insert(0, "fyear", sample["fyear"].to_numpy()[out_idx])
        props.insert(0, "cik", sample["cik"].to_numpy()[out_idx])
        props.to_parquet(FEAT / f"topics{args.tag}_{T}.parquet", index=False)

        theta_fit = lda.transform(Xfit)
        mean_share = theta_fit.mean(axis=0)
        comp = lda.components_
        kept_vocab = vocab[kept_cols]
        top = pd.DataFrame({
            "topic": cols,
            "mean_share": mean_share.round(4),
            "top_words": [" ".join(kept_vocab[np.argsort(-comp[k])[:15]]) for k in range(args.n_topics)],
        })
        top.to_csv(words_dir / f"top_words_{T}.csv", index=False)
        joblib.dump(lda, model_dir / f"lda_{T}.joblib")
        (model_dir / f"vocab_{T}.txt").write_text("\n".join(kept_vocab) + "\n")

        member = [w for w in kept_vocab if w.endswith("member") or w in {"srt", "gaap", "xbrli", "dei", "fasb"}]
        secs = time.time() - t0
        rec = {"test_year": T, "window": f"{years[0]}-{years[-1]}", "cutoff": cutoff.date(),
               "window_rows": int(in_win.sum()), "filed_after_cutoff": int(late.sum()),
               "fit_rows": len(fit_idx), "test_rows": int(test_mask.sum()), **info,
               "perplexity": round(float(perplexity), 1), "topics_mean_gt_1pct": int((mean_share > 0.01).sum()),
               "xbrl_like_words": len(member), "seconds": round(secs)}
        log.append(rec)
        print(rec, flush=True)
        if member:
            print("  XBRL-like words kept:", member[:20], flush=True)
        print(top.to_string(index=False, max_colwidth=110), "\n", flush=True)

    pd.DataFrame(log).to_csv(words_dir / f"windows_log_{'_'.join(map(str, args.windows))}.csv", index=False)


if __name__ == "__main__":
    main()
