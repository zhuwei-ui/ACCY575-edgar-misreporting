import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from src.evaluate import prepare, weighted_auc
from src.topics_rolling import window_of, window_stopwords


def test_window_and_cutoff():
    years, cutoff = window_of(2016)
    assert years == [2011, 2012, 2013, 2014, 2015]
    assert cutoff == pd.Timestamp("2016-06-30")


def test_window_stopwords_rules():
    # columns: very common, common, mid, rare-by-count, rare-by-docs, absent
    X = sp.csr_matrix(np.array([
        [50, 20, 3, 0, 1, 0],
        [50, 20, 3, 0, 0, 0],
        [50, 20, 3, 2, 0, 0],
        [50, 20, 3, 0, 0, 0],
    ]))
    keep, info = window_stopwords(X, head_share=0.6, min_count_per_doc=1.0, min_df_share=0.5)
    # total 4*73 + 3 = 295; the first word alone is 200/295 = 68% >= 60%, so only it is removed
    assert info["head_removed"] == 1
    # min count = 1.0 * 4 docs = 4 -> word 3 (2) and word 4 (1) removed; min df = 2 docs -> word 4 removed
    assert keep.tolist() == [False, True, True, False, False, False]


def test_prepare_uses_training_moments_only():
    train = pd.DataFrame({"log_at": [1.0, 2.0, 3.0, 100.0], "t00": [0.1, 0.2, 0.3, 0.4], "w": [1.0, 1, 1, 1]})
    test = pd.DataFrame({"log_at": [1000.0], "t00": [0.25], "w": [1.0]})
    Xtr, Xte = prepare(train, test, ["log_at", "t00"])
    lo, hi = np.percentile(train["log_at"], [1, 99], weights=train["w"], method="inverted_cdf")
    clipped = np.clip(train["log_at"].to_numpy(), lo, hi)
    m, s = clipped.mean(), clipped.std()
    assert Xte[0, 0] == pytest.approx((hi - m) / s)  # test value capped at the training 99th pct
    assert Xtr[:, 1].mean() == pytest.approx(0) and Xtr[:, 1].std() == pytest.approx(1)


def test_weighted_auc_equals_duplicated_rows():
    y = np.array([0, 0, 1, 1, 0])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.5])
    w = np.array([5, 5, 1, 1, 5])
    dup = np.repeat(np.arange(5), w)
    assert weighted_auc(y, s, w) == pytest.approx(roc_auc_score(y[dup], s[dup]))
    assert np.isnan(weighted_auc(np.zeros(3), s[:3], w[:3]))
