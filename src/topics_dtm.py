"""M9 §6a — tokenize every evaluation-sample 10-K once into a sparse document-term matrix.

Rows: every CIK-year in data/features/eval_sample.parquet, in that file's order. The topic model in
§6c fits on the window's random20 rows only, and transforms the rest; this matrix just assigns
every word a column. Which columns a window uses is decided inside the window (§6b), so the union
vocabulary is not look-ahead.

CountVectorizer(token_pattern=r"(?u)\\b[a-z]{3,}\\b", lowercase=True, min_df=1, dtype=int32):
alphabetic tokens of 3+ letters. Documents are streamed from disk with a generator.

Writes data/features/dtm.npz, dtm_vocab.txt (one word per line, column order), dtm_rows.parquet
(cik, fyear in row order).

Run: uv run python -m src.topics_dtm
"""

import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

SAMPLE = Path("data/features/eval_sample.parquet")
FEAT = Path("data/features")
TOKEN_PATTERN = r"(?u)\b[a-z]{3,}\b"


def stream(paths):
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
                yield fh.read()
        except Exception as e:  # an unreadable file becomes an empty row; reported
            print(f"failed to read {p}: {e!r}", flush=True)
            yield ""
        if i % 1000 == 0:
            print(f"{i:,}/{len(paths):,} docs, {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    sample = pd.read_parquet(SAMPLE, columns=["cik", "fyear", "text_path"])
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, min_df=1, dtype=np.int32)
    X = vec.fit_transform(stream(list(sample["text_path"]))).tocsr()
    vocab = vec.get_feature_names_out()
    sp.save_npz(FEAT / "dtm.npz", X)
    (FEAT / "dtm_vocab.txt").write_text("\n".join(vocab) + "\n")
    sample[["cik", "fyear"]].to_parquet(FEAT / "dtm_rows.parquet", index=False)
    print(f"DTM shape {X.shape}, non-zeros {X.nnz:,}, total tokens {int(X.sum()):,}")
    empty = int((X.getnnz(axis=1) == 0).sum())
    print(f"empty rows: {empty}")


if __name__ == "__main__":
    main()
