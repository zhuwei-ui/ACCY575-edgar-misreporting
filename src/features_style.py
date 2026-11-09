"""M9 §5 — textual style predictors, one row per CIK-year in data/features/eval_sample.parquet.

Reads each 10-K text file (data/raw/tenk_text/*.txt.gz) one at a time and writes
data/features/style.parquet with:

    log_words, parsed_chars            log word count; characters in the parsed text
    sent_len_mean, sent_len_sd         words per sentence (population SD)
    word_len_sd                        characters per word (population SD)
    pct_short_sent, pct_long_sent      share of sentences with <= 30 and >= 60 words
    repetition                         mean times each distinct sentence appears
                                       (sentences / distinct sentences)
    type_token                         unique / total lowercased words in the first 10,000 words
                                       (the paper uses the whole document)
    fog                                0.4 * (words/sentences + 100 * complex/words); complex = 3+
                                       syllables, counted with pyphen (en_US) on the word's letters
    coleman_liau                       5.88 * chars/words - 29.6 * sentences/words - 15.8, chars =
                                       letters and digits in words (no spaces, no punctuation)
    n_allcaps, n_exclaim, n_question   words with 2+ letters all capitals; '!' and '?' characters

Rules, kept simple and deterministic:
- Word: a whitespace-separated token with leading and trailing punctuation (anything not a letter
  or digit) stripped, kept if it contains at least one letter or digit ("2,345" is a word, "$" is not).
- Sentence: text split on '.', '?' or '!' followed by whitespace and a capital letter. The periods of
  these abbreviations are protected first: Inc. Corp. Co. Ltd. No. U.S. e.g. i.e. Mr. Ms. Mrs. Dr.
  Sentences with no words are discarded.
- Syllables: pyphen hyphenation points + 1 on the word's letters only (a word with no letters, such as
  a number, has 1 syllable). Hyphenation dictionaries approximate syllables; pyphen splits fewer
  places than a full syllabifier, so complex-word shares are conservative.

Run: uv run python -m src.features_style [--limit N]
"""

import argparse
import gzip
import math
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyphen

SAMPLE = Path("data/features/eval_sample.parquet")
OUT = Path("data/features/style.parquet")

STYLE = ["log_words", "parsed_chars", "sent_len_mean", "sent_len_sd", "word_len_sd", "pct_short_sent",
         "pct_long_sent", "repetition", "type_token", "fog", "coleman_liau", "n_allcaps", "n_exclaim",
         "n_question"]

ABBREVIATIONS = ["Inc.", "Corp.", "Co.", "Ltd.", "No.", "U.S.", "e.g.", "i.e.", "Mr.", "Ms.", "Mrs.", "Dr."]
_PLACEHOLDER = "\x00"
_ABBR_RE = re.compile("|".join(r"(?<![A-Za-z])" + re.escape(a) for a in ABBREVIATIONS))
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
_STRIP = re.compile(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$")
_ALNUM = re.compile(r"[0-9A-Za-z]")
_LETTERS = re.compile(r"[^A-Za-z]")
_DIC = pyphen.Pyphen(lang="en_US")


def words_of(text: str) -> list[str]:
    out = []
    for tok in text.split():
        w = _STRIP.sub("", tok)
        if w:
            out.append(w)
    return out


def sentences_of(text: str) -> list[str]:
    protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    return [s.replace(_PLACEHOLDER, ".") for s in _SENT_SPLIT.split(protected)]


@lru_cache(maxsize=500_000)
def syllables(word: str) -> int:
    letters = _LETTERS.sub("", word).lower()
    if not letters:
        return 1
    return len(_DIC.positions(letters)) + 1


def n_alnum(word: str) -> int:
    return sum(c.isalnum() for c in word)


def style_features(text: str) -> dict:
    sents = []
    for s in sentences_of(text):
        ws = words_of(s)
        if ws:
            sents.append((s.strip(), ws))
    words = [w for _, ws in sents for w in ws]
    n_w, n_s = len(words), len(sents)
    if n_w == 0 or n_s == 0:
        return {k: np.nan for k in STYLE} | {"parsed_chars": len(text)}
    sent_lens = np.array([len(ws) for _, ws in sents], dtype=float)
    word_lens = np.array([len(w) for w in words], dtype=float)
    complex_words = sum(syllables(w) >= 3 for w in words)
    chars = sum(n_alnum(w) for w in words)
    first = [w.lower() for w in words[:10_000]]
    distinct = len({s for s, _ in sents})
    return {
        "log_words": math.log(n_w),
        "parsed_chars": len(text),
        "sent_len_mean": sent_lens.mean(),
        "sent_len_sd": sent_lens.std(),
        "word_len_sd": word_lens.std(),
        "pct_short_sent": float((sent_lens <= 30).mean()),
        "pct_long_sent": float((sent_lens >= 60).mean()),
        "repetition": n_s / distinct,
        "type_token": len(set(first)) / len(first),
        "fog": 0.4 * (n_w / n_s + 100 * complex_words / n_w),
        "coleman_liau": 5.88 * chars / n_w - 29.6 * n_s / n_w - 15.8,
        "n_allcaps": sum(1 for w in words if w.isupper() and len(_LETTERS.sub("", w)) >= 2),
        "n_exclaim": text.count("!"),
        "n_question": text.count("?"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sample = pd.read_parquet(SAMPLE, columns=["cik", "fyear", "text_path"])
    if args.limit:
        sample = sample.head(args.limit)
    rows, failures = [], []
    t0 = time.time()
    for i, r in enumerate(sample.itertuples(index=False), 1):
        try:
            with gzip.open(r.text_path, "rt", encoding="utf-8", errors="replace") as fh:
                feats = style_features(fh.read())
        except Exception as e:  # keep going; report at the end
            failures.append((r.cik, r.fyear, r.text_path, repr(e)))
            feats = {k: np.nan for k in STYLE}
        rows.append({"cik": r.cik, "fyear": r.fyear, **feats})
        if i % 500 == 0:
            print(f"{i:,}/{len(sample):,} docs, {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)[["cik", "fyear"] + STYLE]
    print(f"done: {len(out):,} docs in {time.time() - t0:.0f}s; failures {len(failures)}")
    for f in failures:
        print("  failed:", f)
    print("missing per feature:", out[STYLE].isna().sum()[lambda s: s > 0].to_dict())
    with pd.option_context("display.width", 200):
        print(out[STYLE].describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())
    if not args.limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
