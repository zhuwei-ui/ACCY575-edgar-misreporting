# ACCY 575 — EDGAR misreporting panel (Module 8)

This repo builds `data/final/panel.parquet`: one row per CIK-year for fiscal years 2011–2021 (76,013 rows). Each row holds:

- the company's original 10-K;
- financial statement items **as first reported** in that 10-K;
- Item 4.02 non-reliance labels (`restated`, `irregularity`, and when each was disclosed);
- for a case-control sample of 17,987 CIK-years, the full 10-K text.

It follows the Module 8 walkthrough. Any place it departs from the walkthrough is listed under [Deviations](#deviations-and-judgment-calls).

## Rebuild from nothing

```bash
uv sync                                    # Python 3.15, edgartools==5.59.1 (pinned)
```

`.env` (gitignored):

```
EDGAR_IDENTITY="Your Name you@school.edu"
EDGAR_RATE_LIMIT_PER_SEC=5
EDGAR_LOCAL_DATA_DIR="<absolute path to repo>/data/raw/edgar_cache"
ANTHROPIC_API_KEY="..."                    # only for steps 10–11
```

Then run the steps in order. Every step is resumable, and EDGAR allows only one process at a time (see `AGENTS.md`).

| # | Command | Output | Time |
|---|---|---|---|
| 1 | `uv run python -m src.build_universe` | `data/interim/tenk_index.parquet` | minutes |
| 2 | `uv run python -u -m src.build_tenk_panel` | `data/interim/tenk_panel.parquet`, `filings.parquet` | ~1 h (16k submissions) |
| 3 | `bash src/download_fsds.sh` | `data/raw/fsds/*.zip` (52 quarters, 4.6 GB) | ~30 min |
| 4 | `uv run python -u -m src.build_financials` | `data/interim/financials.parquet` (includes step-8 XBRL fallback) | ~30 min |
| 5 | `uv run python -m src.check_financials` | spot checks against filings (§6d) | — |
| 6 | `uv run python -m src.check_402_counts` | `data/interim/402_count_check.csv` (§7b) | minutes |
| 7 | `uv run python -u -m src.build_402_events` | `data/interim/events_402.parquet`, `data/raw/402_sections/` | ~1 h |
| 8 | `uv run python -m src.sample_gold` | 30 events to hand-label → `data/labels/402_gold.csv` | — |
| 9 | `uv run python -m src.score_extractors` | extractor scores, `extractor_disagreements.csv` | seconds (cached) |
| 10 | `uv run python -m src.llm_batch_402` | fills `data/interim/llm_cache_402.jsonl` via the Batches API | ~5 min, ~$5 |
| 11 | `uv run python -m src.build_labels` | `data/interim/labels.parquet` | seconds (cached) |
| 12 | `uv run python -u -m src.sample_and_download_10k` | `text_sample.parquet`, `data/raw/tenk_text/*.txt.gz` (1.5 GB) | ~4 h |
| 13 | `uv run python -m src.build_panel` | `data/final/panel.parquet` (validated with pandera) | seconds |

Run `uv run pytest` for the tests: the fiscal-year rules, `period_to_fyear`, the 10-K panel and the extractors.

The LLM model is `claude-haiku-4-5` only. Every response is cached by SHA-256 of (model, prompt version, section text), so re-running steps 9–11 costs nothing.

## Sample definition

- **Universe** (§4): every 10-K and 10-K/A in EDGAR's full index, 2011Q1–2023Q4, covering 16,207 CIKs.
- **CIK-year panel** (§5):
  - One original 10-K per CIK and fiscal year.
  - `fyear` follows the Compustat rule (d = reportDate − 7 days; fyear = d.year if d.month ≥ 6, else d.year − 1).
  - Asset-backed issuers (SIC 6189) are dropped.
  - Duplicates are logged to `tenk_duplicates.parquet`.
  - Flags: `combined` (one 10-K filed by several CIKs) and `late_filer`.
- **Financials** (§6):
  - Taken from the SEC Financial Statement Data Sets and selected **by accession**, so each number is the one that 10-K reported, with no look-ahead.
  - 10-Ks with XBRL that are missing from every `sub.txt` (314 filings) are filled from the filing's own XBRL (`fin_source = "xbrl"`; 18 of them return no XBRL).
  - Coverage of `at` is 0.891 overall. 6,587 panel 10-Ks have no financials, 1,874 of them combined-filing subsidiaries.
- **Item 4.02 events** (§7):
  - 3,974 8-K and 8-K/A filings carrying Item 4.02 form 3,624 events across 2,658 CIKs. An 8-K/A within 365 days of an event joins it.
  - Section text is saved per filing.
  - Yearly counts match `search_filings` to within a few filings (e.g. 2016: 183 vs 184).
- **Text sample** (§8b): see the case-control section below.

## Tag lists (§6c)

The lists are exactly the walkthrough's table; the full lists are in `src/build_financials.py`.

- Each variable takes the first tag in its list that has a value, separately for the current and lag period.
- `sstk` and `dltis` take the maximum across their tags.
- `lt` is derived from `LiabilitiesAndStockholdersEquity − equity − temporary equity` when `Liabilities` is missing (`lt_derived`).
- `pi` is derived from domestic plus foreign pre-tax income when needed.
- When `at` is present, `rect`, `invt`, `pstk`, `sstk` and `dltis` are zero-filled. The zero-filled variables are listed in the `zero_filled` column.

## Labels (§8a)

### Restated-year rule

A CIK-year is `restated = 1` if an Item 4.02 declares unreliable its annual statements or any interim period inside that fiscal year (the walkthrough's rule).

- Period-end dates are mapped to fiscal years with `src.fiscal.period_to_fyear`, against every original 10-K the company filed, using the walkthrough's 10-day slack and 380-day limit.
- `reveal_date` is the event's first filing date. A CIK-year hit by several events keeps the earliest one.
- Each event is read from its **first** filing's section. If that section is empty, the next filing in the event with text is used; this happened once.
- A disclosure cannot withdraw a period that hadn't ended. Extracted period ends after the event's first filing date (17 periods in 12 events, mostly typos or mentions of upcoming filings) are dropped and logged to `labels_implausible.csv`.
- Only fiscal years 2011–2021 are kept.

**Owner's stricter rule.** The hand-labeled gold set counts a year only if the company had **already filed** that year's original 10-K before the 4.02. A year with only quarters restated and its 10-K still to come is not restated, because that 10-K reports corrected numbers.

- In the panel this rule is exactly `restated == 1 & ~revealed_before_10k`.
- The panel keeps the broader label, so those rows aren't sampled as clean controls even though their 10-Ks discuss the restatement.
- `src.extract_402.keep_filed_fyears` applies the stricter rule when scoring against the gold set.

### Irregularity rule

Following Hennes, Leone and Miller (2008), `irregularity` is true when the disclosure mentions fraud or irregularities, or an investigation by the audit committee, the SEC or the DOJ. The LLM must quote its evidence verbatim, and the quote is kept in `event_extractions.parquet`.

- `irregularity_reveal_date` is the earliest first filing date among the events that flag it.

### Extractor scores

The gold set is 30 seeded random events, hand-labeled. One event (731653-1) has an empty saved section and is not scored, leaving n = 29.

| Extractor | Exact match | Year precision | Year recall | Irregularity accuracy |
|---|---|---|---|---|
| Regex baseline | 0.897 | 0.893 | 0.962 | 1.000 |
| `claude-haiku-4-5`, forced tool output | **1.000** | **1.000** | **1.000** | **1.000** |

The regex extractor's remaining errors are dates that are mentioned but not restated ("balances as of December 31, 2022 and 2021", a 10-K called "discussed") and fiscal-year names ("fiscal 2024" for an April year-end).

**Choice:** the LLM extractor. It ran on 3,539 sections through the Message Batches API, with 0 errors, 7.2M input and 0.6M output tokens, for about $5.11.

**Coverage:** 41 events have no usable Item 4.02 text in any filing. Another 31 yield no restated period: most say only "see Item 4.01 above", or are 4.02(b) notices saying the company isn't aware of anything.

## 10-K text: case-control sample (§8b)

- The panel is joined to the labels and sorted by (cik, fyear). Then `u = default_rng(575).random(n)` is drawn once for every CIK-year, and `random20 = u < 0.2`.
- Every labeled CIK-year is kept (`sampled_as = "restated"`, `w = 1`; 3,360 rows). So is every unlabeled CIK-year with `random20` (`"control"`, `w = 5`; 14,627 rows).
- Only the primary document is downloaded and parsed with edgartools' `HTMLParser(ParserConfig(form="10-K"))`. The text is then NFKC-normalized, stripped of box-drawing characters, whitespace-collapsed and gzipped.
- **Result:** all 17,859 unique accessions were downloaded. Seven failed on temporary 503s and succeeded on a retry. The median document is 48,700 words.
- No document starts with an inline-XBRL header. One SPAC 10-K, 0001104659-21-044518, starts at its notes to the financial statements.

## Final panel (§9)

`tenk_panel ← financials ← labels ← text_sample`, validated by `src.schemas.panel_schema`. The checks:

- (cik, fyear) is unique, and an accession can repeat only when at most one of its rows has financials.
- `reveal_date` is present exactly when `restated == 1`. `irregularity_reveal_date` is present only when `irregularity` is true, and never before `reveal_date`.
- `at` is never negative.
- `w` is present exactly when `sampled_as` is, and `revealed_before_10k` is a non-null boolean.

| fyear | CIK-years | `at` share | restated | revealed before 10-K | irregularity | blank check | restated & blank check | with text |
|---|---|---|---|---|---|---|---|---|
| 2011 | 8,116 | 0.802 | 433 | 197 | 53 | 216 | 6 | 2,057 |
| 2012 | 7,728 | 0.870 | 360 | 171 | 57 | 169 | 6 | 1,859 |
| 2013 | 7,540 | 0.886 | 303 | 141 | 60 | 144 | 5 | 1,843 |
| 2014 | 7,389 | 0.895 | 258 | 115 | 49 | 127 | 3 | 1,637 |
| 2015 | 6,982 | 0.902 | 221 | 106 | 33 | 116 | 0 | 1,534 |
| 2016 | 6,612 | 0.898 | 177 | 76 | 31 | 101 | 1 | 1,421 |
| 2017 | 6,369 | 0.906 | 172 | 80 | 24 | 94 | 2 | 1,424 |
| 2018 | 6,167 | 0.915 | 142 | 67 | 20 | 74 | 1 | 1,344 |
| 2019 | 5,976 | 0.918 | 172 | 60 | 20 | 63 | 6 | 1,299 |
| 2020 | 6,237 | 0.911 | 432 | 54 | 28 | 207 | 105 | 1,631 |
| 2021 | 6,897 | 0.928 | 690 | 557 | 44 | 532 | 308 | 1,938 |
| **Total** | **76,013** | **0.891** | **3,360** | **1,624** | **419** | **1,843** | **443** | **17,987** |

**What the table shows**

- Restatements drift down from 2011 to 2018–19, then spike in 2020–21. Of those 1,122 restated rows, 413 are SPACs: the 2021 warrant and Class A share-classification waves.
- 1,624 restated CIK-years were disclosed on or before their own 10-K. Module 9 must exclude these, and they make up 81% of fiscal 2021.
- 100% of restated CIK-years have text.

The panel has 3,360 restated rows, against 3,583 labeled CIK-years; the other 223 have no 10-K in the panel.

## Module 9: replicating Brown, Crowley & Elliott (2020)

Reads only `data/final/panel.parquet` and the 10-K text files, with no network calls. The predictions, results and discussion are in [results.md](results.md).

| # | Command | Output | Time |
|---|---|---|---|
| 1 | `uv run python -m src.features_financial` | `data/features/financial.parquet` | seconds |
| 2 | `uv run python -m src.eval_sample` | `data/features/eval_sample.parquet` (6,213 CIK-years) | seconds |
| 3 | `uv run python -m src.features_style` | `data/features/style.parquet` | ~9 min |
| 4 | `uv run python -m src.topics_dtm` | `data/features/dtm.npz`, `dtm_vocab.txt`, `dtm_rows.parquet` | ~1 min |
| 5 | `uv run python -m src.topics_rolling` | `data/features/topics_{2016..2021}.parquet`, `lda/`, `results/m9/topics/` | ~7 min |
| 6 | `uv run python -m src.evaluate` | `results/m9/auc.csv`, `auc_diff.csv`, `coefs.csv` | ~1 min |

Robustness runs:
- **Stopword threshold:** `topics_rolling --tag _df0025 --min-df-share 0.0025`, then `evaluate --topics-tag _df0025 --out-tag _df0025`.
- **Without the warrant wave:** `evaluate --drop-warrant-wave --out-tag _nowave`.

## Deviations and judgment calls

1. **Wrong period dates entered by filers (`period_mismatch`).** In 186 rows the FSDS balance-sheet date differs from the filer-entered `reportDate` by more than 31 days, and about 40 of these would move to a different fyear. Decision: keep §5's fyear and flag the rows rather than correct them.
2. **Stacked Item 4.02 headings (§7c).** Some 8-Ks list "Item 4.02" with no body directly before another item. The section cutter skips these heading-only hits and falls back to the next occurrence. Fix counts: cut 93, fallback 70, not found 47, stacked heading 6.
3. **Gold-set rule versus panel rule.** See [Restated-year rule](#restated-year-rule). The gold set uses the owner's stricter rule; the panel uses the walkthrough's rule plus `revealed_before_10k`.
4. **Period ends after the disclosure date are dropped** before mapping to fiscal years (§8a). The walkthrough's "log implausible years" step is tightened to this period-level check.
5. **Two negative `at` values** (CIK 1404212 FY2012, −$17; CIK 1625285 FY2018, −$13), which look like shell-company filing errors, are set to missing.
6. **XBRL fallback (step 8)** covers only filings absent from every `sub.txt`. Combined-filing subsidiaries are not filled, because their parent's row carries the numbers.
7. **LLM via the Batches API** instead of one call at a time. The model, prompt, tool and cache key are the same; the price is half.

## Repository rules

See `AGENTS.md`. In short:

- Import `src.sec` before `edgar`.
- Run one EDGAR process at a time.
- Take financials only from FSDS by accession.
- Never use `get_financials()`.
- Use `claude-haiku-4-5` only, and cache every call.
