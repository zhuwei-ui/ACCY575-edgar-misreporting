# Module 9 — Replicating Brown, Crowley, and Elliott (2020) on Item 4.02 restatements

## Predictions

*Written and committed before any model was estimated. The only things looked at were the design and the count of positives in the Module 8 panel. No features, topics, or AUCs existed yet.*

The paper finds that 10-K topics add significant out-of-sample detection power over both the financial and the style models (abstract: up to 59%). For each way my design differs from theirs, I predict whether it makes **topic look more or less useful than in the paper**, and why.

### Difference by difference

**1. Period: fiscal 2011–2021 instead of 1994–2012 → less useful.**
Many of the paper's frauds come from the pre-SOX years and the early 2000s, when misreporting firms were often in unusual businesses with unusual disclosures. By 2011, 10-Ks are longer, more standardized, and full of mandated boilerplate (Item 1A risk factors since 2005, ICFR certifications, XBRL-era disclosure templates). Standardization shrinks the differences in content that a topic model can pick up between firms. The share of genuine frauds among restatements is also lower after SOX.

**2. Window year: Compustat `fyear` instead of filing year → no material effect.**
The two differ only for off-calendar year-ends and late filers. The filing-date cutoff in each window removes the one way this could leak future information.

**3. Outcome: all Item 4.02s, plus an irregularity subset, instead of AAERs and irregularity restatements → mixed: more useful for all 4.02s relative to FIN, less detectable for irregularities.**
- *All 4.02s.* Most are honest errors: revenue recognition (ASC 606 adoption falls inside the period), income taxes, cash-flow classification, debt and equity classification, warrants. The paper's argument is that what a company chooses to discuss reveals its incentives. That doesn't apply to honest errors. But honest errors **cluster in businesses with complicated accounting**: biotech with collaboration revenue, software with multiple-element contracts, small companies with complex financing instruments, companies with foreign operations. A topic model sees those business models directly. So I expect TOPIC to predict all 4.02s reasonably well, mostly by **identifying industry and business complexity rather than suspicious disclosure**. The Dechow financial model was built for intentional misstatements (AAERs), so it should do comparatively worse on honest errors, and the increment from adding TOPIC to FIN should be at least as large as in the paper.
- *Irregularities.* The paper's argument fits better here, but there are few positives: about 20–60 irregularity CIK-years per fiscal year with text, and fewer after the evaluation-sample filters. I expect the point estimates for topic's increment to be positive but the confidence intervals to be wide, and to include zero. The keyword-based irregularity label (any mention of an audit-committee investigation counts) also mixes in non-fraud cases, which blurs the signal.

**4. Sample: case-control with sampling weights; prediction on companies with complete XBRL inputs → slightly less useful.**
Weights fix the class balance, so they shouldn't bias AUC. The complete-inputs rule drops shells and very small filers with missing tags, which are the firms where content is most unusual. This removes some of the easiest cases for TOPIC. Excluding `revealed_before_10k` rows removes 81% of fiscal-2021 restated rows, so 2021 will have fewer positives than its raw count suggests.

**5. Financial predictors: about 9 of the 17 → more useful (relative to FIN).**
With no book-to-market, no lagged market-adjusted return, no auditor type, and no lease, M&A, restructuring, or ex ante financing variables, my FIN model is **weaker than theirs by construction**. Some of the dropped variables (market returns, issuance need) are among the stronger Dechow predictors. A weaker benchmark makes TOPIC's increment over FIN, and the TOPIC-vs-FIN comparison, look better than they would against the full model. I expect FIN's pooled AUC to be modest (roughly 0.58–0.65).

**6. Style predictors: reduced set (no tone, no voice, no formatting or paragraph measures) → more useful (relative to STYLE).**
STYLE loses its tone and structure variables. What remains is mostly length and readability, and filing length is largely a proxy for firm size and complexity. I expect STYLE to be the weakest single model and TOPIC's increment over STYLE to be the clearest result.

**7. Topic model: fit on the `random20` sample (a few thousand filings per window, versus tens of thousands), rescaled stopword cutoffs, learning offset 10, `transform` instead of projection → less useful.**
Fewer documents per window means noisier, less stable topics, and more topics may collapse into boilerplate or industry clusters. Whatever topic achieves here is probably a lower bound on what a full-corpus fit would give. The `random20` rule is right for avoiding look-ahead, but it costs topic quality.

**8. Test years: 2016–2021 (six years, versus 14 in the paper) → noisier.**
With 150–450 restated rows per year before filters, and far fewer irregularities, the per-year AUCs will bounce around. The pooled AUC differences will have wide intervals. This doesn't push in either direction, but it lowers the chance that any increment is statistically distinguishable from zero.

**9. The 2020–2021 SPAC wave → would have made topic look much more useful, but the exclusions neutralize it, except for the de-SPACs.**
Blank-check shells would be trivially detectable from vocabulary ("trust account", "business combination"). They are removed twice: SIC 6770 falls in the financial-firm exclusion, and they fail the complete-inputs rule. About 30 de-SPAC warrant restatements in fiscal 2020 remain. Their 10-Ks share vocabulary ("business combination", "warrants", "PIPE"), and they also enter the 2021 training window as positives. I expect them to **raise TOPIC's fiscal-2020 AUC** and possibly to create a "former SPAC" topic with a positive coefficient. That would be a mechanical effect, not evidence that the disclosure itself is suspicious.

### Overall predictions

- **FIN+TOPIC > FIN for all 4.02s**, with an AUC gain of about 0.02–0.06 and a bootstrap interval that excludes zero: **partial replication.**
- **STYLE+TOPIC > STYLE**, with a larger gain and the interval excluding zero.
- **TOPIC alone ≥ FIN alone** for all 4.02s. This is mainly because FIN is handicapped (difference 5) and TOPIC captures industry and complexity (difference 3), not because topics detect intent.
- **For irregularities**, a positive point estimate for TOPIC's increment, but with an interval that includes zero, because of too few positives.
- **I don't expect the paper's "up to 59%" magnitude.** That figure comes from classification tests on AAERs, which I don't run.
- **The topics with the largest coefficients** will mostly be industry or business-model topics (biotech and clinical trials, software and subscription revenue, small companies with going-concern or convertible-financing language, China-based operations, de-SPAC warrants). Topics that look like the paper's "suspicious content" (advertising, franchising growth, strategic alliances) will be rarer.
- **Without the shell exclusions**, TOPIC's 2020–2021 AUC would jump sharply, because "this company is a shell" is easy to read from the text.

---

## Build notes

Scripts, in run order: `src/features_financial.py` → `src/eval_sample.py` → `src/features_style.py` → `src/topics_dtm.py` → `src/topics_rolling.py` → `src/evaluate.py`. Tests: `tests/test_features_financial.py` (RSST identity, cash-sales lag), `tests/test_features_style.py` (hand-checked Fog and Coleman-Liau), `tests/test_m9_pipeline.py` (stopword rules, training-only winsorizing and standardizing, weighted AUC).

### Evaluation sample (§4)

| Dropped because (first failed filter) | CIK-years | of which restated |
|---|---|---|
| no 10-K text (not in the Module 8 text sample) | 58,026 | 0 |
| a §4 financial variable is missing (incl. 953 rows with `at` or `at_lag` ≤ 0) | 10,686 | 2,169 |
| restatement revealed on or before the 10-K | 550 | 550 |
| financial firm (SIC 6000–6799) | 538 | 59 |
| **kept** | **6,213** | **582** (169 irregularities) |

- **The complete-inputs rule is the binding constraint.** It removes 58% of the non-SPAC text rows. The main gaps are pretax income (`pi`, missing in about 61% of dropped rows), `sale`, and `ppent`: small companies often tag only net income, and pre-revenue companies tag no revenue. The prediction sample therefore leans toward larger companies that report revenue.
- `ch_cash_sales` fell back to `sale_lag` for 19.2% of rows with cash sales (4,482 of them in fiscal 2011, which has no prior row). One kept row has no SIC code.
- **SPACs.** `blank_check` is zero in the evaluation sample (asserted). Of the 1,122 restated fiscal 2020–2021 CIK-years with text, the filters removed 1,016: 908 for missing financials (all 413 blank-check shells among them), 97 as revealed before the 10-K, and 11 as financial firms.
- **Warrant wave.** 41 restated rows in the sample have a `reveal_date` between 2021-04-12 and 2021-06-30: 26 in fiscal 2020, 12 in 2019, and 3 in 2018.

Test-year rows and positives (the out-of-sample evaluation):

| fyear | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | pooled |
|---|---|---|---|---|---|---|---|
| rows | 552 | 600 | 507 | 466 | 515 | 547 | 3,187 |
| restated | 43 | 47 | 36 | 37 | 57 | 49 | 269 |
| irregularity | 17 | 14 | 9 | 5 | 5 | 10 | 60 |

The irregularity years 2018–2020 have 5–9 positives each, so their per-year AUCs are noise. Only the pooled irregularity number is worth reading, and even it rests on 60 positives.

### Style (§5)

- Implemented as briefed. Syllables are counted with `pyphen` (en_US hyphenation points + 1, letters only). Hyphenation dictionaries split fewer places than true syllables ("revenue" counts as 2), so complex-word shares and Fog are slightly conservative.
- `type_token` uses the first 10,000 words; the paper uses the whole document.
- The sentence splitter protects Inc., Corp., Co., Ltd., No., U.S., e.g., i.e., Mr., Ms., Mrs., Dr.
- Results: 6,213 documents and no missing values. The median 10-K has about 50,000 words, Fog 21.7 and Coleman-Liau 15.1.

### Topics (§6)

- **Window stopwords**, computed on each window's fit rows only:
  - the most frequent words until 60% of occurrences are gone (about 315 words);
  - words with fewer than 0.0275 × n_docs occurrences (the paper's 1,100 over roughly 40,000 10-Ks per window; about 75 here);
  - words in fewer than 1% of the window's filings (about 27).
  
  About 13,000–13,700 words survive per window.
- **Fit rows:** 2,454–2,760 `random20` filings per window. Between 11 and 17 window rows per window were filed after the cutoff and excluded from both the LDA fit and the regressions.
- **Convergence:** perplexity is 2,619–2,706. In each window, 23–29 of the 31 topics carry more than 1% of document share. No run-together XBRL names survived; the words flagged by the check are `member`, `gaap`, and `fasb`, plus `dei` in 2021, which is diversity, equity and inclusion from the new human-capital disclosures.
- **Topic content:** the topics read as industries and recognizable 10-K themes: clinical trials and FDA, oil and gas reserves, China-based companies, retail stores, restaurants and franchising, gaming, utilities, airlines, coal and MLPs, software and cloud, and small-company financing (preferred stock, convertibles, warrants, related-party officers). Top words for every window are in `results/m9/topics/`.
- **Runtime:** about 70 seconds per window.

## Results

### Pooled out-of-sample AUC, 2016–2021 (weighted by `w`)

| Model | All 4.02s (269 positives) | Irregularities (60 positives) |
|---|---|---|
| FIN | 0.603 | 0.584 |
| STYLE | 0.524 | 0.554 |
| TOPIC | **0.628** | 0.592 |
| FIN+STYLE | 0.590 | 0.583 |
| FIN+TOPIC | 0.621 | **0.599** |
| STYLE+TOPIC | 0.613 | 0.596 |
| FIN+STYLE+TOPIC | 0.612 | 0.594 |

### AUC differences, cluster bootstrap by CIK (1,000 resamples, 2,169 CIKs; all resamples usable)

| Comparison | All 4.02s: diff [95% CI] | Irregularities: diff [95% CI] |
|---|---|---|
| FIN+TOPIC − FIN | +0.018 [−0.017, +0.055] | +0.016 [−0.033, +0.062] |
| STYLE+TOPIC − STYLE | **+0.090 [+0.046, +0.141]** | +0.042 [−0.024, +0.108] |
| TOPIC − FIN | +0.025 [−0.027, +0.081] | +0.008 [−0.078, +0.093] |
| FIN+STYLE+TOPIC − FIN+STYLE | +0.023 [−0.010, +0.058] | +0.011 [−0.034, +0.057] |

### Per-year AUC, all 4.02s

| Model | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|---|
| FIN | 0.638 | 0.559 | 0.523 | 0.631 | 0.631 | 0.700 |
| STYLE | 0.541 | 0.533 | 0.480 | 0.583 | 0.570 | 0.523 |
| TOPIC | 0.574 | 0.616 | 0.608 | 0.637 | 0.674 | 0.679 |
| FIN+TOPIC | 0.608 | 0.582 | 0.566 | 0.642 | 0.669 | 0.686 |

The per-year irregularity AUCs range from 0.38 to 0.90 for the same model across years, which is what 5–17 positives a year produces. They are in `results/m9/auc.csv` and are not interpreted here.

All AUCs are modest. Every model scores in the 0.52–0.63 range. The paper's AUCs are higher, partly because its financial model includes market variables and partly because AAERs and irregularity restatements are more extreme events than the typical 4.02.

## Discussion

### 1. Did I replicate the headline? Partially.

- **Over STYLE: yes, for all 4.02s.** STYLE+TOPIC beats STYLE by 0.090 AUC, with a bootstrap interval of [+0.046, +0.141], which excludes zero. It survives both robustness checks below. For irregularities the gain is +0.042, but the interval [−0.024, +0.108] includes zero.
- **Over FIN: no, not significantly.** FIN+TOPIC beats FIN by 0.018 for all 4.02s, interval [−0.017, +0.055]. For irregularities the gain is +0.016, interval [−0.033, +0.062]. Adding TOPIC to FIN+STYLE gives +0.023 [−0.010, +0.058]. The point estimates are positive throughout, but none of the intervals excludes zero, and the robustness checks show that even the point estimate over FIN is fragile.

TOPIC alone has the highest pooled AUC for all 4.02s (0.628), and it beats FIN in four of the six years. The paper's broad claim is that what a 10-K discusses carries information about misreporting. That claim survives here in a weaker form: topics carry about as much information as my reduced financial model, but they don't add a reliable increment to it.

### 2. Which Predictions held?

| # | Prediction | Outcome |
|---|---|---|
| 1 | Later, more standardized period makes topic **less** useful | Consistent: increments are small. This is not separately testable. |
| 3 | All 4.02s: topic works by recognizing industry and business complexity; increment over FIN at least as large as the paper's | Mechanism **held**: the predictive topics are industries and financing structures (Q4). The size **did not hold**: the increment over FIN is small and not significant. |
| 3 | Irregularities: positive point estimates, intervals that include zero | **Held** for every comparison. |
| 4 | Complete-inputs rule removes easy cases | **Held, and larger than I expected**: 58% of non-SPAC text rows are dropped, mainly small and pre-revenue companies. That likely removed exactly the firms whose topic mix is most distinctive. |
| 5 | Handicapped FIN makes topic look **more** useful; FIN AUC about 0.58–0.65 | FIN = 0.603, inside the range. TOPIC ≥ FIN **held** (0.628 vs 0.603), but not significantly. |
| 6 | STYLE weakest, and topic's increment over STYLE the clearest | **Held**: STYLE = 0.524, close to chance, and +0.090 is the only significant increment. |
| 7 | Small `random20` fits make topic **less** useful | Can't be isolated without a full-sample fit. Most topics did converge to recognizable themes. |
| 8 | Few test years means wide intervals | **Held**: the intervals for the FIN comparisons span about 0.07 (all 4.02s) to 0.17 (irregularities). |
| 9 | De-SPAC warrant restatements inflate TOPIC in fiscal 2020 | **Held, and it matters more than I expected.** Dropping the 41 wave rows lowers TOPIC's 2020 AUC from 0.674 to 0.638, raises FIN's from 0.631 to 0.706, and turns the pooled FIN+TOPIC − FIN increment from +0.018 to **−0.019**. |
| — | Overall: partial replication, significant increment over FIN | **Wrong on the FIN part.** I expected the handicapped FIN benchmark to leave room for topic. Instead, topic and FIN appear to capture overlapping information, mainly size and business type. |

The main lesson: I expected the weak financial benchmark to favor topic. What it did instead is leave both models predicting mostly *who* restates (small, financing-dependent companies with soft assets), and that is information both models can see.

### 3. All restatements vs. irregularities

- **Topic is not more useful for irregularities in this sample.** The increments are similar in size (+0.016 vs +0.018 over FIN) and much noisier. The paper's argument is that the topic mix reveals incentives and should therefore matter more for intentional misreporting. That argument is plausible, but 60 out-of-sample positives can't test it. With this label, an irregularity is any 4.02 that mentions an investigation, including many routine audit-committee reviews. That further dilutes the "intent" signal.
- **Why the financial-firm exclusion matters more in 2020–2021.** Without it, about 400 blank-check shells would be fiscal 2020–2021 positives. "This 10-K is about a trust account and a pending business combination" is trivially detectable from vocabulary, and it would have sent TOPIC's 2020–2021 AUC far above anything the paper saw. That would be a finding about SPAC accounting under the April 2021 SEC staff statement, not about disclosure.
- **The de-SPACs that remained (fiscal 2020) did leak a version of this.** They load on the "preferred / convertible / warrants" topic, and they are what lifts TOPIC above FIN in 2020. Robustness check B below shows the increment over FIN depends on them.

### 4. What are the predictive topics?

Coefficients are from FIN+TOPIC on standardized topic proportions. The labels are mine.

| Window → test year | Largest positive (restated more) | Largest negative (restated less) |
|---|---|---|
| 2011–15 → 2016, all 4.02s | Oil & gas reserves (+0.17); China-based companies (+0.16); small-company financing: preferred, warrants, convertibles, officers (+0.14); equipment leasing and fleets (+0.13) | Industrial materials and parts (−0.24); semiconductors and devices (−0.20); auto dealers (−0.18) |
| 2014–18 → 2019, all 4.02s | Small-company governance: committee, director, officers, related parties (+0.35); China-based companies (+0.19); coal mining (+0.16); airlines and transport (+0.14) | Retail stores (−0.75); large-company pension and restructuring (−0.14); medical devices (−0.11) |
| 2016–20 → 2021, all 4.02s | Preferred / convertible / warrants financing (+0.42); ASU adoption and inventory accounting (+0.14); software and cloud subscriptions (+0.14); construction projects and joint ventures (+0.13) | Drug development and FDA (−0.63); MLPs and coal partnerships (−0.46); semiconductors (−0.22) |
| 2016–20 → 2021, irregularities | Large-company pension and restructuring (+0.47); preferred / convertible / warrants (+0.31); projects and joint ventures (+0.26); software (+0.21) | Drug development and FDA (−0.50); digital media and advertising (−0.32); China-based companies (−0.28) |

- **Comparison with the paper.** The paper reports that firms with irregularity restatements discuss advertising, securitized or guaranteed securities, strategic alliances, and franchise growth more, and discuss mergers, foreign-country risk, hedging, legal proceedings, credit arrangements, and stock-option plans less.
  - Little of that pattern appears here. One weak echo: joint ventures and projects carry a positive sign in 2021, which resembles the paper's "strategic alliances."
  - Advertising and media goes the *opposite* way in 2021 (negative).
  - China-based companies (foreign-country risk) are positive for all 4.02s in 2016 and 2019, also opposite to the paper's sign.
- **What the predictive topics are.** The most consistent positive is a **"small, financing-dependent company"** topic: preferred stock, convertibles, warrants, officer employment agreements, and audit-committee language, with large coefficients in every window. It picks out small issuers with complex equity instruments, which is the business model where 4.02s concentrate (debt-versus-equity classification errors, and the 2021 warrant restatements). That is a finding about **who has hard accounting**, not about **suspicious disclosure**.
- **Signs flip across windows.** The rest are industry topics whose signs change from window to window (oil and gas positive in 2016, coal and MLPs negative in 2021). This looks like the paper's own observation that some topics matter in most years without a consistent sign, but here it applies to most topics, not only income performance.

### 5. Robustness checks

**A. Stopword thresholds** (`--min-df-share 0.0025`, the paper's 100 filings scaled to my windows, instead of 1%). This keeps about 16,000 words per window instead of about 13,000.

| | Main (1%) | Paper-proportional (0.25%) |
|---|---|---|
| TOPIC, all 4.02s | 0.628 | 0.612 |
| FIN+TOPIC − FIN, all 4.02s | +0.018 [−0.017, +0.055] | +0.006 [−0.030, +0.045] |
| STYLE+TOPIC − STYLE, all 4.02s | +0.090 [+0.046, +0.141] | +0.074 [+0.033, +0.120] |
| FIN+TOPIC − FIN, irregularities | +0.016 [−0.033, +0.062] | +0.023 [−0.038, +0.081] |

The qualitative answer doesn't change. Topic adds significantly over STYLE and not over FIN. Keeping rarer words makes TOPIC slightly worse for all 4.02s and slightly better for irregularities, and the per-year values move by up to ±0.08 (2019–2020). That is a reminder that topic-model choices that look harmless move individual-year results noticeably.

**B. Drop the 41 warrant-wave positives** (`reveal_date` between 2021-04-12 and 2021-06-30), from both training and testing.

| | Main | Without warrant wave |
|---|---|---|
| FIN, all 4.02s | 0.603 | 0.612 |
| TOPIC, all 4.02s | 0.628 | 0.593 |
| FIN+TOPIC − FIN, all 4.02s | +0.018 [−0.017, +0.055] | **−0.019 [−0.057, +0.017]** |
| STYLE+TOPIC − STYLE, all 4.02s | +0.090 [+0.046, +0.141] | +0.048 [+0.001, +0.092] |

This is the more informative check. The 41 mechanical warrant restatements (7% of restated rows in the sample) are enough to move TOPIC from above FIN to below it. Without them, topic's increment over the financial model is a precisely estimated zero to slightly negative. Its increment over STYLE survives, but halves and barely clears zero.

## Bottom line

- **Replicated, partially:** the topic mix of a 10-K predicts Item 4.02 restatements out of sample about as well as a reduced Dechow financial model (pooled AUC 0.63 vs 0.60), and clearly better than surface style measures.
- **Not replicated:** the paper's central claim that topic adds detection power on top of the financial model. The increment is +0.018 and not significant, and it disappears once 41 mechanical de-SPAC warrant restatements are removed.
- **Why the differences could explain it:**
  - My outcome is mostly honest errors, not frauds.
  - My period is post-SOX, with standardized 10-Ks.
  - My prediction sample excludes the smallest firms.
  - My topics are fit on about 2,700 filings per window, not tens of thousands.
  - The predictive topics mainly encode business type and financing structure, information a financial model already partly carries.
  
  Irregularities, the closer analogue of the paper's outcome, have too few positives (60) to confirm or reject the paper's result.
