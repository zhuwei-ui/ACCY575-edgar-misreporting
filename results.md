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
