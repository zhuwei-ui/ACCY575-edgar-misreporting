# EDGAR rules for this repo

- Every script imports `src.sec` before anything from `edgar`. Never hard-code a User-Agent.
- One EDGAR process at a time. No threads, no async, no parallel downloads.
- Financial statement numbers come from the SEC Financial Statement Data Sets (§6), selected by
  accession. Never use edgartools' `get_financials()`, `Financials`, `XBRLS`, or company-level
  facts without `include_metadata=True, pit_mode=True`: they return the latest (restated) values.
  Never use the `fiscal_year`/`fy` fields to choose a period.
- Look up filings from panel rows with `Filing(cik=int(cik), company="", form=..., filing_date=...,
  accession_no=...)` (`company` is required but only used for display). Never call `find()` or
  `get_by_accession_number()` in a loop.
- `search_filings(forms="8-K", ...)`: forms as a string, never a list (a list silently drops the 8-Ks).
- Never call `Filing.text()`, `.html()`, or `.obj()` on 10-Ks in bulk, or `download_filings()` at all:
  they download whole submissions (every exhibit) or daily archives. For 10-K text, download the
  primary document (§8b). `.obj()` on the Item 4.02 8-Ks (§7c) is fine: 8-K submissions are small.
- Wrap every network call in try/except, log failures to a CSV, and keep going.
- Don't run `edgar.ai.install_skill()`.
- Anthropic API calls use `claude-haiku-4-5` only. Never switch to a more expensive model (Sonnet,
  Opus, etc.) without the owner's explicit approval. Cache every response so re-runs cost nothing.
