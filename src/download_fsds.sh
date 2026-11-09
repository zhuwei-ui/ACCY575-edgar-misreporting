#!/usr/bin/env bash
# §6b — Download the SEC Financial Statement Data Sets, 2011Q1-2023Q4, into data/raw/fsds/.
# Resumable: finished ZIPs are skipped; a partial download (.part) is retried.
set -u
set -a; source .env; set +a   # EDGAR_IDENTITY for the User-Agent
mkdir -p data/raw/fsds
for y in $(seq 2011 2023); do for q in 1 2 3 4; do
  f=data/raw/fsds/${y}q${q}.zip
  if [ -f "$f" ]; then echo "$f exists"; continue; fi
  if curl -sf -A "$EDGAR_IDENTITY" -o "$f.part" \
      "https://www.sec.gov/files/dera/data/financial-statement-data-sets/${y}q${q}.zip"; then
    mv "$f.part" "$f"; echo "$f $(du -h "$f" | cut -f1)"
  else
    echo "$f FAILED"
  fi
  sleep 1
done; done
echo DONE
