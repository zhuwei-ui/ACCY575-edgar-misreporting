"""§8a — run the LLM extractor over every event's first-filing section via the Message Batches API.

Same model, prompt, and tool as src.extract_402.llm_extract (claude-haiku-4-5 only; AGENTS.md), but
submitted as one batch, which costs half as much as individual calls. Results go into the same JSONL
cache, so build_labels then runs llm_extract entirely from cache. Sections already cached (e.g. the
gold set) and empty sections are not sent.

Resumable: the batch id is saved to data/interim/llm_batch_402.json; re-running polls that batch
instead of submitting a new one. Errored requests are logged to data/interim/llm_batch_failures.csv
and can be re-sent by running again with --new.

Run: uv run python -m src.llm_batch_402 [--new] [--dry-run]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.extract_402 import cache_key, get_client, is_cached, request_params, store_response

EVENTS = Path("data/interim/events_402.parquet")
SECTIONS = Path("data/raw/402_sections")
STATE = Path("data/interim/llm_batch_402.json")
FAILURES = Path("data/interim/llm_batch_failures.csv")


def pending_sections() -> dict[str, str]:
    ev = pd.read_parquet(EVENTS)
    todo = {}
    for accs in ev["accessions"]:
        p = SECTIONS / f"{accs[0]}.txt"
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        if text.strip() and not is_cached(text):
            todo[cache_key(text)] = text  # sha256 hex = 64 chars, the custom_id limit
    return todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true", help="submit a new batch for whatever is still uncached")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = get_client()
    if STATE.exists() and not args.new:
        batch_id = json.loads(STATE.read_text())["batch_id"]
        print(f"resuming batch {batch_id}")
    else:
        todo = pending_sections()
        chars = sum(len(t) for t in todo.values())
        print(f"{len(todo)} sections to send ({chars:,} chars)")
        if args.dry_run or not todo:
            return
        batch = client.messages.batches.create(
            requests=[{"custom_id": k, "params": request_params(t)} for k, t in todo.items()]
        )
        batch_id = batch.id
        STATE.write_text(json.dumps({"batch_id": batch_id, "n": len(todo),
                                     "submitted": pd.Timestamp.now().isoformat()}))
        print(f"submitted batch {batch_id}")

    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"[{(time.time() - t0) / 60:5.1f} min] {b.processing_status}: processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored} expired={c.expired}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(60)

    ok, fails, tin, tout = 0, [], 0, 0
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            try:
                store_response(r.custom_id, r.result.message)
                ok += 1
                tin += r.result.message.usage.input_tokens
                tout += r.result.message.usage.output_tokens
            except StopIteration:
                fails.append({"custom_id": r.custom_id, "error": "no tool_use block"})
        else:
            err = getattr(r.result, "error", None)
            fails.append({"custom_id": r.custom_id, "error": f"{r.result.type}: {err}"})
    # batch price = half of standard Haiku 4.5 ($1/M in, $5/M out)
    print(f"stored {ok}; failed {len(fails)}; tokens in {tin:,} out {tout:,}; "
          f"cost ~${(tin / 1e6 * 1 + tout / 1e6 * 5) / 2:.2f}")
    if fails:
        pd.DataFrame(fails).to_csv(FAILURES, index=False)
        print(f"failures -> {FAILURES}; rerun with --new to resend")
    STATE.unlink()


if __name__ == "__main__":
    main()
