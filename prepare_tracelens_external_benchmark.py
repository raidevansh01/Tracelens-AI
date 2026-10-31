"""Download a separate, labeled AI-vs-human essay evaluation sample for TraceLens.

Version 2 fixes Hugging Face split metadata that reports zero rows for nonempty splits.
Consult /size and verify /rows instead of trusting the /splits count.


Source: andythetechnerd03/AI-human-text, Hugging Face, TEST split only.
Source labels: generated=0 (human); generated=1 (AI).
Output: datasets/independent_test.csv (text,label,group_id) and provenance JSON.

This is an *external evaluation benchmark*, NOT new training data.
It excludes exact/normalized training duplicates but cannot guarantee that
all source documents are independently authored, non-overlapping, or current.

Windows, run from TraceLens folder:
  .\\.venv\\Scripts\\python.exe prepare_tracelens_external_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "andythetechnerd03/AI-human-text"
BASE = "https://datasets-server.huggingface.co"
SOURCE = "https://huggingface.co/datasets/andythetechnerd03/AI-human-text"
USER_AGENT = "TraceLens-Evaluation-Dataset-Builder/1.0 (academic research; public HF Dataset Viewer API)"
TOKEN_RE = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)


def tokens(s):
    return TOKEN_RE.findall(s)


def signature(s):
    """Normalized fingerprint to catch inconsequential whitespace/case differences."""
    return hashlib.sha256(" ".join(w.casefold() for w in tokens(s)).encode("utf-8")).hexdigest()


def get_json(path, params, *, fetch=None):
    if fetch is not None:
        return fetch(path, params)
    url = BASE + path + "?" + urlencode(params)
    last = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(request, timeout=30) as response:
                body = response.read(6_000_000)
            return json.loads(body)
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            last = error
            if isinstance(error, HTTPError) and error.code in (400, 401, 403, 404):
                break
            if attempt < 3:
                time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"Dataset download failed at {path}: {last}. "
                       "Verify internet access and https://datasets-server.huggingface.co; "
                       "if rate-limited (429), retry later.")


def training_signatures(path):
    known = set()
    if not path.is_file():
        print("WARNING: training CSV absent; overlap with training data CANNOT be checked.", flush=True)
        return known
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("text", "").strip():
                known.add(signature(row["text"]))
    print(f"Training overlap check loaded {len(known)} training fingerprints.", flush=True)
    return known


def discover_test_rows(config, original_split_count, *, fetch=None):
    """Determine test split size with real row verification, never using train data.

    /splits may contain num_rows=0 even when the source's test data exists.
    /size contains per-split sizes; /rows exposes num_rows_total.
    On services without usable counts, scan actual /rows pages sequentially.
    """
    size = max(0, int(original_split_count or 0))
    origin = "/splits" if size else None
    if not size:
        try:
            info = get_json("/size", {"dataset": DATASET}, fetch=fetch)
            for entry in info.get("size", {}).get("splits", []):
                if entry.get("config") == config and entry.get("split") == "test":
                    candidate = int(entry.get("num_rows") or 0)
                    if candidate > 0:
                        size, origin = candidate, "/size"
                        break
        except RuntimeError as exc:
            print(f"WARNING: /size lookup failed: {exc}", flush=True)

    params = {"dataset": DATASET, "config": config, "split": "test", "offset": 0, "length": 100}
    try:
        first_page = get_json("/rows", params, fetch=fetch)
    except RuntimeError as exc:
        raise RuntimeError(
            "The split is advertised but the Hugging Face /rows endpoint is not serving it. "
            f"Underlying error: {exc}. No dataset was created. "
            "Check the dataset's test split in the public viewer, then retry."
        ) from exc
    if not isinstance(first_page, dict) or not isinstance(first_page.get("rows"), list):
        raise RuntimeError("Invalid /rows API response. No evaluation data was written.")
    actual = first_page["rows"]
    if not actual:
        raise RuntimeError("The live /rows endpoint returned NO test documents; this is not merely a /splits metadata bug. "
                           "No CSV was written, and the training split will NOT be substituted.")
    from_rows = int(first_page.get("num_rows_total") or 0)
    if from_rows > 0:
        # /rows is the most relevant live count; use it even if /size disagrees.
        if size and size != from_rows:
            print(f"WARNING: size metadata differs ({size}) from actual /rows total ({from_rows}); using /rows.", flush=True)
        size, origin = from_rows, "/rows"
    elif not size:
        origin = "sequential actual /rows pages (unknown total)"
    if size and size < 2:
        raise RuntimeError(f"Live test split has only {size} rows. Not enough for a balanced benchmark.")
    print(f"Verified nonempty TEST rows. Count: {size if size else 'unknown'}; source: {origin}", flush=True)
    return size, origin, first_page


def build(*, root, per_class=60, seed=20261029, max_pages=55, fetch=None):
    root = Path(root)
    if not (root / "app.py").exists():
        raise ValueError(f"Project root not found: {root}. Place script beside app.py and run from there.")
    if per_class < 1 or per_class > 500:
        raise ValueError("--per-class must be between 1 and 500")
    if max_pages < 1 or max_pages > 300:
        raise ValueError("--max-pages must be between 1 and 300")
    output = root / "datasets" / "independent_test.csv"
    manifest_path = root / "datasets" / "independent_test_provenance.json"
    training_path = root / "datasets" / "ai_writing.csv"
    training = training_signatures(training_path)

    print("Looking up the public dataset TEST split ...", flush=True)
    splits = get_json("/splits", {"dataset": DATASET}, fetch=fetch).get("splits", [])
    candidates = [s for s in splits if s.get("split") == "test"]
    if not candidates:
        raise ValueError("Dataset no longer exposes a test split. No evaluation CSV was written.")
    chosen = next((x for x in candidates if x.get("config") == "default"), candidates[0])
    config = chosen["config"]
    size, count_origin, first_page = discover_test_rows(config, chosen.get("num_rows"), fetch=fetch)
    if size and size < per_class * 2:
        raise RuntimeError(f"Only {size} test rows exist; cannot obtain {per_class} per class.")
    print(f"Source: {DATASET} / {config} / test ({size if size else 'unknown number of'} rows)", flush=True)

    # Spread pulls throughout the official test split when the count is known.
    # For servers that omit the count, read successive actual test pages instead.
    # Request no more than 100 rows per call. NEVER switch to train split.
    pages = list(range(0, size, 100)) if size else list(range(0, max_pages * 100, 100))
    rng = random.Random(seed)
    if size:
        rng.shuffle(pages)
    want = {"0": per_class, "1": per_class}
    chosen_rows = {"0": [], "1": []}
    seen = set(training)
    seen_prefix = set()  # drop within-eval prefix-near-duplicates
    rejected = Counter()
    page_attempts = 0

    for offset in pages[:max_pages]:
        if offset == 0:
            response = first_page  # live-verified; do not fetch the same page twice
        else:
            response = get_json("/rows", {"dataset": DATASET, "config": config, "split": "test", "offset": offset,
                                           "length": min(100, size-offset) if size else 100}, fetch=fetch)
        page_rows = response.get("rows", [])
        if not isinstance(page_rows, list):
            raise RuntimeError(f"Bad /rows response for offset={offset}; no benchmark was written.")
        if not page_rows:
            print(f"Reached end of live test rows at offset={offset}.", flush=True)
            break
        for sample in page_rows:
            raw = sample.get("row", {})
            text = raw.get("text")
            label = raw.get("generated")
            if label is True:
                label = 1
            elif label is False:
                label = 0
            if not isinstance(text, str) or label not in (0, 1):
                rejected["invalid_label_or_text"] += 1
                continue
            label = str(label)  # Source convention identical to TraceLens.
            if len(chosen_rows[label]) >= want[label]:
                continue
            content = text.strip()
            wordlist = tokens(content)
            if not (120 <= len(wordlist) <= 1500):
                rejected["word_length_outside_120_1500"] += 1
                continue
            s = signature(content)
            if s in seen:
                rejected["duplicate_or_training_overlap"] += 1
                continue
            # Catch near duplicates with identical first 40 normalized words.
            prefix = tuple(w.casefold() for w in wordlist[:40])
            if prefix in seen_prefix:
                rejected["near_duplicate_prefix"] += 1
                continue
            seen.add(s)
            seen_prefix.add(prefix)
            index = sample.get("row_idx", offset)
            chosen_rows[label].append({"text": content, "label": label,
                                        "group_id": f"external-essay-test-{index}"})
        page_attempts += 1
        print(f"Page {page_attempts}: human {len(chosen_rows['0'])}/{per_class} | AI {len(chosen_rows['1'])}/{per_class}", flush=True)
        if all(len(chosen_rows[k]) >= per_class for k in ('0', '1')):
            break

    if not all(len(chosen_rows[k]) == per_class for k in ('0', '1')):
        raise RuntimeError(f"Only {len(chosen_rows['0'])} human and {len(chosen_rows['1'])} AI valid records "
                           f"after {page_attempts} pages. CSV NOT written. "
                           "Try --max-pages 120, or check the dataset's availability.")
    samples = chosen_rows["0"] + chosen_rows["1"]
    rng.shuffle(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents partial files if writing is interrupted.
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "label", "group_id"))
        writer.writeheader()
        writer.writerows(samples)
    tmp.replace(output)
    manifest = {
        "source": SOURCE, "dataset": DATASET, "config": config, "split": "test",
        "generated_mapping": {"0": "human", "1": "AI"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "human": len(chosen_rows['0']), "ai": len(chosen_rows['1']),
        "minimum_words": 120, "maximum_words": 1500,
        "training_csv_checked": training_path.is_file(),
        "training_exact_normalized_overlap_excluded": bool(training),
        "selection": ("Seed-shuffled 100-row pages of upstream test split" if size else
                      "Sequential 100-row pages of upstream test split with unknown size") + "; no training split retrieved",
        "live_row_count": size if size else None,
        "row_count_source": count_origin,
        "pages_accessed": page_attempts, "excluded": dict(rejected),
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": [
            "Upstream labels/provenance not independently audited; possible mislabeled examples.",
            "The upstream test split could overlap with external training data beyond exact-text and prefix checks.",
            "This dataset is mostly essays, not all domains or current LLM generators.",
            "Never use this held-out evaluation set to train models or tune thresholds.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSUCCESS: Created", output, flush=True)
    print("Human:", len(chosen_rows['0']), " | AI:", len(chosen_rows['1']), flush=True)
    print("Dataset source:", SOURCE, flush=True)
    print("Provenance:", manifest_path, flush=True)
    print("WARNING: Source labels are externally supplied. Dataset is not proof of authorship.", flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description="Automatically prepare a labeled external TraceLens evaluation dataset")
    ap.add_argument("--per-class", type=int, default=60, help="Documents from each class (default: 60)")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--max-pages", type=int, default=55)
    args = ap.parse_args()
    try:
        build(root=Path.cwd(), per_class=args.per_class, seed=args.seed, max_pages=args.max_pages)
    except (ValueError, RuntimeError, OSError) as exc:
        print("\nDATASET PREPARATION FAILED:", exc, file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
