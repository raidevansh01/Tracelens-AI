"""Create a separate training corpus from the official TRAIN split, never TEST.

Requires Python standard library only. Place beside app.py in TraceLens AI.
This source is AI-human-text essay data, NOT verified current ChatGPT output.
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

BASE = 'https://datasets-server.huggingface.co'
DATASET = 'andythetechnerd03/AI-human-text'
SPLIT = 'train'
TOKEN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)


def words(s):
    return TOKEN.findall(s)


def fingerprint(s):
    return hashlib.sha256(' '.join(w.casefold() for w in words(s)).encode('utf8')).hexdigest()


def api(path, params, *, fetch=None):
    if fetch is not None:
        return fetch(path, params)
    url = BASE + path + '?' + urlencode(params)
    last = None
    for attempt in range(4):
        try:
            req = Request(url, headers={'User-Agent': 'TraceLens-AI-v2-Training/1.0', 'Accept': 'application/json'})
            with urlopen(req, timeout=40) as response:
                data = response.read(10_000_000)
            return json.loads(data)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code in (400, 401, 403, 404):
                break
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Dataset API {path} request failed: {last}. Check internet access or rate limits.')


def existing_fingerprints(csvpath):
    sigs = set()
    prefixes = set()
    if not csvpath.is_file():
        return sigs, prefixes
    with csvpath.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            s = row.get('text', '')
            if s.strip():
                ws = words(s)
                sigs.add(fingerprint(s))
                if len(ws) >= 45:
                    prefixes.add(tuple(w.casefold() for w in ws[:45]))
    return sigs, prefixes


def build(root: Path, *, per_class=1000, seed=20261030, max_pages=100, fetch=None):
    root = Path(root)
    if not (root / 'app.py').is_file():
        raise ValueError('Run this script from the TraceLens folder containing app.py.')
    if not 50 <= per_class <= 5000:
        raise ValueError('Choose --per-class between 50 and 5000.')
    if not 1 <= max_pages <= 400:
        raise ValueError('--max-pages must be 1..400.')
    dataset_dir = root / 'datasets'
    independent = dataset_dir / 'independent_test.csv'
    hc3 = dataset_dir / 'ai_writing.csv'
    out = dataset_dir / 'ai_v2_train.csv'
    if not independent.exists():
        raise ValueError('Missing datasets/independent_test.csv; preserve the previous external benchmark before training.')
    train_sigs, train_prefix = existing_fingerprints(hc3)
    test_sigs, test_prefix = existing_fingerprints(independent)
    if not test_sigs:
        raise ValueError('Independent benchmark is empty; refusing to build training data.')
    seen = train_sigs | test_sigs
    seen_prefix = train_prefix | test_prefix
    print('Protected benchmark fingerprints:', len(test_sigs), '| HC3 fingerprints:', len(train_sigs), flush=True)
    first = api('/rows', {'dataset': DATASET, 'config': 'default', 'split': SPLIT, 'offset': 0, 'length': 100}, fetch=fetch)
    total = int(first.get('num_rows_total') or 0)
    if not isinstance(first.get('rows'), list) or not first['rows'] or total < 500:
        raise RuntimeError('The TRAIN split is unavailable or too small. No files changed.')
    print('Confirmed upstream TRAIN split:', total, 'rows.', flush=True)
    pages = list(range(0, total, 100))
    random.Random(seed).shuffle(pages)
    collected = {'0': [], '1': []}
    rejected = Counter()
    processed = 0
    for offset in pages[:max_pages]:
        if offset == 0:
            response = first
        else:
            response = api('/rows', {'dataset': DATASET, 'config': 'default', 'split': SPLIT,
                                     'offset': offset, 'length': min(100, total-offset)}, fetch=fetch)
        if not isinstance(response.get('rows'), list):
            raise RuntimeError(f'Unexpected row format at offset {offset}. No files changed.')
        for sample in response['rows']:
            r = sample.get('row') or {}
            label = r.get('generated')
            s = r.get('text')
            if isinstance(label, bool):
                label = int(label)
            if label not in (0, 1) or not isinstance(s, str):
                rejected['bad_value'] += 1
                continue
            label = str(label)
            if len(collected[label]) >= per_class:
                continue
            s = s.strip()
            w = words(s)
            if not 120 <= len(w) <= 1500:
                rejected['too_short_or_long'] += 1
                continue
            fp = fingerprint(s)
            prefix = tuple(t.casefold() for t in w[:45])
            if fp in seen or prefix in seen_prefix:
                rejected['duplicate_or_overlap'] += 1
                continue
            seen.add(fp)
            seen_prefix.add(prefix)
            idx = sample.get('row_idx', offset)
            collected[label].append({'text': s, 'label': label, 'group_id': f'ai-human-train-{idx}'})
        processed += 1
        if processed % 5 == 0 or min(map(len, collected.values())) >= per_class:
            print(f'Pages {processed}: human {len(collected["0"])}/{per_class}, AI {len(collected["1"])}/{per_class}', flush=True)
        if all(len(collected[k]) == per_class for k in ('0', '1')):
            break
    if any(len(collected[k]) < per_class for k in ('0', '1')):
        raise RuntimeError(f'Only human={len(collected["0"])}, AI={len(collected["1"])} obtained. '
                           'Try --max-pages 180 (no partial data saved).')
    rows = collected['0'] + collected['1']
    # HC3 has known ChatGPT-generated answers; blend it into training, not evaluation.
    hc3_added = 0
    if hc3.is_file():
        with hc3.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                s = row.get('text', '').strip()
                if row.get('label') not in ('0','1') or not s or not 120 <= len(words(s)) <= 1500:
                    continue
                if fingerprint(s) in test_sigs:
                    continue
                rows.append({'text': s, 'label': row['label'],
                             'group_id': 'hc3-' + row.get('group_id', f'row-{hc3_added}')})
                hc3_added += 1
    random.Random(seed).shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text','label','group_id'])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(out)
    manifest = {
        'source': f'https://huggingface.co/datasets/{DATASET}', 'training_split': SPLIT,
        'source_license': 'apache-2.0 (source dataset card)',
        'generated_mapping': {'0':'human','1':'AI'}, 'seed': seed,
        'original_train_rows_selected_per_class': per_class, 'hc3_rows_added': hc3_added,
        'total_training_rows': len(rows), 'excluded_overlap_count': rejected['duplicate_or_overlap'],
        'test_file': str(independent), 'test_split_never_downloaded_for_training': True,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'training_file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'limitations': [
            'The external test benchmark and extra training essays come from different splits of the SAME upstream dataset; domain independence is limited.',
            'AI essay labels and generative model identities are supplied by the publisher and not independently verified.',
            'HC3 contains older ChatGPT answer styles and may not represent current ChatGPT.',
            'Exact and shared-prefix duplicates were checked; other near duplicates may remain.',
            'Do not use datasets/independent_test.csv for training or threshold tuning.'
        ]
    }
    (dataset_dir/'ai_v2_train_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('CREATED', out, 'documents:', len(rows), '| HC3 added:', hc3_added, flush=True)
    print('The independent_test.csv file was NOT modified.', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261030)
    opts = parser.parse_args()
    try:
        build(Path.cwd(), per_class=opts.per_class, max_pages=opts.max_pages, seed=opts.seed)
        return 0
    except Exception as ex:
        print('DATASET PREPARATION FAILED:', ex, file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
