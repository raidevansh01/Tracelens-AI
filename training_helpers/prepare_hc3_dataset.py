#!/usr/bin/env python3
"""Download real HC3 human/ChatGPT pairs and prepare a TraceLens training CSV.

Citation: Guo et al., 2023, 'How Close is ChatGPT to Human Experts?'
Dataset: https://huggingface.co/datasets/Hello-SimpleAI/HC3
License: CC BY-SA 4.0, except any stricter component-source terms.

No text is fabricated. No model is trained until explicitly requested.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

FILES = ('wiki_csai.jsonl', 'finance.jsonl', 'medicine.jsonl', 'open_qa.jsonl', 'reddit_eli5.jsonl')
BASE_URL = 'https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/'
WORD = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", re.UNICODE)
SPACES = re.compile(r'\s+')
SIGNATURE = re.compile(r'\b(as an ai language model|as a language model|i am an ai|chatgpt (?:is|was) an ai)\b',re.I)


def tokens(s: str) -> list[str]:
    return [m.group().casefold() for m in WORD.finditer(s)]


def clean(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.replace('\x00','').replace('\r\n','\n').strip()
    return SPACES.sub(' ', s)


def fingerprint(s: str) -> str:
    return hashlib.sha256(' '.join(tokens(s)).encode('utf-8')).hexdigest()


def download_file(path: Path, filename: str) -> None:
    if path.exists() and path.stat().st_size > 500:
        print(f'CACHE {filename}: {path.stat().st_size / 1e6:.2f} MB')
        return
    try:
        import requests
    except ImportError as e:
        raise RuntimeError('Install requests first: python -m pip install requests') from e
    url=BASE_URL+filename+'?download=true'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    print('DOWNLOADING',url,flush=True)
    try:
        with requests.get(url,stream=True,timeout=(20,120),headers={
            'User-Agent':'TraceLensAI-DatasetBuilder/1.0 (student research; HC3)'}) as resp:
            resp.raise_for_status()
            if 'text/html' in resp.headers.get('Content-Type','').lower():
                raise RuntimeError('Dataset host returned HTML instead of JSONL')
            size=0
            with temp.open('wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        size+=len(chunk)
                        if size > 110_000_000:
                            raise RuntimeError(f'Source file unexpectedly large: {filename}')
        if size < 100:
            raise RuntimeError('Download returned an empty file')
        with temp.open('rb') as f:
            if not f.read(200).lstrip().startswith(b'{'):
                raise RuntimeError('Downloaded content is not expected JSONL; check source URL')
        temp.replace(path)
        print(f'DOWNLOADED {filename}: {size / 1e6:.2f} MB',flush=True)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def find_pairs(path: Path, min_words: int, max_words: int, max_ratio: float):
    """Yield one authentic pair per question with basic length/disclaimer checks."""
    source_fallback=path.stem
    parsed=0
    with path.open('r',encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            parsed+=1
            try:
                obj=json.loads(line)
            except json.JSONDecodeError:
                continue
            question=clean(obj.get('question',''))
            if not question:
                continue
            humans=obj.get('human_answers') or []
            ais=obj.get('chatgpt_answers') or []
            if isinstance(humans,str): humans=[humans]
            if isinstance(ais,str): ais=[ais]
            good_h=[]
            good_a=[]
            for text in humans:
                text=clean(text)
                w=len(tokens(text))
                if min_words<=w<=max_words and not SIGNATURE.search(text):
                    good_h.append((text,w))
            for text in ais:
                text=clean(text)
                w=len(tokens(text))
                if min_words<=w<=max_words and not SIGNATURE.search(text):
                    good_a.append((text,w))
            if not good_h or not good_a:
                continue
            choices=[(max(hw,aw)/min(hw,aw),abs(hw-aw),h,a,hw,aw)
                     for h,hw in good_h for a,aw in good_a if h!=a]
            choices=[x for x in choices if x[0] <= max_ratio]
            if not choices:
                continue
            _,_,h,a,hw,aw=min(choices,key=lambda x:(x[0],x[1]))
            src=clean(obj.get('source') or source_fallback)
            # Same question always stays in same group even if dataset contains repeated rows.
            gid='hc3_'+hashlib.sha256(question.casefold().encode('utf-8')).hexdigest()[:16]
            yield {'group_id':gid,'question':question,'source':src,
                   'human':h,'ai':a,'human_words':hw,'ai_words':aw,
                   'source_file':path.name,'source_row':parsed}


def choose_pairs(by_source: dict, n: int, seed: int):
    r=random.Random(seed)
    sources=sorted(by_source)
    for src in sources:
        r.shuffle(by_source[src])
    chosen=[]
    used_g=set()
    used_text=set()
    while len(chosen)<n:
        advanced=False
        for src in sources:
            remaining=by_source[src]
            while remaining:
                pair=remaining.pop()
                h=fingerprint(pair['human']); a=fingerprint(pair['ai'])
                if pair['group_id'] in used_g or h==a or h in used_text or a in used_text:
                    continue
                used_g.add(pair['group_id'])
                used_text.add(h); used_text.add(a)
                chosen.append(pair)
                advanced=True
                break
            if len(chosen)>=n:
                break
        if not advanced:
            break
    return chosen


def validate(rows, minimum=100):
    if len(rows)<minimum or len(rows)%2:
        raise ValueError(f'Dataset has {len(rows)} rows; need >= {minimum} and an even row count')
    counters=collections.Counter(r['label'] for r in rows)
    if counters['0']!=counters['1']:
        raise ValueError('Unbalanced class labels')
    groups=collections.defaultdict(list)
    hashes=set()
    for row in rows:
        if len(tokens(row['text'])) < 120:
            raise ValueError('A selected row contains fewer than 120 words')
        groups[row['group_id']].append(row['label'])
        fp=fingerprint(row['text'])
        if fp in hashes:
            raise ValueError('A duplicate text appeared in two different rows')
        hashes.add(fp)
    if len(groups)<20 or any(sorted(x)!=['0','1'] for x in groups.values()):
        raise ValueError('Missing, mismatched or repeated prompt groups')


def build(paths: list[Path], out: Path, n: int, seed: int,
          min_words: int=120,max_words: int=550,max_ratio: float=1.7):
    by_source=collections.defaultdict(list)
    for path in paths:
        pairs=list(find_pairs(path,min_words,max_words,max_ratio))
        print(f'PARSED {path.name}: {len(pairs)} eligible authentic pairs')
        for pair in pairs:
            by_source[pair['source']].append(pair)
    selected=choose_pairs(by_source,n,seed)
    if len(selected)<n:
        raise ValueError(f'Only {len(selected)} qualifying unique human/AI pairs were found; '
                         'rerun with additional HC3 source files, or use a lower --pairs count.')
    r=random.Random(seed)
    r.shuffle(selected)
    rows=[]
    for p in selected:
        rows.append({'text':p['human'],'label':'0','group_id':p['group_id']})
        rows.append({'text':p['ai'],'label':'1','group_id':p['group_id']})
    validate(rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['text','label','group_id'])
        writer.writeheader();writer.writerows(rows)
    meta_path=out.with_name('hc3_selection_metadata.csv')
    with meta_path.open('w',encoding='utf-8',newline='') as f:
        fields=['group_id','question','source','human_words','ai_words','source_file','source_row']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows([{k:p[k] for k in fields} for p in selected])
    provenance={'dataset':'Hello-SimpleAI/HC3','source_url':'https://huggingface.co/datasets/Hello-SimpleAI/HC3',
                'citation':'Guo et al. (2023), How Close is ChatGPT to Human Experts? arXiv:2301.07597',
                'license':'CC-BY-SA-4.0; stricter licenses may apply to source components',
                'generated_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,
                'method':'Matched human and ChatGPT answers to the same question; length- and duplication-filtered',
                'pairs':len(selected),'rows':len(rows),'source_counts':dict(collections.Counter(p['source'] for p in selected)),
                'min_words':min_words,'max_words':max_words,'max_length_ratio':max_ratio,
                'input_files':[p.name for p in paths],
                'limitations':['Older ChatGPT-era examples, not representative of all modern AI models',
                               'Writing genres differ from college essays',
                               'Some source corpora may have additional license terms',
                               'Statistical predictions cannot prove AI authorship']}
    provenance_path=out.with_name('hc3_provenance.json')
    provenance_path.write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSUCCESS: formatted the provided HC3-style labeled corpus (verify source provenance)')
    print('CSV:',out)
    print('Rows:',len(rows),'| human:',len(selected),'| AI:',len(selected),'| independent groups:',len(selected))
    print('Source distribution:',dict(provenance['source_counts']))
    print('Metadata:',meta_path)
    print('Provenance:',provenance_path)
    return provenance


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=int,default=150,help='Number of paired prompts (default 150 => 300 rows)')
    p.add_argument('--out',type=Path,default=Path('datasets/ai_writing.csv'))
    p.add_argument('--cache',type=Path,default=Path('datasets/hc3_cache'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--input-jsonl',type=Path,nargs='+',help='Use already downloaded HC3 JSONL files; avoid network requests')
    p.add_argument('--train',action='store_true',help='Train current TraceLens classifier after preparing CSV')
    p.add_argument('--min-words',type=int,default=120)
    p.add_argument('--max-words',type=int,default=550)
    p.add_argument('--max-length-ratio',type=float,default=1.7)
    args=p.parse_args(argv)
    if args.pairs < 50:
        p.error('--pairs must be >= 50 because current TraceLens training script needs at least 100 rows')
    if args.min_words<120:
        p.error('--min-words must be >=120 for current TraceLens training script')
    if args.min_words>=args.max_words:
        p.error('--min-words must be smaller than --max-words')
    paths=[]
    if args.input_jsonl:
        for file in args.input_jsonl:
            if not file.exists():
                p.error(f'No file: {file}')
            paths.append(file)
    else:
        # Prefer smaller files first; only download the largest when more examples are needed.
        for filename in FILES:
            path=args.cache / filename
            try:
                download_file(path,filename)
            except Exception as exc:
                print(f'WARNING: could not retrieve {filename}: {exc}',file=sys.stderr)
                continue
            paths.append(path)
            count=sum(1 for f in paths for _ in find_pairs(f,args.min_words,args.max_words,args.max_length_ratio))
            print(f'Eligible pairs collected before deduplication: {count}',flush=True)
            if count>=int(args.pairs*1.5):
                break
    if not paths:
        p.error('No HC3 files available. Verify your internet connection or download .jsonl '
                'files from the official HC3 dataset and use --input-jsonl.')
    try:
        build(paths,args.out,args.pairs,args.seed,args.min_words,args.max_words,args.max_length_ratio)
    except ValueError as exc:
        p.error(str(exc))
    if args.train:
        trainer=Path('train_ai_patterns.py')
        if not trainer.exists():
            p.error('--train requires running inside TraceLens-AI project root containing train_ai_patterns.py')
        print('\nTRAINING the experiment classifier...')
        subprocess.run([sys.executable,str(trainer),'--data',str(args.out)],check=True)
        print('Model written by TraceLens training code to storage/ai_pattern_model.joblib')
    return 0


if __name__=='__main__':
    sys.exit(main())
