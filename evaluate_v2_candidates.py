"""Evaluate v2 candidates on a frozen external benchmark, NOT for tuning."""
from __future__ import annotations
import argparse
import csv
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from core.ai_patterns import make_windows
from train_v2_candidates import score_features, stats

ROOT=Path(__file__).resolve().parent


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    valid=[]
    for r in rows:
        if r.get('label') in ('0','1') and make_windows(r.get('text','')):
            valid.append({'text':r['text'],'label':int(r['label']),'group_id':r.get('group_id','')})
    if len(valid)<10 or {r['label'] for r in valid}!={0,1}:
        raise ValueError('External benchmark must contain both classes and at least 10 usable texts.')
    return valid


def check_overlap(train, test):
    def norm(t):
        import re
        return hashlib.sha256(' '.join(re.findall(r'\w+',t.lower())).encode()).hexdigest()
    tr={norm(r['text']) for r in train}
    overlaps=sum(norm(r['text']) in tr for r in test)
    groups={r['group_id'] for r in train if r['group_id']}
    group_overlaps=sum(r['group_id'] in groups for r in test if r['group_id'])
    if overlaps or group_overlaps:
        raise ValueError(f'Benchmark contaminated: {overlaps} exact normalized duplicates, {group_overlaps} group overlaps.')


def run(*, dataset, training_data, method, out_dir=ROOT/'datasets'/'evaluation_reports'):
    rows=read_rows(dataset)
    check_overlap(read_rows(training_data),rows)
    methods=('ngram','minilm') if method=='both' else (method,)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for m in methods:
        p=ROOT/'storage'/'v2_experiments'/f'ai_{m}_candidate.joblib'
        if not p.exists():
            print(f'SKIPPED {m}: no candidate. Train it before evaluation.');continue
        item=joblib.load(p)  # only load trusted, locally trained models
        if item.get('schema')!='tracelens_comparator_1' or item.get('method')!=m:
            raise ValueError('Incompatible model artifact: '+str(p))
        if m=='minilm':
            from core.detection import load_semantic_model
            enc=load_semantic_model()
        else:
            enc=None
        scores,y=score_features(item['model'],item['vectorizer'],rows,m,enc)
        threshold=item['threshold']
        metric=stats(y,scores,threshold)
        results[m]={'model':m,'threshold':threshold,'external_test':metric,
                    'note':'Public test split of same upstream essay corpus; further ChatGPT-specific testing required.'}
        print(m.upper(), 'TP',metric['true_positives'],'FP',metric['false_positives'],
              'TN',metric['true_negatives'],'FN',metric['false_negatives'],
              '| recall',round(metric['recall'],3),
              '| FPR',round(metric['false_positive_rate'],3),
              '| precision',None if metric['precision'] is None else round(metric['precision'],3),flush=True)
    if not results:
        raise ValueError('No candidates were evaluated; train at least one first.')
    result={'evaluated_utc':datetime.now(timezone.utc).isoformat(),
            'benchmark_csv':str(dataset),
            'benchmark_sha256':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            'results':results}
    target=out_dir/'v2_external_comparison.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('EXTERNAL REPORT',target)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='datasets/independent_test.csv')
    p.add_argument('--training-data',default='datasets/ai_v2_train.csv')
    p.add_argument('--method',choices=['ngram','minilm','both'],default='both')
    a=p.parse_args()
    try:
        run(dataset=Path(a.data),training_data=Path(a.training_data),method=a.method)
        return 0
    except Exception as ex:
        print('EVALUATION FAILED:',ex)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
