"""Independent evaluation of the optional research classifier (NO training).

Input CSV: text,label (0 human, 1 AI); optionally group_id.
Use samples not present in HC3 training or threshold-selection data.

python qa/evaluate_external_writing.py --data datasets/independent_test.csv
python qa/evaluate_external_writing.py --data datasets/independent_test.csv --training-data datasets/ai_writing.csv
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.ai_patterns import infer
from core.ai_comparator import inspect_candidates
from core.extractor import word_tokens


def read_rows(path):
    with open(path,encoding='utf-8-sig',newline='') as fp:
        rows=list(csv.DictReader(fp))
    if not rows or not {'text','label'}.issubset(rows[0]):
        raise ValueError('CSV requires text,label columns; labels: 0=human, 1=AI')
    return [r for r in rows if r.get('label') in ('0','1') and len(word_tokens(r.get('text','')))>=120]


def run(dataset, train_dataset=None, model="stylometry"):
    rows=read_rows(dataset)
    if not rows:
        print('FAIL: no usable 120+ word labeled samples'); return 2
    if train_dataset:
        training=read_rows(train_dataset)
        training_hashes={hashlib.sha256(r['text'].strip().encode('utf-8')).hexdigest() for r in training}
        overlaps=sum(hashlib.sha256(r['text'].strip().encode('utf-8')).hexdigest() in training_hashes for r in rows)
        train_groups={r.get('group_id') for r in training if r.get('group_id')}
        group_overlaps=sum(r.get('group_id') in train_groups for r in rows if r.get('group_id'))
        if overlaps or group_overlaps:
            print(f'INVALID EVALUATION: {overlaps} text duplicates, {group_overlaps} group overlaps with training data.'); return 2
    counts={'TP':0,'FP':0,'TN':0,'FN':0}
    unknown=0
    for row in rows:
        if model=='stylometry':
            outcome=infer(row['text'])
            if outcome.get('status')!='experimental':
                print('MODEL UNAVAILABLE:',outcome.get('message',outcome.get('status')))
                return 2
            pred=int(any(w['review_flag'] for w in outcome['windows']))
            uncertain=any(w.get('distribution_warning') for w in outcome['windows'])
        else:
            outcome=inspect_candidates(row['text'])['models'].get(model,{})
            if outcome.get('status')!='experimental':
                print('MODEL UNAVAILABLE:',model, 'Train using train_ai_comparator.py --method',model)
                return 2
            pred=int(any(outcome['flags']))
            uncertain=False  # No validated distribution detector for this model.
        actual=int(row['label'])
        key='TP' if pred and actual else 'FP' if pred else 'FN' if actual else 'TN'
        counts[key]+=1
        if uncertain:
            unknown+=1
    tp,fp,tn,fn=(counts[k] for k in ('TP','FP','TN','FN'))
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    fpr=fp/(fp+tn) if fp+tn else 0
    print('Independent evaluation:',len(rows),'documents; model=',model)
    print('Confusion matrix:',counts)
    print(f'Precision={precision:.3f} Recall={recall:.3f} False-positive-rate={fpr:.3f}')
    if model=='stylometry': print('Out-of-training-range / potentially inconclusive samples:',unknown)
    else: print('Distribution-shift check: not validated for this model. All results remain experimental.')
    if len(rows)<100 or min(tp+fn,fp+tn)<40:
        print('WARNING: Small test set. Results are too uncertain to support broad detection claims.')
    print('These metrics characterize only the evaluated examples, not proof of authorship.')
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--training-data')
    parser.add_argument('--model',choices=['stylometry','ngram','minilm'],default='stylometry')
    args=parser.parse_args()
    raise SystemExit(run(args.data,args.training_data,args.model))
