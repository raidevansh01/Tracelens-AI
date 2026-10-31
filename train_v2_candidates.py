"""Train isolated TraceLens AI-writing candidate models with group-held-out calibration.

Uses the v1.8 candidate joblib schema for direct compatibility with the app.
Never overwrites the currently installed models; no external test data is read.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from core.ai_patterns import make_windows

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'storage' / 'v2_experiments'


def load_rows(path: Path):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('label') not in ('0','1') or not r.get('group_id') or not make_windows(r.get('text', '')):
                continue
            rows.append({'text': r['text'], 'label': int(r['label']), 'group_id': r['group_id']})
    if len(rows) < 150 or len(set(r['group_id'] for r in rows)) < 100:
        raise ValueError('Training data must contain at least 150 documents from 100 distinct groups.')
    if {r['label'] for r in rows} != {0,1}:
        raise ValueError('Both human and AI labels are required.')
    return rows


def score_features(model, vectorizer, rows, method, encoder=None, batch_size=16):
    texts=[]; lengths=[]; actual=[]
    for r in rows:
        samples=[p['text'] for p in make_windows(r['text'])]
        texts.extend(samples)
        lengths.append(len(samples)); actual.append(r['label'])
    if method == 'ngram':
        X = vectorizer.transform(texts)
    else:
        X = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    raw = model.predict_proba(X)[:, 1]
    max_scores=[]; pos=0
    for n in lengths:
        max_scores.append(float(max(raw[pos:pos+n]))); pos+=n
    return np.asarray(max_scores), np.asarray(actual)


def threshold_for_validation_humans(scores, desired_fpr=.05):
    """Select threshold using only human validation scores, not external test data.

    Guarantees no more than floor(n * desired_fpr) flags on THIS validation sample,
    including ties; makes no guarantee about future data.
    """
    if len(scores) < 40:
        raise ValueError('At least 40 human validation documents required for threshold calibration.')
    if not 0 < desired_fpr < .5:
        raise ValueError('desired FPR must be between 0 and 0.5')
    scores=np.sort(np.asarray(scores, dtype=float))
    allowed=int(math.floor(len(scores)*desired_fpr))
    i=max(0, len(scores)-allowed-1)
    return float(min(1.0, np.nextafter(scores[i], np.inf)))


def stats(y, scores, threshold):
    pred=(scores>=threshold).astype(int)
    tn, fp, fn, tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return {
        'samples':int(len(y)), 'human':int(sum(y==0)), 'ai':int(sum(y==1)),
        'true_positives':int(tp), 'false_positives':int(fp),
        'true_negatives':int(tn), 'false_negatives':int(fn),
        'precision':float(tp/(tp+fp)) if tp+fp else None,
        'recall':float(tp/(tp+fn)) if tp+fn else 0,
        'false_positive_rate':float(fp/(fp+tn)) if fp+tn else 0,
        'accuracy':float((tp+tn)/len(y)) if len(y) else 0,
    }


def train(path, methods=('ngram','minilm'), target_fpr=.05, *, output_dir=EXPERIMENT, seed=20261030):
    rows=load_rows(Path(path))
    groups=np.asarray([r['group_id'] for r in rows])
    idx=np.arange(len(rows))
    trainidx, validx=next(GroupShuffleSplit(test_size=.25,random_state=seed).split(idx,groups=groups))
    training=[rows[i] for i in trainidx]
    validation=[rows[i] for i in validx]
    if min(sum(r['label']==i for r in validation) for i in (0,1))<40:
        raise ValueError('Validation contains insufficient human/AI documents; collect more training examples.')
    xtrain=[]; ytrain=[]
    for r in training:
        w=[v['text'] for v in make_windows(r['text'])]
        if len(w)>3:
            # Avoid overweighting lengthy essays: first, middle and final windows.
            w=[w[0],w[len(w)//2],w[-1]]
        xtrain.extend(w)
        ytrain.extend([r['label']]*len(w))
    ytrain=np.asarray(ytrain)
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results={}
    for method in methods:
        print('Training',method,'with',len(xtrain),'windows and',len(validation),'calibration documents',flush=True)
        if method == 'ngram':
            vectorizer=FeatureUnion([
                ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True)),
                ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=50000,sublinear_tf=True))])
            X=vectorizer.fit_transform(xtrain)
            encoder=None
        elif method == 'minilm':
            from core.detection import load_semantic_model
            encoder=load_semantic_model()
            X=encoder.encode(xtrain,batch_size=16,normalize_embeddings=True,
                             convert_to_numpy=True,show_progress_bar=True)
            vectorizer=None
        else:
            raise ValueError('Unsupported method '+method)
        clf=LogisticRegression(max_iter=1500, class_weight='balanced', C=.7,random_state=seed)
        clf.fit(X,ytrain)
        scores,yval=score_features(clf,vectorizer,validation,method,encoder)
        threshold=threshold_for_validation_humans(scores[yval==0],target_fpr)
        metrics=stats(yval,scores,threshold)
        metrics.update({'method':method,'validation_human_samples':int(sum(yval==0)),
                        'training_documents':len(training),'training_windows':len(xtrain),
                        'target_validation_fpr':target_fpr,
                        'note':'Validation only, from same mixed training corpus. This is not external accuracy.'})
        pathout=out/f'ai_{method}_candidate.joblib'
        item={'schema':'tracelens_comparator_1','method':method,'model':clf,
              'vectorizer':vectorizer,'threshold':threshold,'evaluation':metrics,
              'training_metadata':{'source_file':str(path),'training_rows':len(rows),'seed':seed,
                                   'validation_groups_separate':True,'candidate_release':'v2-experimental'}}
        joblib.dump(item,pathout)
        print('Saved',pathout,'threshold',round(threshold,4), 'val recall',round(metrics['recall'],3),
              'val FPR',round(metrics['false_positive_rate'],3),flush=True)
        results[method]={'path':str(pathout),'threshold':threshold,'validation':metrics}
    (out/'training_report.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='datasets/ai_v2_train.csv')
    p.add_argument('--method',choices=['ngram','minilm','both'],default='both')
    p.add_argument('--target-fpr',type=float,default=.05)
    opts=p.parse_args()
    methods=('ngram','minilm') if opts.method=='both' else (opts.method,)
    try:
        train(opts.data,methods=methods,target_fpr=opts.target_fpr)
    except Exception as exc:
        print('TRAINING FAILED:',str(exc))
        return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
