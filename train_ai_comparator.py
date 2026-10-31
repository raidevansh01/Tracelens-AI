"""Train an independent n-gram or MiniLM experimental comparison model.

PowerShell: .\\.venv\\Scripts\\python.exe train_ai_comparator.py --data datasets\\ai_writing.csv --method ngram
Optional: --method minilm (requires requirements-ai.txt and a model download)
This creates a separate model, never overwrites the original classifier.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from core.ai_patterns import make_windows
from core.ai_comparator import MODELS


def read_samples(path):
    rows=[]
    with open(path,encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('label') not in ('0','1') or not row.get('group_id') or not row.get('text','').strip():
                continue
            if not make_windows(row['text']):
                continue
            rows.append({'text':row['text'],'group_id':row['group_id'],'label':int(row['label'])})
    if len(set(r['group_id'] for r in rows))<30:
        raise ValueError('Provide >=30 independent group IDs with enough text for >=120-word windows.')
    if set(r['label'] for r in rows) != {0,1}:
        raise ValueError('Both labeled human (0) and AI (1) documents required.')
    return rows


def _split(rows):
    idx=np.arange(len(rows)); group=[r['group_id'] for r in rows]
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2026).split(idx,groups=group))
    train,val=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=2027).split(trainval,groups=np.asarray(group)[trainval]))
    return [rows[i] for i in trainval[train]],[rows[i] for i in trainval[val]],[rows[i] for i in test]


def _window_training(rows):
    x=[]; y=[]
    for r in rows:
        for w in make_windows(r['text']):
            x.append(w['text']); y.append(r['label'])
    return x,np.asarray(y)


def train(path,method='ngram',output=None):
    if method not in ('ngram','minilm'):
        raise ValueError('Method must be ngram or minilm')
    rows=read_samples(path)
    trainrows,valrows,testrows=_split(rows)
    xtrain,ytrain=_window_training(trainrows)
    vectorizer=None
    if method=='ngram':
        vectorizer=FeatureUnion([
          ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),
          ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=55000,sublinear_tf=True))])
        X=vectorizer.fit_transform(xtrain)
    else:
        from core.detection import load_semantic_model
        X=load_semantic_model().encode(xtrain,normalize_embeddings=True,convert_to_numpy=True,batch_size=16,show_progress_bar=False)
    clf=LogisticRegression(max_iter=1200,class_weight='balanced',C=.6)
    clf.fit(X,ytrain)
    def score(rows):
        predicted=[]; expected=[]
        for row in rows:
            windows=[w['text'] for w in make_windows(row['text'])]
            features=(vectorizer.transform(windows) if method=='ngram' else
                      load_semantic_model().encode(windows,normalize_embeddings=True,convert_to_numpy=True,batch_size=16,show_progress_bar=False))
            predicted.append(float(max(clf.predict_proba(features)[:,1])))
            expected.append(row['label'])
        return np.asarray(predicted),np.asarray(expected)
    validation,yl=score(valrows)
    human=validation[yl==0]
    if len(human)<3:
        raise ValueError('Not enough human validation examples to calibrate a threshold.')
    # Conservative observed-human threshold, not a guaranteed population FPR.
    threshold=float(min(0.999,max(.5,float(max(human))+.001)))
    prediction,yt=score(testrows)
    pred=(prediction>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel().tolist()
    evaluation={'method':method,'samples':len(yt),'human_test_samples':int(sum(yt==0)),
                'ai_test_samples':int(sum(yt==1)), 'true_positives':tp,'false_positives':fp,
                'true_negatives':tn,'false_negatives':fn,'precision':float(precision_score(yt,pred,zero_division=0)),
                'recall':float(recall_score(yt,pred,zero_division=0)),
                'false_positive_rate':fp/max(fp+tn,1),
                'validation_human_samples':len(human),'trained_windows':len(xtrain),
                'note':'Same-corpus, group-held-out test ONLY; not a reliability guarantee on modern or edited AI writing.'}
    destination=Path(output or MODELS[method])
    destination.parent.mkdir(exist_ok=True,parents=True)
    joblib.dump({'schema':'tracelens_comparator_1','method':method,'model':clf,
                 'vectorizer':vectorizer,'threshold':threshold,'evaluation':evaluation},destination)
    return {'saved_to':str(destination),'review_threshold':threshold,'evaluation':evaluation}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--method',choices=['ngram','minilm'],default='ngram')
    ap.add_argument('--output')
    args=ap.parse_args()
    print(train(args.data,args.method,args.output))
