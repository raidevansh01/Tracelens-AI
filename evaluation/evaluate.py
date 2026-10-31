"""Educational comparison on a tiny illustrative dataset, NOT a performance benchmark."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from core.detection import load_semantic_model

PATH=Path(__file__).parent/'toy_pairs.csv'

def compare_toy(use_semantic=False):
    with PATH.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    a=[r['submitted'] for r in rows]
    b=[r['source'] for r in rows]
    y=np.array([int(r['related']) for r in rows])
    vect=TfidfVectorizer(ngram_range=(1,2))
    data=vect.fit_transform(a+b)
    lexical=np.asarray(data[:len(a)].multiply(data[len(a):]).sum(axis=1)).flatten()
    result={'note':'Tiny synthetic examples; scores are not an accuracy claim and thresholds are illustrative.',
            'n':len(rows),'labels':y.tolist(),'lexical':lexical.tolist(),
            'samples':rows}
    def metrics(scores,threshold):
        predictions=np.array(scores)>=threshold
        p,r,f,_=precision_recall_fscore_support(y,predictions,average='binary',zero_division=0)
        return {'precision':round(float(p),3),'recall':round(float(r),3),'f1':round(float(f),3),
                'threshold':threshold}
    result['lexical_metrics']=metrics(lexical,.25)
    if use_semantic:
        model=load_semantic_model()
        ea=model.encode(a,normalize_embeddings=True)
        eb=model.encode(b,normalize_embeddings=True)
        sem=np.sum(np.asarray(ea)*np.asarray(eb),axis=1)
        result['semantic']=sem.tolist()
        result['semantic_metrics']=metrics(sem,.70)
    return result
