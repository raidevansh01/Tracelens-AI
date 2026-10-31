"""Optional independently evaluated writing-pattern candidates.

These supervised models are NOT reliable authorship tests. Their predictions
are shown independently; disagreement or training shift => inconclusive.
Only load locally trained/trusted joblib artifacts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .ai_patterns import make_windows

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'storage'
MODELS={'ngram':MODEL_DIR/'ai_ngram_candidate.joblib',
        'minilm':MODEL_DIR/'ai_minilm_candidate.joblib'}


def inspect_candidates(text: str, *, paths: dict | None = None) -> dict:
    import joblib
    paths=paths or MODELS
    windows=make_windows(text)
    if not windows:
        return {'status':'insufficient_text','models':{},'message':'At least 120 words needed.'}
    result={}
    for method,path in paths.items():
        p=Path(path)
        if not p.exists():
            continue
        try:
            item=joblib.load(p)  # trusted, locally trained models ONLY
            if item.get('schema') != 'tracelens_comparator_1' or item.get('method') != method:
                raise ValueError('Incompatible model schema')
            texts=[w['text'] for w in windows]
            if method=='ngram':
                X=item['vectorizer'].transform(texts)
            elif method=='minilm':
                from .detection import load_semantic_model
                X=load_semantic_model().encode(texts,batch_size=16,
                       normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            else:
                continue
            probabilities=item['model'].predict_proba(X)[:,1]
            threshold=float(item['threshold'])
            result[method]={'status':'experimental','threshold':threshold,
                     'scores':[round(float(v),4) for v in probabilities],
                     'flags':[bool(v>=threshold) for v in probabilities],
                     'evaluation':item['evaluation'],
                     'model_version':item.get('training_metadata',{}).get('candidate_release','original-candidate'),
                     'note':'Model predictions are not reliable authorship probabilities.'}
        except Exception as exc:
            result[method]={'status':'unavailable','message':f'{type(exc).__name__}: Could not run candidate.'}
    return {'status':'available' if result else 'not_trained','models':result,
            'message':'Compare experimental models; disagreement is inconclusive, not a vote on authorship.'}
