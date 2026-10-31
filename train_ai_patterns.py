"""Train the experimental TraceLens v2 writing-style classifier.

Input CSV columns: text,label,group_id. Human=0, AI=1.
Split by original prompt/document GROUP before training, so no related source
appears across train, validation, and test partitions.
Every partition is then scored using the same sliding windows as inference.
Results measure this dataset, not general AI authorship detection.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from core.ai_patterns import (features, make_windows, FEATURE_NAMES,
                              MODEL_SCHEMA_VERSION, MIN_WORDS,
                              WINDOW_WORDS, WINDOW_STRIDE, MODEL_FILE)
from core.extractor import word_tokens


def _document_score(model, text: str) -> float:
    """The maximum window score is the document's review score, not probability."""
    windows = make_windows(text)
    return float(np.max(model.predict_proba(
        np.array([features(w['text']) for w in windows])
    )[:, 1]))


def train(csv_path: str, output: Path = MODEL_FILE) -> dict:
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'text', 'label', 'group_id'}.issubset(rows[0]):
        raise ValueError('CSV needs text,label,group_id columns.')
    rows = [r for r in rows if len(word_tokens(r['text'])) >= MIN_WORDS
            and r['label'] in ('0', '1') and r['group_id'].strip()]
    labels = np.array([int(r['label']) for r in rows]); groups = np.array([r['group_id'] for r in rows])
    if len(rows) < 100 or min(np.bincount(labels, minlength=2)) < 35 or len(set(groups)) < 20:
        raise ValueError('Need >=100 qualifying samples, >=35 per class, and >=20 independent groups.')

    # A group is assigned to precisely one partition, before window creation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=42)
    train_i, hold_i = next(splitter.split(np.zeros(len(rows)), labels, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=43)
    val_sub, test_sub = next(second.split(np.zeros(len(hold_i)), labels[hold_i], groups[hold_i]))
    val_i, test_i = hold_i[val_sub], hold_i[test_sub]
    for indices in (train_i, val_i, test_i):
        if len(set(labels[indices])) < 2:
            raise ValueError('All grouped splits need both labels; add more independently paired groups.')

    # Train on exactly the representations that will be used at inference time.
    train_x, train_y = [], []
    for i in train_i:
        for window in make_windows(rows[i]['text']):
            train_x.append(features(window['text']))
            train_y.append(int(labels[i]))
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1200, class_weight='balanced')),
    ])
    train_x = np.asarray(train_x, dtype=float)
    pipeline.fit(train_x, np.asarray(train_y))
    # Record normal training-feature ranges to surface distribution shift.
    profile = {'lower': np.quantile(train_x, .01, axis=0).tolist(),
               'upper': np.quantile(train_x, .99, axis=0).tolist()}

    # Select a high-specificity review threshold using *documents* on validation.
    val_scores = np.asarray([_document_score(pipeline, rows[i]['text']) for i in val_i])
    human_val = val_scores[labels[val_i] == 0]
    threshold = min(1.0, max(0.5, float(np.quantile(human_val, .99)) + 1e-6))

    # Evaluate once, using independent test DOCUMENTS (not correlated windows).
    test_scores = np.asarray([_document_score(pipeline, rows[i]['text']) for i in test_i])
    predictions = (test_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels[test_i], predictions, labels=[0, 1]).ravel()
    evaluation = {
        'test_samples': int(len(test_i)),
        'human_test_samples': int(np.sum(labels[test_i] == 0)),
        'ai_test_samples': int(np.sum(labels[test_i] == 1)),
        'false_positive_rate': round(fp / max(fp + tn, 1), 4),
        'precision': round(float(precision_score(labels[test_i], predictions, zero_division=0)), 4),
        'recall': round(float(recall_score(labels[test_i], predictions, zero_division=0)), 4),
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
        'validation_human_samples': int(len(human_val)),
        'train_windows': int(len(train_x)),
        'model_note': 'Held-out, same-corpus document evaluation only. Small validation sets '
                      'and domain/model changes can invalidate measured rates; scores do not prove authorship.',
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'schema_version': MODEL_SCHEMA_VERSION,
        'feature_names': FEATURE_NAMES,
        'pipeline': pipeline,
        'threshold': threshold,
        'min_words': MIN_WORDS,
        'window_words': WINDOW_WORDS,
        'window_stride': WINDOW_STRIDE,
        'evaluation': evaluation,
        'trained_on_group_split': True,
        'training_profile': profile,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }, output)
    return evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV with text,label,group_id')
    args = parser.parse_args()
    print(train(args.data))
