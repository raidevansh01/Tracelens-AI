"""EXPERIMENTAL stylometry-only writing-pattern classifier.

It flags similarity to distributions in a labeled training dataset; it cannot
establish AI authorship. Version 2 uses exactly the same fixed-length analysis
windows for training and inference, and deliberately excludes raw document
length from the classifier features.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from .extractor import TOKEN_RE, word_tokens

MODEL_SCHEMA_VERSION = 2
MIN_WORDS = 120
WINDOW_WORDS = 220
WINDOW_STRIDE = 180
FEATURE_NAMES = [
    'mean_sentence_words', 'std_sentence_words', 'lexical_diversity',
    'repeat_bigram_ratio', 'mean_word_length', 'comma_per_word',
    'function_word_ratio',
]
FUNCTION_WORDS = set(
    'the a an and or if but is are was were be been to of in on for with at '
    'by from as this that it its they we our you your'.split()
)
MODEL_FILE = Path(__file__).resolve().parents[1] / 'storage' / 'ai_pattern_model.joblib'


def features(text: str) -> list[float]:
    tokens = word_tokens(text)
    n = max(len(tokens), 1)
    sentence_lens = [
        len(word_tokens(s)) for s in re.split(r'[.!?]+', text) if word_tokens(s)
    ]
    bigrams = list(zip(tokens, tokens[1:]))
    duplicate_bigram = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.
    return [
        float(np.mean(sentence_lens)) if sentence_lens else 0.,
        float(np.std(sentence_lens)) if sentence_lens else 0.,
        len(set(tokens)) / n,
        duplicate_bigram,
        sum(map(len, tokens)) / n,
        text.count(',') / n,
        sum(t in FUNCTION_WORDS for t in tokens) / n,
    ]


def make_windows(text: str, *, min_words: int = MIN_WORDS,
                 window_words: int = WINDOW_WORDS,
                 stride: int = WINDOW_STRIDE) -> list[dict]:
    """Offset-addressable overlapping windows, shared by training and inference.

    A final partial window is included only if it reaches min_words.
    """
    if min_words <= 0 or window_words < min_words or not (0 < stride <= window_words):
        raise ValueError('Invalid text window parameters')
    words = list(TOKEN_RE.finditer(text))
    windows = []
    for start in range(0, len(words), stride):
        end = min(start + window_words, len(words))
        if end - start < min_words:
            break
        a, b = words[start].start(), words[end - 1].end()
        windows.append({'start': a, 'end': b, 'text': text[a:b], 'words': end - start})
        if end == len(words):
            break
    return windows


def infer(text: str, model_path: str | Path = MODEL_FILE) -> dict:
    import joblib
    p = Path(model_path)
    if not p.exists():
        return {
            'status': 'unavailable',
            'message': 'No experimental writing-pattern model installed. Train a labeled model first.',
        }
    # Only load a trusted, locally trained joblib file. Joblib deserialization
    # executes code from pickles and must never be used on untrusted files.
    item = joblib.load(p)
    if item.get('schema_version') != MODEL_SCHEMA_VERSION or item.get('feature_names') != FEATURE_NAMES:
        return {
            'status': 'unavailable',
            'message': 'Outdated classifier: the previous model used incompatible whole-document features. '
                       'Back up storage/ai_pattern_model.joblib and retrain with the v1.4 training script.',
        }
    if not item.get('evaluation') or not item.get('trained_on_group_split'):
        return {'status': 'unavailable', 'message': 'Classifier has no documented group-separated evaluation.'}
    windows = make_windows(text)
    if not windows:
        return {
            'status': 'insufficient_text',
            'message': f'At least {MIN_WORDS} words required for this experimental model.',
        }
    feature_matrix = np.asarray([features(w['text']) for w in windows], dtype=float)
    scores = item['pipeline'].predict_proba(feature_matrix)[:, 1]
    profile = item.get('training_profile')
    threshold = float(item['threshold'])
    outliers = []
    for vector in feature_matrix:
        if profile and len(profile.get('lower', [])) == len(FEATURE_NAMES):
            lower=np.asarray(profile['lower'],dtype=float)
            upper=np.asarray(profile['upper'],dtype=float)
            outliers.append([FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES))
                             if vector[i] < lower[i] or vector[i] > upper[i]])
        else:
            outliers.append([])
    return {
        'status': 'experimental',
        'threshold': threshold,
        'training_profile_available': bool(profile),
        'windows': [
            {'start': w['start'], 'end': w['end'], 'score': round(float(s), 4),
             'review_flag': bool(float(s) >= threshold) and len(outside)<2,
             'unusual_features': outside,
             'distribution_warning': len(outside)>=2,
             'outcome': ('Inconclusive: outside training range' if len(outside)>=2 else
                         'Review suggested' if float(s)>=threshold else 'Not flagged')}
            for w, s, outside in zip(windows, scores, outliers)
        ],
        'evaluation': item['evaluation'],
        'message': 'Experimental stylistic similarity only, NOT a reliable authorship determination. '
                   'Scores are sensitive to writing domain and AI generator.',
    }
