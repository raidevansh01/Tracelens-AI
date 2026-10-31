"""Classifier v2 regression tests: format and behavior, NOT real authorship accuracy."""
from pathlib import Path
import csv
import joblib
import numpy as np
from core.ai_patterns import make_windows, features, infer, FEATURE_NAMES, MODEL_SCHEMA_VERSION
from train_ai_patterns import train


def test_windows_cover_supported_text_with_original_offsets():
    txt = ' '.join(f'word{n}' for n in range(577))
    windows = make_windows(txt)
    assert len(windows) == 3
    assert [w['words'] for w in windows] == [220, 220, 217]
    assert all(txt[w['start']:w['end']] == w['text'] for w in windows)
    assert all(len(features(w['text'])) == len(FEATURE_NAMES) for w in windows)
    assert 'paragraph_wordcount' not in FEATURE_NAMES


def test_insufficient_text(tmp_path):
    assert make_windows('short content only') == []
    assert infer('short content', tmp_path/'missing.joblib')['status'] == 'unavailable'


def test_v1_model_requires_retraining(tmp_path):
    outdated = tmp_path/'outdated.joblib'
    joblib.dump({'evaluation': {'test_samples': 2}, 'trained_on_group_split': True}, outdated)
    result = infer(' '.join(['apple'] * 125), outdated)
    assert result['status'] == 'unavailable'
    assert 'Outdated classifier' in result['message']


def _example(i, ai):
    # ONLY synthetic data for a plumbing test. Not evidence of real detection.
    if ai:
        base = 'Furthermore this framework presents a structured analysis of interconnected relationships and a comprehensive overview of relevant aspects.'
    else:
        base = 'Yesterday I went to the store, bumped into my friend, and forgot why I had left the house in the first place.'
    # Form 130+ words and vary data per independent prompt group.
    return (' '.join([base] * (13+(i % 3)))) + f' Independent topic identifier {i}.'


def test_training_and_inference_use_same_window_shape(tmp_path):
    csv_file = tmp_path/'dataset.csv'
    with csv_file.open('w',encoding='utf-8',newline='') as f:
        out = csv.DictWriter(f, fieldnames=['text','label','group_id'])
        out.writeheader()
        for i in range(50):
            for ai in (0,1):
                out.writerow({'text':_example(i,ai),'label':ai,'group_id':f'prompt_{i:03d}'})
    model_file = tmp_path/'model.joblib'
    result = train(str(csv_file),model_file)
    model = joblib.load(model_file)
    assert model['schema_version'] == MODEL_SCHEMA_VERSION
    assert model['feature_names'] == FEATURE_NAMES
    assert model['evaluation']['test_samples'] == 20
    assert model['evaluation']['train_windows'] > 0
    assert result['human_test_samples'] == 10
    assert result['ai_test_samples'] == 10
    text = ' '.join([_example(2,1),_example(3,1),_example(4,1)])
    output=infer(text,model_file)
    assert output['status'] == 'experimental'
    assert len(output['windows']) >= 2
    assert all(0 <= x['score'] <= 1 for x in output['windows'])
    assert all(isinstance(x['review_flag'],bool) for x in output['windows'])
