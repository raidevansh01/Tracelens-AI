from core.v2_presentation import writing_v2_snapshot


def test_ngram_priority():
    report = {'ai_comparator_analysis': {'models': {
        'minilm': {'status': 'experimental', 'scores': [0.5], 'flags': [False], 'threshold': 0.7},
        'ngram': {'status': 'experimental', 'scores': [0.91], 'flags': [True], 'threshold': 0.6,
                  'model_version': 'v2-experimental'}}}}
    v = writing_v2_snapshot(report)
    assert v['state'] == 'ready'
    assert v['primary'] == 'ngram'
    assert v['flags'] == 1
    assert v['model_version'] == 'v2-experimental'


def test_old_analysis_without_candidates():
    v = writing_v2_snapshot({'ai_pattern_analysis': {'status': 'experimental'}})
    assert v['state'] == 'missing'
    assert 'NEW analysis' in v['message']


def test_secondary_fallback_and_invalid():
    v = writing_v2_snapshot({'ai_comparator_analysis': {'models': {
        'ngram': {'status': 'unavailable', 'message': 'missing'},
        'minilm': {'status': 'experimental', 'scores': [0.2,0.9], 'flags': [False, True], 'threshold': 0.7}}}})
    assert v['primary'] == 'minilm' and v['count'] == 2 and v['flags'] == 1
    invalid = writing_v2_snapshot({'ai_comparator_analysis': {'models': {
        'ngram': {'status': 'experimental', 'scores':[0.9], 'flags':[], 'threshold':0.4}}}})
    assert invalid['state'] == 'invalid'
