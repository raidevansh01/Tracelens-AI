"""Regression checks for the most confusing real-world UI cases."""
import pymupdf
from core.presentation import source_summary, writing_summary, model_execution_label
from core.report import make_pdf
from core.analyzer import analyze_document
from core.extractor import extract_paste


def test_zero_sources_never_looks_like_a_clean_zero_percent():
    result={'references_checked':0,'coverage':0.,'matches':[],
            'source_modes':{'local':0,'web':0}}
    visible=source_summary(result)
    assert visible['state']=='not_assessed'
    assert visible['coverage_label']=='N/A'
    assert 'not assessed' in visible['headline'].lower()
    assert model_execution_label(result)[0]=='No sources for semantic comparison'


def test_coverage_zero_with_sources_is_not_not_assessed():
    result={'references_checked':2,'coverage':0.,'matches':[],
            'source_modes':{'local':2,'web':0}}
    visible=source_summary(result)
    assert visible['state']=='no_matches'
    assert visible['coverage_label']=='0.00%'
    assert visible['references']==2


def test_separate_writing_classifier_can_run_without_references(monkeypatch):
    import core.analyzer as a
    mock={'status':'experimental','threshold':0.8,
          'windows':[{'start':0,'end':150,'score':0.04,'review_flag':False}],
          'evaluation':{'precision':0.4,'recall':0.5}}
    monkeypatch.setattr(a, 'infer', lambda text: mock)
    result=analyze_document(extract_paste('A small introductory note on modern engineering education.'),
                            [], semantic=False, ai_patterns=True)
    assert result['references_checked']==0
    assert result['ai_pattern_analysis']['status']=='experimental'
    writing=writing_summary(result['ai_pattern_analysis'])
    assert writing['flags']==0
    assert 'does not establish human authorship' in writing['detail']


def test_ai_pattern_no_flags_is_not_called_human_written():
    pat={'status':'experimental','threshold':.79,
         'windows':[{'score':.0037,'review_flag':False},
                    {'score':.0014,'review_flag':False},
                    {'score':.0034,'review_flag':False}]}
    out=writing_summary(pat)
    assert out['count']==3
    assert out['flags']==0
    assert out['peak']==.0037
    assert 'does not establish human authorship' in out['detail']


def test_ai_pattern_abstentions_are_explicit():
    assert writing_summary({'status':'disabled'})['state']=='disabled'
    assert writing_summary({'status':'unavailable', 'message':'model missing'})['state']=='unavailable'
    assert writing_summary({'status':'insufficient_text','message':'120 words'})['state']=='insufficient_text'


def test_report_zero_sources_says_not_assessed():
    data={'title':'test input','references_checked':0,'coverage':0.,'word_count':175,
          'matches':[],'warnings':[],'method':'sample','interpretation':'No inference',
          'ai_pattern_analysis':{'status':'disabled'}}
    pdf=make_pdf(data)
    d=pymupdf.open(stream=pdf,filetype='pdf')
    text=' '.join(page.get_text() for page in d)
    assert 'N/A - no reference sources checked' in text
    assert 'SOURCE COMPARISON NOT ASSESSED' in text
    assert 'Not selected' in text
    d.close()


def test_report_separate_ai_window_scores_appear():
    data={'title':'AI writing fixture','references_checked':0,'coverage':0.,'word_count':577,
          'matches':[],'warnings':[],'method':'sample','interpretation':'No inference',
          'ai_pattern_analysis':{'status':'experimental','threshold':.7985,
           'windows':[{'start':0,'end':200,'score':.0037,'review_flag':False},
                      {'start':120,'end':340,'score':.0014,'review_flag':False}],
           'evaluation':{'precision':1.0,'recall':.6667,'false_positive_rate':0}}}
    pdf=make_pdf(data)
    d=pymupdf.open(stream=pdf,filetype='pdf')
    text=' '.join(p.get_text() for p in d)
    assert 'AI-writing pattern' in text
    assert '0.0037' in text
    assert 'No flags does not prove human' in text.replace('\n',' ') or 'No flags does not prove human' in ' '.join(text.split())
    assert 'false-positive rate=0' in ' '.join(text.split())
    d.close()


def test_semantic_execution_explanation_is_not_ai_authorship():
    result={'references_checked':3,'semantic_used':True}
    label, explanation=model_execution_label(result)
    assert label=='MiniLM executed'
    assert 'source passages' in explanation
