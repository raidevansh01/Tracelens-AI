"""Regression tests for the redesigned UI's reliable, pure-model boundaries."""
import io
import pytest
import pymupdf

from db.store import Store
from core.extractor import extract_paste
from core.report import make_pdf
from ui.logic import check_readiness, result_overview, safe_filename


def kwargs(**changes):
    base = dict(has_document=True, local_enabled=True, reference_count=3,
                online_provider='off', direct_url='', writing_enabled=False,
                external_consent=False, wikipedia_contact='', tavily_configured=False)
    return {**base, **changes}


def test_preflight_requires_an_actual_input():
    status = check_readiness(**kwargs(has_document=False))
    assert not status.can_run and 'Upload' in status.reasons[0]


def test_empty_library_is_blocked_without_other_routes():
    result = check_readiness(**kwargs(reference_count=0))
    assert not result.can_run
    assert any('empty' in reason.lower() for reason in result.reasons)


def test_experimental_only_is_supported_without_reference_documents():
    result = check_readiness(**kwargs(reference_count=0, writing_enabled=True))
    assert result.can_run
    assert 'Experimental writing analysis' in result.checks


def test_external_sharing_requires_explicit_consent():
    denied = check_readiness(**kwargs(reference_count=0, online_provider='both'))
    allowed = check_readiness(**kwargs(reference_count=0, online_provider='both', external_consent=True))
    assert not denied.can_run
    assert allowed.can_run
    assert len(allowed.warnings)>=2  # missing Wikimedia and Tavily configurations


def test_direct_url_alone_is_a_valid_analysis_route():
    state = check_readiness(**kwargs(local_enabled=False,reference_count=0,
                                    direct_url='https://example.org/public', external_consent=True))
    assert state.can_run


def test_result_overview_prefers_v2_and_does_not_fabricate_plagiarism_coverage():
    result = {'references_checked':0,'matches':[], 'ai_pattern_analysis':{'status':'experimental',
        'windows':[{'score':0.001,'review_flag':False}], 'threshold':.75},
        'ai_comparator_analysis':{'models':{'ngram':{'status':'experimental',
        'scores':[.9385,.8967,.8662], 'flags':[True,True,True], 'threshold':.6024}}}}
    summary = result_overview(result)
    assert summary['source']['coverage_label']=='N/A'
    assert summary['writing']['state']=='ready'
    assert summary['writing']['primary']=='ngram'
    assert summary['writing']['flags']==3


def test_safe_filename_blocks_path_segments():
    assert safe_filename('../../secret data.txt') == 'secret_data_tracelens_report.pdf'


def test_v2_reviews_persist_are_isolated_and_exported(tmp_path):
    store=Store(tmp_path/'reviews.sqlite3')
    doc=extract_paste('This document includes enough words for controlled source and model testing.')
    doc['title']='submission.txt'
    id_=store.add_document(doc,'submission')
    result={'title':doc['title'], 'word_count':len(doc['text'].split()),'references_checked':0,
            'coverage':0,'matches':[],'warnings':[], 'ai_pattern_analysis':{'status':'disabled'},
            'ai_comparator_analysis':{'models':{
                'ngram':{'status':'experimental','threshold':0.6,'scores':[.9,.1],
                         'flags':[True,False], 'evaluation':{}},
                'minilm':{'status':'experimental','threshold':0.7,'scores':[.3,.2],
                          'flags':[False,False], 'evaluation':{}}}}}
    aid=store.save_analysis(id_,result)
    store.update_candidate_review(aid,'ngram',0,'Discuss with author','Please inspect drafts')
    store.update_candidate_review(aid,'minilm',0,'No concern','Independent review')
    reopened=Store(tmp_path/'reviews.sqlite3')
    assert reopened.candidate_reviews_for(aid,'ngram')[0]['decision']=='Discuss with author'
    assert reopened.candidate_reviews_for(aid,'minilm')[0]['decision']=='No concern'
    assert reopened.candidate_reviews_for(aid,'ngram')[0]['note']=='Please inspect drafts'
    assert reopened.writing_reviews_for(aid)=={}
    with pytest.raises(ValueError):
        reopened.update_candidate_review(aid,'ngram',2,'No concern')
    with pytest.raises(ValueError):
        reopened.update_candidate_review(aid,'fake',0,'No concern')
    with pytest.raises(ValueError):
        reopened.update_candidate_review(aid,'ngram',0,'Definitely AI')
    pdf=make_pdf(result,{}, {},reopened.candidate_reviews_for(aid))
    d=pymupdf.open(stream=pdf,filetype='pdf')
    contents='\n'.join(p.get_text() for p in d)
    d.close()
    assert 'Please inspect drafts' in contents
    assert 'Independent review' in contents
    assert reopened.delete_analysis(aid)
    assert reopened.candidate_reviews_for(aid)=={}
