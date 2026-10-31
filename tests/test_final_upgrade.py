"""Offline acceptance tests for combined web search and independent AI-writing reviews."""
from pathlib import Path
import pytest
import fitz

from core.analyzer import analyze_document
from core.extractor import extract_paste
from core.report import make_pdf
from db.store import Store


def _doc(text, title):
    return {**extract_paste(text), 'title': title}


def test_both_search_providers_and_cross_provider_url_dedup(monkeypatch):
    import core.analyzer as a
    submitted = _doc('Scientific research increasingly uses machine learning models to detect meaningful patterns in complex data collected from laboratories and experiments.', 'test')
    text = ('Scientific research increasingly uses machine learning models to detect meaningful patterns in complex data collected from laboratories and experiments. '
            'These findings are analyzed using reproducible evaluation procedures in universities.')
    shared_url = 'https://en.wikipedia.org/wiki/Machine_learning'
    wiki = {'id':'web:'+shared_url, 'title':'Machine learning — Wikipedia', 'type':'web',
            'provider':'wikipedia', 'url':shared_url,'text':text,'pages':[{'page':1,'start':0,'end':len(text)}]}
    tavily_same = {**wiki,'provider':'tavily','title':'Machine learning (Tavily result)'}
    separate_url='https://example.org/test-article'
    tavily_other={**wiki,'id':'web:'+separate_url,'url':separate_url,'provider':'tavily',
                  'title':'Another article'}
    calls=[]
    def wiki_fn(*args,**kwargs):
        calls.append('wiki')
        return [dict(wiki)], []
    def tavily_fn(*args,**kwargs):
        calls.append('tavily')
        return [dict(tavily_same),dict(tavily_other)], []
    monkeypatch.setattr(a,'search_wikipedia',wiki_fn)
    monkeypatch.setattr(a,'search_online',tavily_fn)
    result=analyze_document(submitted,[],semantic=False,online=True,online_provider='both',wikipedia_contact='contact@example.com')
    assert calls==['wiki','tavily']
    assert result['provider_status']['wikipedia']['retrieved']==1
    assert result['provider_status']['tavily']['retrieved']==2
    assert result['source_modes']['web']==2
    assert result['references_checked']==2
    shared=next(s for s in result['source_inventory'] if s['url']==shared_url)
    assert set(shared['discovered_via']) == {'wikipedia','tavily'}
    assert result['matches']


def test_both_search_partial_failure_keeps_other_provider(monkeypatch):
    import core.analyzer as a
    doc=_doc('Research teams use statistical machine learning models to analyze information and find patterns within large datasets.','test')
    def fail(*a,**k):
        raise RuntimeError('service unreachable')
    source = {**doc,'id':'w1','type':'web','provider':'wikipedia',
              'url':'https://en.wikipedia.org/wiki/Statistics','title':'Statistics'}
    monkeypatch.setattr(a,'search_wikipedia',lambda *a,**k:([source],[]))
    monkeypatch.setattr(a,'search_online',fail)
    result=analyze_document(doc,[],online=True,online_provider='both',semantic=False)
    assert result['source_modes']['web']==1
    assert result['provider_status']['tavily']['state']=='failed'
    assert result['provider_status']['wikipedia']['state']=='retrieved'
    assert any('Tavily' in warning for warning in result['warnings'])


def test_both_search_unconfigured_does_not_claim_internet_checked(monkeypatch):
    import core.analyzer as a
    monkeypatch.setattr(a,'search_wikipedia',lambda *args,**kw:([],['Missing contact']))
    monkeypatch.setattr(a,'search_online',lambda *args,**kw:([],['Missing key']))
    result=analyze_document(_doc('A sufficiently long test passage discussing machine learning and computational data analysis in different scientific projects.','x'),[],semantic=False,online=True,online_provider='both')
    assert result['references_checked']==0
    assert result['source_modes']['web']==0
    assert any('not a clean internet' in warning.lower() for warning in result['warnings'])
    assert len(result['provider_status'])==2


def _writing_result():
    return {'title':'example.txt','coverage':0,'word_count':300,'references_checked':0,
            'matches':[],'warnings':[],'ai_pattern_analysis':{
            'status':'experimental','threshold':0.78,
            'windows':[{'start':0,'end':500,'score':0.83,'review_flag':True},
                       {'start':450,'end':950,'score':0.02,'review_flag':False}],
            'evaluation':{'precision':0.77,'recall':0.6,'false_positive_rate':0.0}}}


def test_writing_review_persists_is_independent_and_migrates_old_database(tmp_path):
    database=tmp_path/'legacy.db'
    store=Store(database)
    d=store.add_document(_doc('A document with enough content to test the writing reviewer.','a.txt'),kind='submission')
    aid=store.save_analysis(d,_writing_result())
    store.update_review(aid,'source-match-1','Needs investigation','possible copied text')
    store.update_writing_review(aid,0,'Discuss with author','Student should be asked about drafting process')
    store.update_writing_review(aid,1,'No concern','Human reviewer found no other supporting evidence')
    reopened=Store(database)
    reviews=reopened.writing_reviews_for(aid)
    assert reviews[0]['decision']=='Discuss with author'
    assert reviews[1]['decision']=='No concern'
    assert reopened.reviews_for(aid)['source-match-1']['decision']=='Needs investigation'
    assert reopened.get_analysis(aid)['result']['ai_pattern_analysis']['windows'][0]['score']==0.83
    with pytest.raises(ValueError):
        reopened.update_writing_review(aid,2,'No concern')
    with pytest.raises(ValueError):
        reopened.update_writing_review(aid,0,'Certainly AI-written')
    assert reopened.delete_analysis(aid)
    assert reopened.writing_reviews_for(aid)=={}


def test_pdf_contains_ai_review_and_provider_status():
    result=_writing_result()
    result['provider_status']={'wikipedia':{'retrieved':1,'state':'retrieved'},
                               'tavily':{'retrieved':0,'state':'failed'}}
    result['online_requested']=True
    pdf=make_pdf(result, {}, {0:{'decision':'Discuss with author','note':'Check draft history'},
                              1:{'decision':'No concern','note':'No corroborating evidence'}})
    doc=fitz.open(stream=pdf,filetype='pdf')
    all_text='\n'.join(page.get_text() for page in doc)
    assert 'Discuss with author' in all_text
    assert 'Check draft history' in all_text
    assert 'No concern' in all_text
    assert 'W ikipedia' not in all_text
    assert 'Tavily' in all_text
    assert 'N/A' in all_text
    doc.close()


def test_flagged_text_share_counts_overlapping_words_once():
    from core.presentation import writing_flagged_share
    text='one two three four five six seven eight nine ten'
    pattern={'status':'experimental','windows':[
        {'start':0,'end':len('one two three four five six'),'review_flag':True},
        {'start':len('one two three '),'end':len(text),'review_flag':False}]}
    assert writing_flagged_share(pattern,text)==60.0
    assert writing_flagged_share({'status':'unavailable'},text) is None
