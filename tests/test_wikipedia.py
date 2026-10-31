"""Offline provider contract tests; no Wikimedia network calls."""
import pytest

from core import wikipedia
from core.analyzer import analyze_document
from core.extractor import extract_paste


class FakeSession:
    def __enter__(self): return self
    def __exit__(self, *args): return None


def test_missing_wikimedia_contact_explains_setup(monkeypatch):
    monkeypatch.delenv('WIKIMEDIA_CONTACT', raising=False)
    sources, warnings = wikipedia.search_wikipedia([{'text':'Artificial intelligence has many applications','window':1}])
    assert not sources
    assert any('WIKIMEDIA_CONTACT' in warning for warning in warnings)


def test_wikipedia_search_returns_only_extracted_sources(monkeypatch):
    monkeypatch.setattr(wikipedia.requests, 'Session', FakeSession)
    calls=[]
    def fake_get(session, params, headers):
        assert 'my-college.edu' in headers['User-Agent']
        calls.append(params)
        if params.get('list')=='search':
            return {'query':{'search':[{'pageid':42,'title':'Artificial intelligence'},
                                       {'pageid':53,'title':'Short source'}]}}
        if params.get('pageids')=='42':
            return {'query':{'pages':[{'pageid':42,'title':'Artificial intelligence',
                'extract':('Artificial intelligence is the capability of computational '
                           'systems to perform tasks that are typically associated with '
                           'human intelligence, such as learning, reasoning and perception. '
                           'Modern systems combine machine learning and statistical methods.')}]}}
        return {'query':{'pages':[{'pageid':53,'title':'Short source','extract':'too short'}]}}
    monkeypatch.setattr(wikipedia,'_api_get',fake_get)
    passages=[{'text':'Artificial intelligence technology helps researchers evaluate computational reasoning systems in multiple scientific applications.', 'window':1}]
    sources, warnings=wikipedia.search_wikipedia(passages, contact='https://my-college.edu')
    assert len(sources)==1
    assert sources[0]['title']=='Artificial intelligence — Wikipedia'
    assert sources[0]['url']=='https://en.wikipedia.org/wiki/Artificial_intelligence'
    assert sources[0]['type']=='web'
    assert len(calls)>=6  # Diversified queries plus parse/fallback per unique page
    assert any('no usable' in w for w in warnings)


def test_direct_wikipedia_url_uses_wikipedia_api(monkeypatch):
    monkeypatch.setattr(wikipedia.requests, 'Session', FakeSession)
    def fake_get(session, params, headers):
        if params.get('prop')=='info':
            assert params['titles']=='Machine learning'
            return {'query':{'pages':[{'pageid':99,'title':'Machine learning'}]}}
        return {'query':{'pages':[{'pageid':99,'title':'Machine learning',
                  'extract':'Machine learning approaches enable computer systems to learn patterns '
                            'from experience without explicitly programming individual decision rules. '
                            'Many statistical methods can build predictive models from training data.'}]}}
    monkeypatch.setattr(wikipedia,'_api_get',fake_get)
    result=wikipedia.wikipedia_url_source('https://en.wikipedia.org/wiki/Machine_learning', contact='contact@sample.edu')
    assert result['id'].startswith('web:https://en.wikipedia.org')
    assert 'Machine learning approaches' in result['text']
    with pytest.raises(wikipedia.WikipediaError):
        wikipedia.wikipedia_url_source('https://evil.wikipedia.org/wiki/Machine_learning', contact='contact@sample.edu')


def test_wikipedia_provider_failure_does_not_block_local(monkeypatch):
    import core.analyzer as analyzer
    monkeypatch.setattr(analyzer,'search_wikipedia',lambda *args,**kwargs:([],['Wikipedia temporary failure']))
    submitted=extract_paste('Artificial intelligence is the ability of computers to perform complex tasks with mathematical and statistical models.')
    reference={**extract_paste('Artificial intelligence is the ability of computers to perform complex tasks with mathematical and statistical models.'),
               'id':'source1', 'type':'local', 'title':'Local study'}
    result=analyze_document(submitted,[reference],online=True, online_provider='wikipedia', semantic=False)
    assert result['source_modes']['local']==1
    assert any('Wikipedia temporary failure' in w for w in result['warnings'])
    assert result['references_checked']==1
