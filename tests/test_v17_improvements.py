"""Regression tests for v1.7 source discovery and uncertainty diagnostics."""
from core.extractor import extract_paste, segment
from core.web_sources import select_search_queries
from core.wikipedia import _page_from_api
from core.detection import match_documents
from core.ai_patterns import features, infer
from train_ai_patterns import train
from unittest.mock import patch
import csv
import joblib
from pathlib import Path


def test_short_literal_and_broad_queries():
    text=('Original educators developed a machine learning approach to predict '
          'student achievement through carefully constructed mathematical models.')
    result=select_search_queries(segment(text),limit=4)
    assert len(result)>=2
    assert result[0].startswith('"') and result[0].endswith('"')
    assert len(result[0].split())<=10
    assert any('"' not in x for x in result)
    assert all(len(x)<=180 for x in result)


def test_wikipedia_parse_extracts_article_body(monkeypatch):
    from core import wikipedia
    class FakeSession: pass
    def fake_api(session,params,headers):
        assert params['action']=='parse'
        return {'parse':{'title':'Machine learning', 'text':
            '<div class="mw-parser-output"><p>Machine learning systems discover relationships in data through mathematical models and examples gathered during repeated experiments.</p>'
            '<p>Students evaluate predictions using performance metrics and independent validation cases to measure the quality of an algorithm.</p>'
            '<table><tr><td>Unrelated sidebar text</td></tr></table>'
            '</div>'}}
    monkeypatch.setattr(wikipedia,'_api_get',fake_api)
    src=_page_from_api(FakeSession(),123,{'User-Agent':'test'})
    assert src and 'Machine learning systems' in src['text']
    assert 'Unrelated sidebar text' not in src['text']
    assert src['url'].endswith('/Machine_learning')


def test_long_reference_tail_is_considered():
    before=' '.join(f'Topic {i} describes many different unrelated items and concepts in a complicated paragraph. ' for i in range(280))
    target='Distinctive laboratory authors measured photon scattering with a calibrated instrument in the final experimental measurement.'
    doc=extract_paste(target)
    src={**extract_paste(before+target), 'id':'w','title':'Long source','type':'web'}
    result=match_documents(doc,[src],semantic=False)
    assert result['coverage']>0
    assert any('Distinctive laboratory' in m['source_text'] for m in result['matches'])


def test_profile_diagnostics_preserve_compatibility(tmp_path):
    from core.ai_patterns import FEATURE_NAMES
    def _example(i, ai):
        base=('Furthermore this framework presents a structured analysis of interconnected relationships and a comprehensive overview of relevant aspects.' if ai else 'Yesterday I went to the store, bumped into my friend, and forgot why I left the house.')
        return ' '.join([base] * (13+(i % 3))) + f' Independent topic identifier {i}.'
    f=tmp_path/'train.csv'
    with f.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['text','label','group_id'])
        writer.writeheader()
        for i in range(50):
            for ai in (0,1):
                writer.writerow({'text':_example(i,ai),'label':ai,'group_id':f'pair_{i}'})
    p=tmp_path/'model.joblib'
    train(str(f),p)
    item=joblib.load(p)
    assert len(item['training_profile']['lower']) == 7
    result=infer(_example(5,1),p)
    assert result['training_profile_available'] is True
    assert 'unusual_features' in result['windows'][0]


def test_url_tracking_parameters_deduplicate():
    from core.analyzer import _source_identity
    plain={'id':'1','url':'https://example.org/article?utm_source=chatgpt#top'}
    clean={'id':'2','url':'https://example.org/article'}
    assert _source_identity(plain)==_source_identity(clean)


def test_empty_search_query_selection():
    assert select_search_queries(segment('hello world'))==[]
