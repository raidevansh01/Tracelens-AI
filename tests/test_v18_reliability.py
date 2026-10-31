"""Acceptance tests focused on reported false-negative failure modes."""
import csv
from pathlib import Path
from core.extractor import extract_paste,segment,word_tokens
from core.detection import match_documents
from core.wikipedia import wikipedia_queries
from core.web_sources import select_search_queries
from core.ai_patterns import infer
from core.ai_comparator import inspect_candidates
from train_ai_comparator import train


def test_copied_tail_of_wikipedia_article_finds_exact_copy_when_tfidf_budget_exceeded():
    prefix=('A technical reference contains many unique numerical entries, measurements, and historical details. ' * 1600)
    quote=('The experimental library traces sources using exact consecutive tokens even when a paragraph '
           'appears deep within a lengthy reference article and requires an accurate citation.')
    suffix=(' Another extensive section discusses independent technical observations and methodological limits.'*40)
    source={**extract_paste(prefix+quote+suffix,title='Long Wikipedia article'),
            'id':'wiki:page','type':'web','url':'https://en.wikipedia.org/wiki/Example'}
    user=extract_paste('The professor asked students to explain findings carefully. '+quote+' It also includes a new independent passage.')
    result=match_documents(user,[source],semantic=False)
    assert result['coverage']>55
    matched=[m for m in result['matches'] if m['category']=='Exact wording overlap']
    assert matched
    assert any('experimental library traces sources' in m['source_text'] for m in matched)
    assert all(m['source_text'][a:b] in m['source_text'] for m in matched for a,b in m['source_spans'])


def test_exact_word_comparison_survives_punctuation_and_case():
    quote='Researchers collected detailed measurements to examine how historical patterns shaped the final scientific conclusions.'
    source={**extract_paste('Preface. '+quote+' End.'),'id':'x','type':'web'}
    user=extract_paste(quote.upper().replace(' measurements ',' measurements, '))
    result=match_documents(user,[source],semantic=False)
    assert result['coverage']==100.0
    assert any(m.get('exact_word_count',0)>=12 for m in result['matches'])


def test_search_queries_use_distinctive_literal_phrases_from_multiple_locations():
    text=('Observations of tectonic plate movements provided crucial evidence about the seismic activity reported across the region. '
          'A second sentence clarifies the interpretation of historical earthquake records and measurements. '
          'Scientific investigators subsequently documented precise differences across volcanic rock samples and experimental laboratories.')
    pieces=segment(text)
    wiki=wikipedia_queries(pieces,limit=6)
    tavily=select_search_queries(pieces,limit=6)
    assert sum(q.startswith('"') for q in wiki)>=2
    assert sum(q.startswith('"') for q in tavily)>=2
    assert any('volcanic' in q or 'rock' in q for q in wiki)


def test_comparator_train_infer_separate_model(tmp_path):
    csvfile=tmp_path/'train.csv'
    with csvfile.open('w',newline='',encoding='utf8') as out:
        writer=csv.DictWriter(out,fieldnames=['text','label','group_id'])
        writer.writeheader()
        for i in range(36):
            writer.writerow({'group_id':str(i),'label':0,
                             'text':('I visited our classroom on Thursday and discussed the book with friends. '
                                     'Later our teacher showed students an interesting problem and everyone tried a solution. ')*10 + f'number {i}'})
            writer.writerow({'group_id':str(i),'label':1,
                             'text':('Furthermore the comprehensive analysis elucidates multifaceted implications for systematic academic evaluation. '
                                     'Consequently this structured framework provides a nuanced examination of diverse theoretical perspectives. ')*10+f'number {i}'})
    p=tmp_path/'candidate.joblib'
    report=train(csvfile,method='ngram',output=p)
    assert p.exists()
    assert report['evaluation']['samples']>4
    result=inspect_candidates(('Furthermore the comprehensive analysis elucidates implications. '*35),paths={'ngram':p})
    assert result['status']=='available'
    assert result['models']['ngram']['status']=='experimental'
    assert len(result['models']['ngram']['scores'])>=1


def test_optional_models_do_not_disable_existing_classifier(tmp_path):
    r=inspect_candidates('Testing '*200,paths={'ngram':tmp_path/'not_trained.joblib'})
    assert r['status']=='not_trained'
    assert r['models']=={}
