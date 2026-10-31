"""TraceLens v1.7 diagnostic: offline source matching, optional LIVE source providers,
and an independent known-AI writing sample. No API key is printed.

Run from project root:
 python qa/health_check_v17.py
 python qa/health_check_v17.py --live-wikipedia
 python qa/health_check_v17.py --live-tavily   # external text query; may use credits
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from core.extractor import extract_paste, segment
from core.analyzer import analyze_document
from core.ai_patterns import infer


def run(live_wikipedia=False, live_tavily=False):
    text=('Researchers use machine learning to discover patterns in complex data. '
          'These methods allow predictive models to improve through experience rather '
          'than requiring explicit instructions for each possible observation.')
    local={**extract_paste(text), 'id':'local-test','title':'Verified local source','type':'local'}
    result=analyze_document(extract_paste(text),[local],semantic=False)
    ok=result['coverage']>0 and result['references_checked']==1
    print(f"[{'PASS' if ok else 'FAIL'}] Offline source matching: {result['coverage']:.2f}% covered; "
          f"{len(result['matches'])} match(es)")
    known=(ROOT/'TEST_AI_GENERATED.txt')
    if known.exists():
        prediction=infer(known.read_text(encoding='utf-8'))
        if prediction['status']=='experimental':
            scores=[w['score'] for w in prediction['windows']]
            flags=sum(w['review_flag'] for w in prediction['windows'])
            print(f'[AI RESEARCH CHECK] Known AI-written sample: {len(scores)} windows; '
                  f'{flags} flagged; threshold={prediction["threshold"]:.4f}; scores={scores}')
            if not flags:
                print('[KNOWN MISS] Classifier failed to flag this known AI-written sample. '
                      'This is a real detection failure, not an installation failure.')
        else:
            print('[AI NOT VERIFIED] '+prediction.get('message',prediction['status']))
    if live_wikipedia:
        from core.wikipedia import wikipedia_url_source, search_wikipedia
        url='https://en.wikipedia.org/wiki/Machine_learning'
        try:
            source=wikipedia_url_source(url)
            print('[WIKIPEDIA DIRECT PASS] Fetched article; characters:',len(source['text']))
            text=source['text'][:800]
            if len(text.split())<25:
                print('[WIKIPEDIA CHECK] Intro too short for a comparison.')
            else:
                submitted=extract_paste(text)
                matches=analyze_document(submitted,[source],semantic=False)
                print('[WIKIPEDIA MATCH] matched coverage:',matches['coverage'],
                      'source findings:',len(matches['matches']))
            sources,warnings=search_wikipedia(segment(extract_paste(text)['text']),max_queries=2)
            print('[WIKIPEDIA SEARCH] retrieved:',len(sources),'warnings:',warnings)
        except Exception as exc:
            print('[WIKIPEDIA FAIL]',type(exc).__name__,str(exc)[:240])
    if live_tavily:
        from core.web_sources import search_online, select_search_queries
        doc=extract_paste('Artificial intelligence helps computers learn patterns from data. '
             'Machine learning is used for classification, forecasting, and scientific research. '
             'Researchers train models on example datasets to discover useful relationships.')
        passages=segment(doc['text'])
        print('[TAVILY] Sending selected search queries externally; usage credits may be charged.')
        print('[TAVILY] Selected queries:',select_search_queries(passages))
        sources,warnings=search_online(passages,max_queries=3)
        print('[TAVILY SEARCH] full-text pages:',len(sources),'warnings:',warnings)
        # A full-text retrieval without a relevant match is not a web-coverage guarantee.
    return 0 if ok else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live-wikipedia',action='store_true')
    parser.add_argument('--live-tavily',action='store_true')
    args=parser.parse_args()
    raise SystemExit(run(args.live_wikipedia,args.live_tavily))
