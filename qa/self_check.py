"""Run reproducible QA against ONE submission and bundled local references.

Usage: python qa/self_check.py [--with-ai]
No database changes; report and results written under qa/output/.
External Tavily and Streamlit UI require independent live/manual checks.
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.analyzer import analyze_document
from core.extractor import ExtractionError, extract_upload, extract_paste, segment
from core.report import make_pdf
from db.store import Store

TEST_DOCUMENT = ROOT / 'qa' / 'ONE_FILE_FEATURE_TEST.txt'
REFERENCE_FILES = [ROOT / 'samples' / 'library_ml.txt',
                   ROOT / 'samples' / 'library_energy.txt',
                   ROOT / 'qa' / 'qa_reference_common.txt']


def run(with_ai: bool = False) -> int:
    results=[]
    failures=[]
    def record(name: str, condition: bool, details: str = ''):
        status='PASS' if condition else 'FAIL'
        results.append({'feature':name,'status':status,'details':details})
        print(f'{status:4} | {name}'+(f' — {details}' if details else ''))
        if not condition:
            failures.append(name)

    doc=extract_upload(TEST_DOCUMENT.name, TEST_DOCUMENT.read_bytes())
    refs=[]
    for filename in REFERENCE_FILES:
        reference=extract_upload(filename.name,filename.read_bytes())
        refs.append({**reference,'id':filename.name,'type':'local'})

    record('Single TXT document extraction',doc['kind']=='txt' and len(doc['text'])>500)
    passages=segment(doc['text'],doc['pages'])
    record('Text segmentation and original offsets',bool(passages) and all(doc['text'][p['start']:p['end']]==p['text'] for p in passages),f'{len(passages)} query windows')
    result=analyze_document(doc,refs,semantic=with_ai,ai_patterns=True)
    record('Local reference library',result['references_checked']==3,'3 controlled sources')
    record('Exact copied source A',any(m['source_title']=='library_ml.txt' and m['category']=='Exact wording overlap' for m in result['matches']))
    record('Exact copied source B',any(m['source_title']=='library_energy.txt' and m['category']=='Exact wording overlap' for m in result['matches']))
    record('Low-information phrase flagged separately',any(m['category']=='Low-information overlap' for m in result['matches']))
    record('Citation/quotation indicators',any('Quotation marks present' in m['citation_signals'] and 'Author–year citation pattern' in m['citation_signals'] for m in result['matches']))
    record('Exact coverage is neither zero nor 100%',0<result['coverage']<100,f'{result["coverage"]}%')
    record('Original/unrelated passage not wrongly flagged',not any('cobalt umbrellas' in m['submitted_text'] for m in result['matches']))
    spans_ok=True
    for m in result['matches']:
        for a,b in m['input_spans']:
            if not 0<=a<b<=len(m['submitted_text']): spans_ok=False
        for a,b in m['source_spans']:
            if not 0<=a<b<=len(m['source_text']): spans_ok=False
    record('Highlight offsets in bounds',spans_ok)
    record('TF-IDF scores valid',bool(result['matches']) and all(0<=m['lexical_score']<=1 for m in result['matches']))

    if with_ai:
        record('MiniLM actually ran',result['semantic_used'] is True,
               'Check model download and requirements-ai.txt if this fails')
        if result['semantic_used']:
            record('Semantic scores exist',any(m['semantic_score'] is not None for m in result['matches']))
    else:
        results.append({'feature':'MiniLM live inference','status':'NOT TESTED',
                        'details':'Rerun with --with-ai after installing requirements-ai.txt'})
        print('SKIP | MiniLM live inference — rerun with --with-ai')

    pattern=result['ai_pattern_analysis']
    record('Experimental writing module returns honest status',pattern.get('status') in ('unavailable','insufficient_text','experimental'))
    if pattern.get('status') != 'experimental':
        print('SKIP | AI-generated text classification — no evaluated classifier is available')
        results.append({'feature':'AI-authorship reliability','status':'NOT TESTED',
                        'details':'Requires independently labeled, evaluated human and AI writing data'})

    try:
        with tempfile.TemporaryDirectory() as td:
            store=Store(Path(td)/'qa.sqlite3')
            for ref in refs: store.add_document(ref,'reference')
            sub_id=store.add_document(doc,'submission')
            analysis_id=store.save_analysis(sub_id,result)
            record('SQLite saving and retrieval',store.get_analysis(analysis_id)['result']['coverage']==result['coverage'])
            first=result['matches'][0]
            store.update_review(analysis_id,first['id'],'Needs investigation','QA: reviewed source evidence.')
            record('Human review persists',store.reviews_for(analysis_id)[first['id']]['decision']=='Needs investigation')
            pdf=make_pdf(result,store.reviews_for(analysis_id))
            import pymupdf as fitz
            parsed=fitz.open(stream=pdf,filetype='pdf')
            report_text='\n'.join(p.get_text() for p in parsed)
            record('Downloadable PDF report content',len(pdf)>1000 and 'TraceLens AI' in report_text and 'Needs investigation' in report_text)
            parsed.close()
            output=ROOT/'qa'/'output'
            output.mkdir(exist_ok=True)
            (output/'feature_evidence_report.pdf').write_bytes(pdf)
            record('Analysis history deletion',store.delete_analysis(analysis_id) and store.get_document(sub_id) is None)
    except Exception as e:
        record('SQLite / PDF report workflow',False,str(e))

    with patch('core.analyzer.search_online',side_effect=RuntimeError('offline QA test')):
        fail_web=analyze_document(doc,refs,semantic=False,online=True)
    record('Online failure falls back to local',bool(fail_web['matches']) and any('Online search interrupted' in w for w in fail_web['warnings']))

    mock_text='A public webpage contains distinctive example text that can be retrieved and compared to a submitted student passage. '
    mock_source={'id':'web:https://example.org/qa','title':'Mock public source','url':'https://example.org/qa',
                 'type':'web','text':mock_text,'pages':[{'page':1,'start':0,'end':len(mock_text)}]}
    mock_doc=extract_paste(mock_text+' Additional original prose discusses pineapple-shaped galaxies and clocks.','Mock source comparison')
    with patch('core.analyzer.search_online',return_value=([mock_source],[])):
        matched_web=analyze_document(mock_doc,[],semantic=False,online=True)
    record('Web source integration (mock only)',matched_web['references_checked']==1 and any(m['source_type']=='web' and m['source_url']=='https://example.org/qa' for m in matched_web['matches']))
    results.append({'feature':'Live Tavily search / network retrieval','status':'NOT TESTED',
                    'details':'Requires user API key, network and public webpage access'})
    print('SKIP | Live Tavily search — use the app with a real API key')

    try:
        extract_upload('bad.exe',b'fake')
        bad=False
    except ExtractionError:
        bad=True
    record('Unsupported upload rejected',bad)
    try:
        extract_paste('')
        bad=False
    except ExtractionError:
        bad=True
    record('Empty input rejected',bad)

    results.append({'feature':'Streamlit UI navigation and buttons','status':'MANUAL',
                    'details':'Follow qa/TEST_GUIDE.md steps in browser'})
    results.append({'feature':'Live direct URL fetching','status':'NOT TESTED',
                    'details':'Requires accessible public HTML and robots permission'})
    results.append({'feature':'Scanned PDF OCR','status':'NOT SUPPORTED',
                    'details':'Not implemented in v1.2'})
    output=ROOT/'qa'/'output'
    output.mkdir(exist_ok=True)
    summary={'test_input':'ONE_FILE_FEATURE_TEST.txt',
             'overall':'PASS' if not failures else 'FAIL',
             'coverage_percent':result['coverage'],
             'flagged_passages':len(result['matches']),
             'references_checked':result['references_checked'],
             'semantic_used':result['semantic_used'],
             'ai_pattern_status':pattern.get('status'),
             'checks':results}
    (output/'feature_test_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    from collections import Counter
    counts=Counter(x['status'] for x in results)
    print(f'\n{summary["overall"]}: {counts["PASS"]} passed, {counts["FAIL"]} failed, '+
          f'{counts["NOT TESTED"]} not tested, {counts["MANUAL"]} manual, '+
          f'{counts["NOT SUPPORTED"]} unsupported. Results: {output}')
    return 1 if failures else 0

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-ai',action='store_true',help='Run and assert MiniLM inference (requires model installation)')
    args=parser.parse_args()
    raise SystemExit(run(with_ai=args.with_ai))
