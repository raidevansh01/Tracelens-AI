import io
import pytest
import fitz
from pathlib import Path
from core.extractor import extract_upload, extract_paste, segment, ExtractionError
from core.detection import exact_alignment, citation_signals, match_documents
from core.analyzer import analyze_document
from core.report import make_pdf
from core.web_sources import check_public_url, WebSourceError
from db.store import Store

ROOT=Path(__file__).resolve().parents[1]

def sample(name):
    return extract_upload(name,(ROOT/'samples'/name).read_bytes())

def sources():
    out=[]
    for name in ('library_ml.txt','library_energy.txt'):
        out.append({**sample(name),'id':name,'type':'local'})
    return out

def test_text_processing_offsets():
    item=extract_paste('First sentence with six ordinary words. Another sentence with seven ordinary words here.')
    passages=segment(item['text'],item['pages'])
    assert len(passages)>=2
    assert all(item['text'][p['start']:p['end']]==p['text'] for p in passages)

def test_invalid_upload():
    with pytest.raises(ExtractionError):
        extract_upload('x.exe',b'abc')
    with pytest.raises(ExtractionError):
        extract_upload('bad.txt',b'\xff')
    with pytest.raises(ExtractionError):
        extract_paste('  ')

def test_pdf_extraction():
    doc=fitz.open()
    page=doc.new_page()
    page.insert_text((50,70),'A demonstration document contains meaningful text and copied phrases.')
    data=doc.tobytes()
    doc.close()
    extracted=extract_upload('test.pdf',data)
    assert 'demonstration document' in extracted['text']
    assert extracted['pages'][0]['page']==1

def test_exact_spans_are_real():
    a='The quick brown fox jumps over the lazy dog and goes home.'
    b='Some text. The quick brown fox jumps over the lazy dog and goes home!'
    spans_a,spans_b=exact_alignment(a,b)
    assert spans_a and spans_b
    assert a[spans_a[0][0]:spans_a[0][1]].startswith('The quick brown fox')

def test_reference_matching_and_coverage():
    result=match_documents(sample('assignment_demo.txt'),sources(),semantic=False)
    assert result['coverage']>0
    assert result['matched_words']>0
    assert not result['semantic_used']
    assert any(m['source_title']=='library_ml.txt' for m in result['matches'])
    assert all(m['category']!='Semantic candidate' for m in result['matches'])


def test_citation_not_a_verdict():
    assert citation_signals('Sharma (2022) argued this. See (Sharma, 2022).')
    assert citation_signals('As noted in [12], this is often observed.')

def test_web_unavailable_does_not_break_local(monkeypatch):
    import core.analyzer as core
    def fail(*args,**kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr(core,'search_online',fail)
    result=core.analyze_document(sample('assignment_demo.txt'),sources(),semantic=False,online=True)
    assert result['matches']
    assert any('Online search interrupted' in w for w in result['warnings'])

def test_pdf_report_is_valid():
    result=analyze_document(sample('assignment_demo.txt'),sources(),semantic=False)
    pdf=make_pdf(result)
    doc=fitz.open(stream=pdf,filetype='pdf')
    assert doc.page_count>=1
    assert 'TraceLens AI' in doc[0].get_text()
    assert 'word coverage' in doc[0].get_text() or 'matched-word' in doc[0].get_text()
    doc.close()

def test_sqlite_round_trip(tmp_path):
    store=Store(tmp_path/'trial.db')
    doc=sample('library_ml.txt')
    id_=store.add_document(doc)
    assert store.add_document(doc)==id_
    assert store.get_document(id_)['text']==doc['text']
    submitted=store.add_document(sample('assignment_demo.txt'),'submission')
    aid=store.save_analysis(submitted,{'coverage':25.0,'matches':[]})
    store.update_review(aid,'abc','Needs investigation','compare this')
    assert store.reviews_for(aid)['abc']['decision']=='Needs investigation'
    assert store.get_analysis(aid)['result']['coverage']==25.0
    assert store.remove_reference(id_)

def test_url_rejects_private_hosts():
    with pytest.raises(WebSourceError):
        check_public_url('http://127.0.0.1/private')
    with pytest.raises(WebSourceError):
        check_public_url('file:///etc/passwd')

def test_delete_analysis_removes_submission(tmp_path):
    store=Store(tmp_path/'cleanup.db')
    submitted=store.add_document(sample('assignment_demo.txt'),'submission')
    aid=store.save_analysis(submitted,{'coverage':0,'matches':[]})
    store.update_review(aid,'m','Not relevant')
    assert store.delete_analysis(aid)
    assert store.get_analysis(aid) is None
    assert store.get_document(submitted) is None

def test_common_definition_gets_lower_evidence_category():
    from core.detection import low_information
    assert low_information('Machine learning is a subset of artificial intelligence.')
    assert not low_information('A carefully designed study compared training examples with separate testing cases across many repeated experimental trials.')


def test_identical_reference_reproduction_and_warning():
    submitted = sample('assignment_demo.txt')
    references = sources() + [{**submitted, 'id':'accidentally_indexed_submission', 'type':'local'}]
    result = analyze_document(submitted, references, semantic=False)
    assert result['references_checked'] == 3
    assert result['coverage'] == 100.0  # global anchor scan finds every word in an identical reference
    assert any('Possible self-comparison' in w for w in result['warnings'])
    assert 'assignment_demo.txt' in result['identical_local_reference_titles']


def test_identical_reference_optional_exclusion_does_not_change_regular_sources():
    submitted = sample('assignment_demo.txt')
    references = sources() + [{**submitted, 'id':'accidentally_indexed_submission', 'type':'local'}]
    result = analyze_document(submitted, references, semantic=False,
                              exclude_identical_references=True)
    assert result['references_checked'] == 2
    assert result['coverage'] == 44.78
    assert result['excluded_identical_references']
    assert any('Excluded text-identical' in w for w in result['warnings'])
