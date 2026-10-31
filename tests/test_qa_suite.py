"""Controlled v1.2 single-file acceptance tests; all use one sample submission."""
from pathlib import Path
from core.extractor import extract_upload
from core.analyzer import analyze_document
from core.report import make_pdf

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    path=ROOT/'qa'/'ONE_FILE_FEATURE_TEST.txt'
    return extract_upload(path.name,path.read_bytes())

def refs():
    files=['samples/library_ml.txt','samples/library_energy.txt','qa/qa_reference_common.txt']
    out=[]
    for x in files:
        p=ROOT/x
        out.append({**extract_upload(p.name,p.read_bytes()),'id':p.name,'type':'local'})
    return out

def test_single_fixture_exact_expected_baseline():
    r=analyze_document(fixture(),refs(),semantic=False)
    assert r['word_count']==280
    assert r['references_checked']==3
    assert r['coverage']==26.07
    assert len(r['matches'])>=6  # v1.8 may add a verified global exact run
    assert all(m['semantic_score'] is None for m in r['matches'])

def test_single_fixture_expected_evidence_categories():
    r=analyze_document(fixture(),refs(),semantic=False)
    assert any(m['source_title']=='library_energy.txt' and m['category']=='Exact wording overlap' for m in r['matches'])
    assert any(m['source_title']=='library_ml.txt' and m['category']=='Exact wording overlap' for m in r['matches'])
    assert any(m['category']=='Low-information overlap' for m in r['matches'])
    assert any('Author–year citation pattern' in m['citation_signals'] for m in r['matches'])
    assert not any('cobalt umbrellas' in m['submitted_text'] for m in r['matches'])

def test_single_fixture_pdf_export_has_source_names():
    import fitz
    r=analyze_document(fixture(),refs(),semantic=False)
    f=fitz.open(stream=make_pdf(r),filetype='pdf')
    txt='\n'.join(p.get_text() for p in f)
    assert 'library_ml.txt' in txt
    assert 'library_energy.txt' in txt
    assert 'Low-information overlap' in txt
    f.close()


def test_docx_upload_extracts_paragraphs_and_tables():
    import io
    from docx import Document
    document=Document()
    document.add_paragraph('A sample assignment discusses neural network prediction and evaluation methods.')
    table=document.add_table(rows=1,cols=2)
    table.cell(0,0).text='Solar energy conversion'
    table.cell(0,1).text='Photovoltaic panels'
    output=io.BytesIO()
    document.save(output)
    extracted=extract_upload('example.docx',output.getvalue())
    assert 'neural network prediction' in extracted['text']
    assert 'Photovoltaic panels' in extracted['text']


def test_qa_pdf_extracts_text_and_preserves_page_index():
    import fitz
    pdf=fitz.open()
    p=pdf.new_page()
    p.insert_text((55,75),'Solar energy can be stored in batteries for later electricity usage.')
    extracted=extract_upload('test.pdf',pdf.tobytes())
    pdf.close()
    assert extracted['kind']=='pdf'
    assert 'stored in batteries' in extracted['text']
    assert extracted['pages'][0]['page']==1
