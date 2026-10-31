"""No-web / no-Streamlit smoke test using the included demo content."""
from pathlib import Path
from core.extractor import extract_upload
from core.analyzer import analyze_document
from core.report import make_pdf
ROOT=Path(__file__).resolve().parent
sources=[]
for name in ('library_ml.txt','library_energy.txt'):
    doc=extract_upload(name,(ROOT/'samples'/name).read_bytes())
    sources.append({**doc,'id':name,'type':'local'})
submitted=extract_upload('assignment_demo.txt',(ROOT/'samples'/'assignment_demo.txt').read_bytes())
result=analyze_document(submitted,sources,semantic=False,online=False)
print('Exact matched-word coverage:',result['coverage'])
print('Flagged source passages:',len(result['matches']))
for m in result['matches'][:4]:
    print('-',m['category'], 'from',m['source_title'], 'TF-IDF=',m['lexical_score'])
output=ROOT/'storage'/'demo_report.pdf'
output.parent.mkdir(exist_ok=True)
output.write_bytes(make_pdf(result))
print('Demo PDF written:',output)
