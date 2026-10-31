"""Real, reproducible Wikipedia copy-and-paste acceptance test (network opt-in).

Checks (1) direct official-page extraction, (2) exact match against that page,
(3) optional ARTICLE DISCOVERY using Wikipedia search. Step (3) can fail while
(1)-(2) pass, clearly separating source-discovery from matching errors.

PowerShell:
 .\\.venv\\Scripts\\python.exe qa\\wiki_copy_acceptance.py --url https://en.wikipedia.org/wiki/Machine_learning --search
The user's WIKIMEDIA_CONTACT environment variable or .env must be configured.
"""
import argparse,sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.wikipedia import wikipedia_url_source,search_wikipedia
from core.extractor import extract_paste, segment,TOKEN_RE
from core.detection import match_documents


def run(url,contact=None,search=False,text_file=None):
    source=wikipedia_url_source(url,contact=contact)
    words=list(TOKEN_RE.finditer(source['text']))
    if len(words)<150:
        raise ValueError('Article too short for a representative copy test.')
    start=max(15,(len(words)*2)//3)
    end=min(len(words),start+85)
    copied=(Path(text_file).read_text(encoding='utf-8-sig') if text_file else
            source['text'][words[start].start():words[end-1].end()])
    doc=extract_paste(copied,title='Known Wikipedia excerpt')
    direct=match_documents(doc,[source],semantic=False)
    print(f'[DIRECT FETCH] {source["title"]} | {len(source["text"])} characters')
    print(f'[EXACT-COPY CHECK] coverage={direct["coverage"]}% | findings={len(direct["matches"])}')
    if direct['coverage']<95:
        raise AssertionError('Direct article comparison missed known literal copied text.')
    if search:
        found,warnings=search_wikipedia(segment(doc['text']),contact=contact)
        result=match_documents(doc,found,semantic=False)
        print(f'[AUTOMATIC WIKIPEDIA SEARCH] retrieved={len(found)} | coverage={result["coverage"]}% | findings={len(result["matches"])}')
        print('[RETRIEVED ARTICLES]',[x['title'] for x in found])
        print('[WARNINGS]',warnings)
        if result['coverage']<95:
            print('[DISCOVERY MISS] The article could not be rediscovered by search. Direct comparison succeeded. This is not a clean plagiarism result.')
            return False
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='https://en.wikipedia.org/wiki/Machine_learning')
    parser.add_argument('--contact')
    parser.add_argument('--file',help='Optional TXT file containing your pasted/copied article excerpt')
    parser.add_argument('--search',action='store_true')
    args=parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    try:
        ok=run(args.url,args.contact,args.search,args.file)
        raise SystemExit(0 if ok else 2)
    except Exception as exc:
        print('[FAILED]',type(exc).__name__,str(exc))
        raise SystemExit(1)
