"""Wikipedia public search provider for TraceLens AI (no API key required).

Retrieves actual article extracts, never uses search snippets for scoring.
Requires a meaningful Wikimedia User-Agent with operator contact details.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from .extractor import word_tokens

WIKI_API = "https://en.wikipedia.org/w/api.php"
STOPWORDS = set("""the and with from this that there their those were been have has had than then which while where who what when
here into about before after between over under because such also very more less much many these they them some your our
not for are was will can could should would how why its his her one two three other some all only same different
study research paper source author example article students student document learning system based using used use
is it of to as on by at an a in or be we us do does""".split())

class WikipediaError(Exception):
    """Recoverable Wikipedia configuration or retrieval error."""


def _contact(value: str | None = None) -> str:
    contact = (value if value is not None else os.getenv('WIKIMEDIA_CONTACT', '')).strip()
    if (not contact or len(contact) > 180 or any(c in contact for c in '\r\n()')
            or contact.lower() in ('you@example.com', 'your_email@example.com')
            or not (('@' in contact and ' ' not in contact) or contact.startswith('https://'))):
        raise WikipediaError('Wikipedia requires a contact email or project HTTPS URL. Set WIKIMEDIA_CONTACT in .env or enter it in the app. This is NOT an API key.')
    return contact


def _headers(contact: str | None = None) -> dict:
    return {'User-Agent': f'TraceLensAI/1.6 ({_contact(contact)}) python-requests',
            'Accept': 'application/json'}


def _api_get(session, params: dict, headers: dict) -> dict:
    try:
        response = session.get(WIKI_API, params={**params,'format':'json','formatversion':2},
                               headers=headers, timeout=(4,14), allow_redirects=False)
        if response.status_code in (429, 503):
            raise WikipediaError('Wikipedia is rate-limiting requests. Please try again later.')
        if response.status_code in (301,302,303,307,308):
            raise WikipediaError('Unexpected Wikipedia API redirect; request stopped.')
        response.raise_for_status()
        payload = response.json()
        if 'error' in payload:
            raise WikipediaError('Wikipedia API returned an error: ' + str(payload['error'].get('code','unknown')))
        return payload
    except requests.RequestException as exc:
        raise WikipediaError('Wikipedia request failed: ' + type(exc).__name__) from exc
    except ValueError as exc:
        raise WikipediaError('Wikipedia returned invalid JSON.') from exc


def wikipedia_queries(passages: list[dict], limit: int = 7) -> list[str]:
    """Use independent quoted passages plus topic discovery, across the document.

    Topic-only queries commonly miss the *actual* Wikipedia source of pasted
    paragraphs. CirrusSearch supports quotes as phrase searches. Use nonadjacent
    excerpts from multiple positions before a small broad-topic fallback.
    """
    first_windows=[p for p in passages if p.get('window',1)==1 and len(word_tokens(p.get('text','')))>=10]
    if not first_windows:
        return []
    # Spread across the input (first, middle, last) rather than exhausting the
    # query budget on sentences describing the same part of a document.
    order=[]
    for k in (0,len(first_windows)//2,len(first_windows)-1):
        if k not in order:
            order.append(k)
    remaining=sorted(range(len(first_windows)),key=lambda i:len(set(word_tokens(first_windows[i]['text']))),reverse=True)
    for k in remaining:
        if k not in order: order.append(k)
    queries=[]
    # Discover copied text using a short exact phrase, which works with the
    # ordinary Wikipedia full-text search index, not expensive insource regex.
    for k in order:
        words=word_tokens(first_windows[k]['text'])
        for start in (min(4,max(0,len(words)-9)), max(0,(len(words)-9)//2)):
            phrase=words[start:start+9]
            if len(phrase)>=7:
                q='"'+' '.join(phrase)+'"'
                if q not in queries: queries.append(q)
            if len(queries)>=max(limit-1,1): break
        if len(queries)>=max(limit-1,1): break
    informative=[]
    for k in order:
        for w in word_tokens(first_windows[k]['text']):
            if w not in STOPWORDS and len(w)>3:
                informative.append(w)
    common=[w for w,_ in Counter(informative).most_common(4)]
    if common:
        q=' '.join(common[:3])
        if q not in queries: queries.append(q)
    return queries[:limit]


def _page_from_api(session, page_id: int, headers: dict) -> dict | None:
    """Retrieve parsed Wikipedia article HTML; use legacy extract as fallback.

    MediaWiki's parse API is designed for full rendered page content, while
    TextExtracts is a legacy summary-oriented module.
    """
    title=f'Wikipedia page {page_id}'
    text=''
    try:
        result = _api_get(session, {'action':'parse', 'pageid':str(page_id),
                                    'prop':'text', 'disableeditsection':1,
                                    'disabletoc':1}, headers)
        article = result.get('parse', {})
        title = article.get('title') or title
        raw_html = article.get('text') or ''
        if isinstance(raw_html, dict):
            raw_html=raw_html.get('*','')
        if raw_html:
            soup=BeautifulSoup(raw_html,'html.parser')
            for tag in soup.select('script, style, sup.reference, .mw-editsection, '
                                   '.reflist, .navbox, .metadata, .toc, table, '
                                   'noscript, .noprint, .reference, .mw-cite-backlink'):
                tag.decompose()
            content=soup.select_one('.mw-parser-output') or soup
            text=content.get_text(separator=' ', strip=True)
    except WikipediaError:
        # A MediaWiki site might disable parsing or temporarily reject it.
        pass
    if len(word_tokens(text))<15:
        result=_api_get(session, {'action':'query','prop':'extracts',
                                   'pageids':str(page_id),'explaintext':1,
                                   'exsectionformat':'plain','redirects':1}, headers)
        pages=result.get('query',{}).get('pages',[])
        if not pages:
            return None
        page=pages[0]
        text=(page.get('extract') or '').strip()
        title=page.get('title') or title
    if len(word_tokens(text))<15:
        return None
    url='https://en.wikipedia.org/wiki/'+quote(title.replace(' ','_'),safe='()')
    original_size=len(text)
    text=text[:250_000]
    return {'id':'web:'+url, 'title':title+' — Wikipedia', 'type':'web',
            'provider':'wikipedia', 'url':url, 'text':text, 'truncated':original_size>len(text),
            'pages':[{'page':1,'start':0,'end':len(text)}]}


def search_wikipedia(passages: list[dict], max_queries: int = 7, max_results: int = 5,
                     max_pages: int = 10, contact: str | None = None) -> tuple[list[dict],list[str]]:
    """Search public English Wikipedia and retrieve article text for every accepted source."""
    try:
        headers=_headers(contact)
    except WikipediaError as exc:
        return [],[str(exc)]
    warnings=[]; candidates={}
    queries=wikipedia_queries(passages,limit=max_queries)
    if not queries:
        return [],['No suitable searchable excerpts found for Wikipedia; try a longer distinctive paragraph.']
    with requests.Session() as session:
        for rank,q in enumerate(queries):
            try:
                result=_api_get(session,{'action':'query','list':'search','srsearch':q,
                                         'srnamespace':0,'srlimit':min(10,max_results)},headers)
                for position,item in enumerate(result.get('query',{}).get('search',[])):
                    page_id=item.get('pageid')
                    if isinstance(page_id,int) and page_id not in candidates:
                        # Exact phrase hits from distinct excerpts rank above
                        # broad topic hits. Preserve origin query for auditing.
                        candidates[page_id]={'title':item.get('title',''),
                                             'priority':(0 if q.startswith('"') else 1,rank,position),
                                             'query':q}
            except WikipediaError as exc:
                warnings.append('Wikipedia search query failed: '+str(exc))
        sources=[]
        ordered=sorted(candidates,key=lambda pid:candidates[pid]['priority'])
        for page_id in ordered[:max_pages]:
            try:
                page=_page_from_api(session,page_id,headers)
                if page:
                    if page.get('truncated'):
                        warnings.append('Wikipedia article was clipped at 250,000 characters: '+page['title'])
                    page['discovery_query']=candidates[page_id]['query']
                    sources.append(page)
                else:
                    warnings.append('Wikipedia article had no usable full-text extract: '+candidates[page_id]['title'])
            except WikipediaError as exc:
                warnings.append('Could not retrieve '+candidates[page_id]['title']+': '+str(exc))
        if len(ordered)>max_pages:
            warnings.append(f'Wikipedia checked {max_pages} of {len(ordered)} candidate articles. Other sources remain unchecked.')
    if not sources:
        warnings.append('No Wikipedia article text was retrieved; results are not evidence of a clean internet search.')
    return sources,warnings


def wikipedia_url_source(url: str, contact: str | None = None) -> dict:
    """Get a supplied English Wikipedia article through its API rather than HTML scraping."""
    p=urlparse(url)
    if p.scheme!='https' or p.hostname!='en.wikipedia.org' or p.port is not None or p.username or p.password or not p.path.startswith('/wiki/'):
        raise WikipediaError('Enter an English Wikipedia article URL such as https://en.wikipedia.org/wiki/Machine_learning')
    title=unquote(p.path[len('/wiki/'):]).replace('_',' ')
    if not title or len(title)>300 or any(c in title for c in '\r\n'):
        raise WikipediaError('Wikipedia article title is invalid.')
    headers=_headers(contact)
    with requests.Session() as session:
        lookup=_api_get(session,{'action':'query','prop':'info','titles':title,'redirects':1},headers)
        pages=lookup.get('query',{}).get('pages',[])
        if not pages or pages[0].get('missing') or not isinstance(pages[0].get('pageid'),int):
            raise WikipediaError('Wikipedia article was not found.')
        source=_page_from_api(session,pages[0]['pageid'],headers)
    if source is None:
        raise WikipediaError('The Wikipedia article has no usable text extract.')
    return source
