"""Opt-in public-web candidate discovery. Search hits are NOT verified matches."""
from __future__ import annotations
import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from .extractor import word_tokens

USER_AGENT = 'TraceLensCollegeProject/1.0 (source-text research; respects robots.txt)'

class WebSourceError(Exception):
    pass

def check_public_url(url: str) -> str:
    """Reject internal/reserved hosts for direct URL requests (single-user app hardening)."""
    p = urlparse(url)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
        raise WebSourceError('Enter a public http(s) webpage URL.')
    if p.port and p.port not in (80,443):
        raise WebSourceError('Only standard public HTTP(S) ports are supported.')
    hostname = p.hostname.casefold().rstrip('.')
    if hostname in ('localhost',) or hostname.endswith(('.local','.internal')):
        raise WebSourceError('Local and internal webpages cannot be fetched.')
    try:
        addresses = socket.getaddrinfo(hostname, p.port or (443 if p.scheme=='https' else 80))
    except socket.gaierror as e:
        raise WebSourceError('Could not resolve this webpage host.') from e
    for addr in addresses:
        ip = ipaddress.ip_address(addr[4][0])
        if not ip.is_global:
            raise WebSourceError('Private or reserved network addresses are not allowed.')
    return url

def fetch_page(url: str, max_bytes: int = 750_000) -> str:
    """Fetch only accessible public HTML. Redirects checked; network-boundary guarantees
    require a dedicated proxy for hostile multi-user deployments.
    """
    current = check_public_url(url)
    session = requests.Session()
    try:
        for _ in range(4):
            # Check site policies. Fail closed if robots cannot be retrieved.
            p = urlparse(current)
            robots_url = f'{p.scheme}://{p.netloc}/robots.txt'
            try:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                robots = session.get(robots_url, headers={'User-Agent': USER_AGENT},
                                     timeout=(4,8), allow_redirects=False, stream=True)
                try:
                    if robots.status_code == 404:
                        rp.parse([])
                    elif robots.status_code == 200:
                        raw = robots.content[:100_000].decode('utf-8',errors='replace')
                        rp.parse(raw.splitlines())
                    else:
                        raise WebSourceError("Cannot verify this website\'s robots rules; retrieval stopped.")
                finally:
                    robots.close()
                if not rp.can_fetch(USER_AGENT, current):
                    raise WebSourceError('This site does not permit this automated retrieval.')
            except WebSourceError:
                raise
            except requests.RequestException as exc:
                raise WebSourceError(f'Cannot verify site crawling rules: {type(exc).__name__}') from exc
            response = session.get(current, headers={'User-Agent':USER_AGENT}, timeout=(5,12),
                                   allow_redirects=False, stream=True)
            if response.status_code in (301,302,303,307,308):
                target = response.headers.get('Location')
                response.close()
                if not target:
                    raise WebSourceError('Webpage redirect has no target.')
                current = check_public_url(urljoin(current, target))
                continue
            response.raise_for_status()
            ctype = response.headers.get('Content-Type','').lower()
            if 'text/html' not in ctype and 'application/xhtml' not in ctype:
                response.close()
                raise WebSourceError('Only public HTML webpages are supported by direct URL checking.')
            data = bytearray()
            try:
                for part in response.iter_content(chunk_size=16_384):
                    data.extend(part)
                    if len(data) > max_bytes:
                        raise WebSourceError('Webpage exceeded the content-size limit.')
            finally:
                response.close()
            html = data.decode(response.encoding or 'utf-8', errors='replace')
            try:
                import trafilatura
                text = trafilatura.extract(html, include_comments=False, include_tables=True)
            except ImportError:
                text = None
            if not text:
                soup = BeautifulSoup(html,'html.parser')
                for tag in soup(['script','style','nav','footer','header']):
                    tag.decompose()
                text = soup.get_text(separator=' ',strip=True)
            if len(word_tokens(text or '')) < 15:
                raise WebSourceError('No meaningful text could be extracted from the webpage.')
            return text[:120_000]
        raise WebSourceError('Too many webpage redirects.')
    except requests.RequestException as exc:
        raise WebSourceError(f'Webpage request failed: {type(exc).__name__}') from exc
    finally:
        session.close()

# Short, diversified queries work better than quoting whole paragraphs.
# The search API is a discovery layer, NOT the text-comparison engine.
SEARCH_STOPWORDS = set("the and but or for with from into over under this that these those their them they what why when where how whose about which while before after were have been also your such there then than because does would could should being therefore moreover furthermore including consequently introduction conclusion".split())


def select_search_queries(passages: list[dict], limit: int = 7) -> list[str]:
    """Choose distinctive literal phrases and exploratory keyword searches.

    Sample separate document regions, not several near-identical introductory
    sentences. Excerpts are submitted externally only on explicit consent.
    """
    eligible=[p for p in passages if p.get('window',1)==1 and len(word_tokens(p.get('text','')))>=10]
    if not eligible: return []
    positions=[]
    for idx in (0,len(eligible)//2,len(eligible)-1):
        if idx not in positions: positions.append(idx)
    by_novelty=sorted(range(len(eligible)),key=lambda i:len(set(word_tokens(eligible[i]['text']))),reverse=True)
    positions += [i for i in by_novelty if i not in positions]
    queries=[]
    for idx in positions:
        tokens=word_tokens(eligible[idx]['text'])
        start=min(3,max(0,len(tokens)-9))
        phrase='"'+' '.join(tokens[start:start+9])+'"'
        if phrase not in queries:
            queries.append(phrase)
        if len(queries)>=max(1,limit-2): break
    for idx in positions:
        tok=word_tokens(eligible[idx]['text'])
        informative=[w for w in tok if len(w)>=4 and w not in SEARCH_STOPWORDS]
        phrase=' '.join(dict.fromkeys(informative[:6]))
        if phrase and phrase not in queries: queries.append(phrase)
        if len(queries)>=limit:break
    return queries[:limit]


def search_online(passages: list[dict], max_queries: int = 7, max_results: int = 5,
                  max_pages: int = 10, deep: bool = False) -> tuple[list[dict],list[str]]:
    """Tavily search + full extraction; no snippet-only results treated as verified."""
    warnings=[]
    key = os.getenv('TAVILY_API_KEY','').strip()
    if not key and os.getenv('TAVILY_KEYLESS','0') != '1':
        return [], ['Online search skipped: set TAVILY_API_KEY in .env or enable TAVILY_KEYLESS=1.']
    try:
        from tavily import TavilyClient
    except ImportError:
        return [], ['Online search unavailable: install tavily-python.']
    try:
        client = TavilyClient(api_key=key) if key else TavilyClient()
    except Exception as exc:
        return [], [f'Could not initialize online search: {type(exc).__name__}.']
    discovered={}
    queries = select_search_queries(passages, limit=max_queries)
    if not queries:
        return [], ['No searchable passages were found; use a longer document with distinctive sentences.']
    for query in queries:
        try:
            result = client.search(query=query, search_depth='advanced' if deep else 'basic', max_results=max_results)
            for item in result.get('results', []):
                url=item.get('url','')
                if url.startswith(('https://','http://')) and url not in discovered:
                    discovered[url] = {'url':url,'title':item.get('title') or url}
                    if len(discovered) >= max_pages*3:
                        break
        except Exception as exc:
            warnings.append(f'Tavily query failed ({type(exc).__name__}); verify credentials, quota and connection.')
        if len(discovered)>=max_pages*3:
            break
    if not discovered:
        warnings.append(f'Tavily returned no candidate webpages after {len(queries)} queries; try direct URL or a different passage.')
        return [],warnings
    # Rank search hits against the *actual submitted text* before spending
    # extraction credits. Search snippets are discovery metadata, never evidence.
    urls=list(discovered)[:max_pages]
    if len(discovered)>max_pages:
        warnings.append(f'Tavily discovery found {len(discovered)} candidate URLs; only {max_pages} were selected for extraction.')
    sources=[]
    try:
        extracted=client.extract(urls=urls, extract_depth='advanced' if deep else 'basic', format='text')
        for page in extracted.get('results',[]):
            url=page.get('url','')
            content=page.get('raw_content') or ''
            if url in discovered and len(word_tokens(content)) >= 15:
                sources.append({'id':'web:'+url,'title':discovered[url]['title'],
                                'type':'web','provider':'tavily','url':url,'text':content[:250_000],
                                'pages':[{'page':1,'start':0,'end':min(len(content),250_000)}]})
        for failure in extracted.get('failed_results',[]):
            url = failure.get('url', '') if isinstance(failure, dict) else ''
            warnings.append('Tavily could not extract a candidate webpage: ' + url[:100])
    except Exception as exc:
        warnings.append(f'Tavily full-text extraction failed ({type(exc).__name__}); discovered snippets were NOT scored.')
    if not sources:
        warnings.append('No complete webpage text was verified; snippets are not scored as source matches.')
    if any(len(x.get('raw_content') or '')>250_000 for x in extracted.get('results',[])):
        warnings.append('Some very long web pages were clipped at 250,000 characters; content after that limit was not checked.')
    if len(sources)<len(urls):
        warnings.append('Some discovered webpages were unavailable; web coverage is incomplete.')
    return sources,warnings

def direct_url_source(url: str) -> dict:
    url = check_public_url(url)
    text = fetch_page(url)
    return {'id':'web:'+url,'title':urlparse(url).hostname+' / supplied URL',
            'type':'web', 'provider':'direct_url', 'url':url, 'text':text,
            'pages':[{'page':1,'start':0,'end':len(text)}]}
