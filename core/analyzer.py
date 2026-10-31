"""Single entry point: local matching continues when online retrieval fails."""
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from .detection import match_documents
from .extractor import segment
from .web_sources import search_online, direct_url_source
from .wikipedia import search_wikipedia, wikipedia_url_source
from .ai_patterns import infer
from .ai_comparator import inspect_candidates

def _source_identity(source: dict) -> str:
    """Deduplicate URLs discovered with ordinary tracking variants."""
    raw = source.get('url')
    if not raw:
        return source['id']
    p = urlsplit(raw)
    tracking = {'fbclid','gclid','mc_cid','mc_eid','ref','source'}
    query = [(k,v) for k,v in parse_qsl(p.query,keep_blank_values=True)
             if k.lower() not in tracking and not k.lower().startswith('utm_')]
    path = (p.path.rstrip('/') or '/')
    return urlunsplit((p.scheme.lower(),(p.netloc or '').lower(),path,urlencode(query),''))


def analyze_document(document: dict, references: list[dict], *,
                     semantic: bool=True, online: bool=False, direct_url: str='',
                     online_provider: str='tavily', wikipedia_contact: str | None=None,
                     ai_patterns: bool=False, lexical_threshold: float=.32,
                     semantic_threshold: float=.70,
                     exclude_identical_references: bool=False,
                     deep_search: bool=False) -> dict:
    warnings=[]
    identical_refs=[s for s in references if s.get('text') == document.get('text')]
    if identical_refs:
        titles=', '.join(s.get('title', 'Untitled') for s in identical_refs[:5])
        if exclude_identical_references:
            warnings.append(
                'Excluded text-identical local reference(s) for this run: ' + titles +
                '. Do not use this setting if comparing genuinely duplicated submissions from different authors.'
            )
        else:
            warnings.append(
                'Possible self-comparison: submitted text is IDENTICAL to local reference(s): ' + titles +
                '. High exact-match coverage may be expected. Confirm whether these are intentionally indexed previous submissions.'
            )
    all_sources=[s for s in references if not (exclude_identical_references and s.get('text') == document.get('text'))]
    provider_status = {}
    if online:
        if online_provider not in ('wikipedia', 'tavily', 'both'):
            raise ValueError('Unknown online search provider.')
        requested = ['wikipedia', 'tavily'] if online_provider == 'both' else [online_provider]
        passages = segment(document['text'], document['pages'])
        # Each provider fails independently; a problem with Tavily must never
        # prevent Wikipedia or local matching from completing (and vice versa).
        for name in requested:
            try:
                if name == 'wikipedia':
                    found, web_warnings = search_wikipedia(passages,
                        max_queries=10 if deep_search else 7,
                        max_pages=15 if deep_search else 10, contact=wikipedia_contact)
                else:
                    found, web_warnings = search_online(passages,
                        max_queries=9 if deep_search else 7,
                        max_pages=15 if deep_search else 10, deep=deep_search)
                provider_status[name] = {'requested': True, 'retrieved': len(found),
                                         'warnings': list(web_warnings),
                                         'state': 'retrieved' if found else 'no_verified_sources'}
                for item in found:
                    item.setdefault('provider', name)
                all_sources.extend(found)
                warnings.extend(f'{name.title()}: {warning}' for warning in web_warnings)
            except Exception as exc:
                provider_status[name] = {'requested': True, 'retrieved': 0,
                                         'warnings': [type(exc).__name__], 'state': 'failed'}
                warnings.append(f'Online search interrupted for {name.title()} ({type(exc).__name__}); other providers and local comparison continued.')
    if direct_url.strip():
        try:
            if direct_url.strip().startswith('https://en.wikipedia.org/wiki/'):
                all_sources.append(wikipedia_url_source(direct_url.strip(), contact=wikipedia_contact))
            else:
                all_sources.append(direct_url_source(direct_url.strip()))
        except Exception as exc:
            warnings.append(f'Direct URL could not be checked: {str(exc)[:200]}')
    # Coalesce identical public URLs across search providers and direct-URL mode.
    # Preserve multiple discovery providers as provenance, not duplicate evidence.
    dedup = {}
    for source in all_sources:
        ident = _source_identity(source)
        if ident in dedup:
            previous = dedup[ident]
            origins = set(previous.get('discovered_via', [previous.get('provider', previous.get('type', 'local'))]))
            origins.update(source.get('discovered_via', [source.get('provider', source.get('type', 'local'))]))
            previous['discovered_via'] = sorted(origins)
            if len(source.get('text','')) > len(previous.get('text','')):
                previous['text'] = source['text']
                previous['pages'] = source['pages']
        else:
            copy = dict(source)
            copy['discovered_via'] = list(source.get('discovered_via', [source.get('provider', source.get('type', 'local'))]))
            dedup[ident] = copy
    sources=list(dedup.values())
    result=match_documents(document, sources, semantic=semantic,
                           lexical_threshold=lexical_threshold,
                           semantic_threshold=semantic_threshold)
    result['warnings']=warnings+result['warnings']
    result['provider_status']=provider_status
    result['source_inventory']=[{'id':s['id'], 'discovery_query':s.get('discovery_query'), 'title':s.get('title',''), 'url':s.get('url'), 'type':s.get('type','local'), 'provider':s.get('provider', 'local'), 'discovered_via':s.get('discovered_via', [])} for s in sources]
    result['identical_local_reference_titles']=[s.get('title', 'Untitled') for s in identical_refs]
    result['excluded_identical_references']=bool(exclude_identical_references and identical_refs)
    result['source_modes']={'local':sum(s.get('type')!='web' for s in sources),
                            'web':sum(s.get('type')=='web' for s in sources)}
    # Distinguish successfully downloaded pages from actual matching evidence.
    hit_ids={m['source_id'] for m in result['matches']}
    for provider,info in provider_status.items():
        matching=[s for s in sources if s['id'] in hit_ids and
                  provider in s.get('discovered_via',[s.get('provider')])]
        info['matched_sources']=len(matching)
        info['exact_sources']=sum(any(m['source_id']==s['id'] and m.get('exact_word_count',0)>=7
                                       for m in result['matches']) for s in matching)
        if info.get('retrieved',0)>0 and not matching:
            result['warnings'].append(provider.title()+': Pages were retrieved, but none had a qualifying source match. This is a retrieval result, NOT proof of original writing.')
    result['online_requested']=online or bool(direct_url.strip())
    if result['online_requested'] and not result['source_modes']['web']:
        result['warnings'].append('No web pages were successfully compared; this is NOT a clean internet plagiarism result.')
    if ai_patterns:
        result['ai_pattern_analysis']=infer(document['text'])
        result['ai_comparator_analysis']=inspect_candidates(document['text'])
    else:
        result['ai_pattern_analysis']={'status':'disabled'}
    return result
