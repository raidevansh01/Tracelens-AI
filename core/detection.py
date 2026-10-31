"""Lexical and optional semantic source-grounded passage comparison."""
from __future__ import annotations
import importlib.util
import re
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import sha256
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .extractor import TOKEN_RE, segment, word_tokens
from .exact_retrieval import find_copied_runs, run_to_match

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

@lru_cache(maxsize=1)
def load_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DEFAULT_MODEL, device='cpu')

def exact_alignment(submitted: str, reference: str, min_run: int = 6) -> tuple[list[list[int]],list[list[int]]]:
    """Actual character spans of contiguous equal-word runs, never merely semantic spans."""
    a = list(TOKEN_RE.finditer(submitted))
    b = list(TOKEN_RE.finditer(reference))
    seq = SequenceMatcher(a=[m.group().casefold() for m in a],
                          b=[m.group().casefold() for m in b], autojunk=False)
    spans_a,spans_b=[],[]
    for match in seq.get_matching_blocks():
        if match.size >= min_run:
            spans_a.append([a[match.a].start(),a[match.a+match.size-1].end()])
            spans_b.append([b[match.b].start(),b[match.b+match.size-1].end()])
    return spans_a, spans_b

# Educational filter: these are low-information overlaps, not a universal list.
COMMON_GENERIC_DEFINITIONS={
    'machine learning is a subset of artificial intelligence',
    'artificial intelligence is the simulation of human intelligence by machines',
    'the internet is a global network of connected computers',
    'water is made of hydrogen and oxygen',
}

def low_information(text: str) -> bool:
    words=word_tokens(text)
    return len(words)<9 or ' '.join(words) in COMMON_GENERIC_DEFINITIONS

def citation_signals(text: str) -> list[str]:
    found=[]
    if re.search(r'(?m)[“\"]([^”\"]{14,})[”\"]',text):
        found.append('Quotation marks present')
    if re.search(r'\([\w .,&-]+,\s*(?:19|20)\d{2}[a-z]?\)',text):
        found.append('Author–year citation pattern')
    if re.search(r'\[(?:\d{1,3})(?:\s*,\s*\d{1,3})*\]',text):
        found.append('Numbered citation pattern')
    return found

def _pairs(queries: list[dict], refs: list[dict], semantic: bool,
           top_per_query: int = 8, max_pairs: int = 450) -> tuple[list[dict],list[str],bool]:
    notes=[]
    if not queries or not refs:
        return [], notes, False
    qtexts=[p['text'] for p in queries]
    rtexts=[p['text'] for p in refs]
    try:
        vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode',
                                   max_features=50000)
        vectorizer.fit(qtexts+rtexts)
        qa=vectorizer.transform(qtexts)
        rb=vectorizer.transform(rtexts)
        lex=cosine_similarity(qa,rb).astype(np.float32)
    except ValueError:
        lex=np.zeros((len(queries),len(refs)),dtype=np.float32)
    sem=None
    used_semantic=False
    if semantic:
        if importlib.util.find_spec('sentence_transformers') is None:
            notes.append('Semantic model not installed. Running exact and TF-IDF only; install requirements-ai.txt.')
        else:
            try:
                model=load_semantic_model()
                # Embeddings are normalized: dot product = cosine similarity.
                q_emb=model.encode(qtexts,batch_size=24,normalize_embeddings=True,
                                   show_progress_bar=False,convert_to_numpy=True)
                r_emb=model.encode(rtexts,batch_size=24,normalize_embeddings=True,
                                   show_progress_bar=False,convert_to_numpy=True)
                sem=np.asarray(q_emb @ r_emb.T,dtype=np.float32)
                used_semantic=True
            except Exception as exc:
                notes.append(f'Semantic model could not run ({type(exc).__name__}); lexical analysis continued.')
    pairs=[]
    for i,query in enumerate(queries):
        candidate_ids=set(np.argsort(lex[i])[-top_per_query:].tolist())
        if sem is not None:
            candidate_ids.update(np.argsort(sem[i])[-top_per_query:].tolist())
        for j in candidate_ids:
            pairs.append((i,j,float(lex[i,j]),None if sem is None else float(sem[i,j])))
    # prioritize strong evidence and deduplicate at the output layer
    pairs.sort(key=lambda p:max(p[2],p[3] if p[3] is not None else 0), reverse=True)
    return [{'qi':i,'ri':j,'lexical':l,'semantic':s} for i,j,l,s in pairs[:max_pairs]],notes,used_semantic

def _rank_reference_passages(passages: list[dict], submitted_passages: list[dict], limit: int) -> list[dict]:
    """Keep relevant passages anywhere in a long source, not just the opening.

    This cheap lexical retrieval stage reduces memory/CPU use before the
    slower TF-IDF and optional transformer comparisons. It is not a plagiarism
    decision and does not exclude semantic-only candidates intentionally when
    a full source is below the budget.
    """
    if len(passages) <= limit:
        return passages
    query_tokens = set()
    for passage in submitted_passages:
        query_tokens.update(word_tokens(passage['text']))
    # Retain first few passages for navigation/context, then choose the most
    # overlapping passages throughout the source rather than truncating by page.
    scores = []
    for i,p in enumerate(passages):
        words = set(word_tokens(p['text']))
        relevant = len(words & query_tokens)
        # Relative term overlap, with preference to meaningful multi-word
        # matches; this stage only retrieves candidates for later scoring.
        score = relevant / max(len(words),1) + relevant * 0.01
        scores.append((score, i))
    selected = set(range(min(6, len(passages))))
    for _, idx in sorted(scores, reverse=True)[:max(limit-len(selected),0)]:
        selected.add(idx)
    if len(selected)<limit:
        for _, idx in sorted(scores, reverse=True):
            selected.add(idx)
            if len(selected)==limit:
                break
    return [passages[i] for i in sorted(selected)][:limit]


def match_documents(submitted: dict, references: list[dict],
                    semantic: bool=True, lexical_threshold: float=.32,
                    semantic_threshold: float=.70, max_matches: int=65) -> dict:
    q_passages=segment(submitted['text'],submitted['pages'],max_passages=200)
    source_meta={}
    r_passages=[]
    truncated=False
    for source in references:
        # Scan more positions in long articles and retrieve only the highest
        # lexical candidates for the expensive downstream comparisons.
        all_p=segment(source['text'],source.get('pages'),max_passages=1600)
        p=_rank_reference_passages(all_p, q_passages, limit=140)
        if len(all_p)>len(p):
            truncated=True
        if len(r_passages)+len(p)>1100:
            truncated=True
            p=p[:max(0,1100-len(r_passages))]
        source_meta[source['id']]={'title':source['title'], 'type':source.get('type','local'),
                                  'url':source.get('url')}
        for passage in p:
            r_passages.append({**passage,'source_id':source['id'],
                               'source_title':source['title'], 'source_url':source.get('url')})
        if len(r_passages)>=1100:
            break
    # Literal copying is recovered from the complete source text FIRST; it
    # must not depend on discovery ranking or 140-passage TF-IDF budgets.
    exact_runs = find_copied_runs(submitted, references)
    global_spans = []
    exact_items = []
    for run in exact_runs:
        phrase=submitted['text'][run['submitted_start']:run['submitted_end']]
        if low_information(phrase):
            continue
        global_spans.append((run['submitted_start'],run['submitted_end']))
        item=run_to_match(run,submitted)
        item['citation_signals']=citation_signals(phrase)
        if item['citation_signals']:
            item['explanation']+=' Attribution signals present; review context.'
        exact_items.append(item)
    candidates, warnings, used_semantic=_pairs(q_passages,r_passages,semantic)
    if len(exact_items)>max_matches:
        warnings.append('Exact match display limit reached: all verified runs count toward matched-word coverage, but not all can be listed.')
    if len(q_passages)>=200 or truncated:
        warnings.append('Candidate budget reached: relevant passages were sampled from long documents. Some matches may still be missed.')
    matches=exact_items[:max_matches]
    seen=[]
    # Existing TF-IDF matches remain useful for paraphrase/lexical candidates.
    # Do not duplicate a passage already explained by longer exact-copy evidence.
    exact_ranges = [(m['submitted_start'],m['submitted_end'],m['source_id']) for m in exact_runs]
    for pair in candidates:
        p=q_passages[pair['qi']]; s=r_passages[pair['ri']]
        inp_exact,ref_exact=exact_alignment(p['text'],s['text'])
        lex=pair['lexical']; sem=pair['semantic']
        if inp_exact:
            category='Exact wording overlap'
        elif lex>=.68:
            category='Near-exact wording'
        elif sem is not None and sem>=semantic_threshold:
            category='Semantic candidate'
        elif lex>=lexical_threshold:
            category='Lexical candidate'
        else:
            continue
        if len(word_tokens(p['text']))<8 and not inp_exact:
            continue
        if any(s['source_id']==sid and a <= p['start'] and b >= p['end']
               for a,b,sid in exact_ranges):
            continue
        if low_information(p['text']) and category in ('Exact wording overlap','Near-exact wording'):
            category='Low-information overlap'
        # avoid duplicate overlapping windows from the same source
        duplicate=False
        for m in seen:
            if m['source_id']!=s['source_id']:
                continue
            overlap=max(0,min(p['end'],m['end'])-max(p['start'],m['start']))
            shorter=min(p['end']-p['start'],m['end']-m['start'])
            if shorter and overlap/shorter>.73:
                duplicate=True
                break
        if duplicate:
            continue
        signals=citation_signals(p['text'])
        reasons=[]
        if inp_exact:
            reasons.append('Identical consecutive word sequence found in this source.')
        elif category=='Near-exact wording':
            reasons.append('Strong lexical overlap; examine word changes and context.')
        elif category=='Semantic candidate':
            reasons.append('Similar embedding representations, but no verified exact wording overlap.')
        else:
            reasons.append('Shared vocabulary and phrases; may reflect a common topic.')
        if signals:
            reasons.append('Citation/quotation signals present; attribution must be reviewed manually.')
        if category=='Low-information overlap':
            reasons.append('A short/common passage can match independently; lower evidentiary value.')
        elif len(word_tokens(p['text']))<12:
            reasons.append('Short passage: evidence is less informative.')
        key=f'{s["source_id"]}|{p["start"]}|{p["end"]}|{s["start"]}'
        item={'id':sha256(key.encode()).hexdigest()[:16],
              'source_id':s['source_id'],'source_title':s['source_title'],
              'source_url':s.get('source_url'),'source_type':source_meta[s['source_id']]['type'],
              'submitted_text':p['text'],'source_text':s['text'],
              'submitted_start':p['start'],'submitted_end':p['end'],
              'submitted_page':p['page'],'source_page':s['page'],
              'lexical_score':round(lex,4),'semantic_score':None if sem is None else round(sem,4),
              'category':category,'input_spans':inp_exact,'source_spans':ref_exact,
              'citation_signals':signals,'explanation':' '.join(reasons)}
        matches.append(item)
        seen.append({'source_id':s['source_id'],'start':p['start'],'end':p['end']})
        if len(matches)>=max_matches:
            warnings.append('Match display limit reached; additional candidates were not shown.')
            break
    # Count unique WORD positions touched by verified exact contiguous runs only.
    spans=list(global_spans)
    for m in matches:
        if m['category']=='Low-information overlap':
            continue   # separately visible; not included in the headline coverage metric
        # Exact global runs already use absolute offsets, so do not shift
        # them again through their context excerpts.
        if m.get('exact_word_count'):
            continue
        for a,b in m['input_spans']:
            spans.append((m['submitted_start']+a,m['submitted_start']+b))
    nwords=0; matched=0
    for word in TOKEN_RE.finditer(submitted['text']):
        nwords+=1
        if any(a<=word.start() and word.end()<=b for a,b in spans):
            matched+=1
    coverage=round(100*matched/nwords,2) if nwords else 0.
    return {'title':submitted['title'],'word_count':nwords,
            'references_checked':len(source_meta),'passages_checked':len(q_passages),
            'coverage':coverage,'matched_words':matched, 'matches':matches,
            'semantic_used':used_semantic,'warnings':warnings,
            'method':'Exact token alignment + TF-IDF cosine + optional MiniLM cosine',
            'exact_runs_found':len(exact_items),
            'interpretation':'Exact matched-word coverage is not a plagiarism percentage. Semantic candidates require human review.'}
