"""Global exact-copy detection, independent of TF-IDF ranking and page segmentation.

An indexed seven-token anchor provides high-recall literal copying evidence anywhere
in a source, including the middle of very long Wikipedia articles.  False positives
from repeated generic passages are subject to human review; this is NOT a verdict.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from .extractor import TOKEN_RE, word_tokens

ANCHOR_WORDS = 7


def _page_for(start: int, pages: list[dict] | None) -> int:
    for page in pages or []:
        if int(page['start']) <= start < int(page['end']):
            return int(page['page'])
    return 1


def _context(text: str, start: int, end: int, around: int = 180) -> tuple[str,int,int]:
    a = max(0, start-around)
    b = min(len(text), end+around)
    return text[a:b], start-a, end-a


def find_copied_runs(submitted: dict, references: list[dict], min_words: int = ANCHOR_WORDS,
                     max_hits_per_anchor: int = 24) -> list[dict]:
    """Locate and extend exact token runs; return absolute offsets in both texts.

    A phrase index over the submitted document is scanned against every source.
    Complexity is linear in source words except in highly repetitive text; a per-
    anchor fanout cap prevents adversarial documents from consuming all memory.
    """
    a = list(TOKEN_RE.finditer(submitted['text']))
    if len(a) < min_words:
        return []
    at = [m.group().casefold() for m in a]
    anchors = defaultdict(list)
    for i in range(len(at)-min_words+1):
        key=tuple(at[i:i+min_words])
        if len(anchors[key]) < max_hits_per_anchor:
            anchors[key].append(i)
    output=[]
    for source in references:
        text=source.get('text','')
        b=list(TOKEN_RE.finditer(text))
        bt=[m.group().casefold() for m in b]
        discovered=set()
        covered_diagonals={}  # Avoid quadratic re-extension in identical long documents.
        for j in range(len(bt)-min_words+1):
            possible=anchors.get(tuple(bt[j:j+min_words]),())
            for i in possible:
                diagonal=j-i
                if j < covered_diagonals.get(diagonal,-1):
                    continue
                l,r=i,j
                while l>0 and r>0 and at[l-1]==bt[r-1]:
                    l-=1; r-=1
                e,f=i+min_words,j+min_words
                while e<len(a) and f<len(b) and at[e]==bt[f]:
                    e+=1; f+=1
                if (l,e,r,f) in discovered:
                    continue
                discovered.add((l,e,r,f))
                covered_diagonals[diagonal]=max(covered_diagonals.get(diagonal,0),f)
                output.append({'source_id':source['id'],'source_title':source.get('title','Untitled'),
                               'source_url':source.get('url'),'source_type':source.get('type','local'),
                               'submitted_start':a[l].start(),'submitted_end':a[e-1].end(),
                               'source_start':b[r].start(),'source_end':b[f-1].end(),
                               'submitted_page':_page_for(a[l].start(),submitted.get('pages')),
                               'source_page':_page_for(b[r].start(),source.get('pages')),
                               'matched_word_count':e-l,
                               'source_text_full':text})
    # Longer runs take precedence over fragments of the same aligned sequence.
    output.sort(key=lambda x:x['matched_word_count'],reverse=True)
    keep=[]
    for m in output:
        if any(t['source_id']==m['source_id'] and
               t['submitted_start'] <= m['submitted_start'] and t['submitted_end']>=m['submitted_end']
               and t['source_start'] <= m['source_start'] and t['source_end']>=m['source_end']
               for t in keep):
            continue
        keep.append(m)
    return keep


def run_to_match(run: dict, submitted: dict) -> dict:
    """Convert a global copied run into an existing, offset-highlightable UI finding."""
    qa,qx,qy=_context(submitted['text'],run['submitted_start'],run['submitted_end'])
    sa,sx,sy=_context(run['source_text_full'],run['source_start'],run['source_end'])
    uid=f"{run['source_id']}:{run['submitted_start']}:{run['submitted_end']}:{run['source_start']}"
    return {'id':sha256(uid.encode()).hexdigest()[:16], 'source_id':run['source_id'],
            'source_title':run['source_title'],'source_url':run['source_url'],
            'source_type':run['source_type'],'submitted_text':qa,'source_text':sa,
            'submitted_start':run['submitted_start']-qx,'submitted_end':run['submitted_start']-qx+len(qa),
            'submitted_page':run['submitted_page'],'source_page':run['source_page'],
            'lexical_score':1.0,'semantic_score':None,'category':'Exact wording overlap',
            'input_spans':[[qx,qy]],'source_spans':[[sx,sy]], 'citation_signals':[],
            'explanation':f"Verified {run['matched_word_count']} consecutive identical words with exact source offsets. Check quotation and attribution in context.",
            'exact_word_count':run['matched_word_count']}
