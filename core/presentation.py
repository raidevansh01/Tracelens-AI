"""UI-safe interpretations of analysis output (pure functions; regression-testable)."""
from __future__ import annotations


def source_summary(result: dict) -> dict:
    """Keep unchecked documents distinct from genuinely zero matching evidence."""
    count = int(result.get('references_checked') or 0)
    matches = result.get('matches') or []
    coverage = result.get('coverage', 0)
    modes = result.get('source_modes') or {}
    if count <= 0:
        return {
            'state': 'not_assessed', 'coverage_label': 'N/A',
            'headline': 'Source similarity was not assessed',
            'detail': 'No local reference or successfully retrieved webpage was available. Add a reference, choose Wikipedia/Tavily, or provide a permitted public URL.',
            'references': 0, 'findings': 0,
            'local': int(modes.get('local', 0)), 'web': int(modes.get('web', 0)),
        }
    if not matches:
        headline = 'No qualifying matches in checked sources'
        detail = 'The checked sources produced no displayed matches at current thresholds. Other sources or paraphrases may still exist.'
        state = 'no_matches'
    else:
        headline = f'{len(matches)} source finding'+('' if len(matches) == 1 else 's')+' need inspection'
        detail = 'Check quotations, citations and overlapping text before drawing conclusions. Similarity is not proof of plagiarism.'
        state = 'matches'
    return {
        'state': state, 'coverage_label': f'{float(coverage):.2f}%',
        'headline': headline, 'detail': detail, 'references': count,
        'findings': len(matches), 'local': int(modes.get('local', 0)),
        'web': int(modes.get('web', 0)),
    }


def writing_summary(pattern: dict | None) -> dict:
    """Make classifier status explicit and never label low scores as human authored."""
    pattern = pattern or {'status': 'disabled'}
    status = pattern.get('status', 'disabled')
    if status == 'experimental':
        windows = pattern.get('windows') or []
        flags = sum(bool(w.get('review_flag')) for w in windows)
        peak = max((float(w.get('score', 0)) for w in windows), default=None)
        return {'state': status, 'count': len(windows), 'flags': flags,
                'peak': peak, 'threshold': pattern.get('threshold'),
                'headline': f'{flags} of {len(windows)} writing windows flagged for review',
                'detail': ('No windows crossed the learned threshold. This does not establish human authorship.'
                           if not flags else 'Flagged windows show patterns learned from the training data; they do not establish AI authorship.')}
    if status == 'unavailable':
        headline = 'Writing-pattern model unavailable'
        detail = pattern.get('message') or 'Train and validate a model to use this optional experiment.'
    elif status == 'insufficient_text':
        headline = 'Not enough text for writing analysis'
        detail = pattern.get('message') or 'Provide a longer passage.'
    else:
        headline = 'Writing-pattern analysis not selected'
        detail = 'Enable Experimental writing pattern analysis before running a document check.'
    return {'state': status, 'count': 0, 'flags': 0, 'peak': None,
            'threshold': None, 'headline': headline, 'detail': detail}


def model_execution_label(result: dict, *, requested: bool | None = None) -> tuple[str,str]:
    if result.get('semantic_used'):
        return 'MiniLM executed', 'The pretrained semantic model generated comparisons against retrieved source passages.'
    if not result.get('references_checked'):
        return 'No sources for semantic comparison', 'MiniLM cannot compare the submission without a reference source. This is separate from AI-writing analysis.'
    if requested is False:
        return 'MiniLM not requested', 'Semantic comparison was disabled for this analysis.'
    return 'MiniLM not executed', 'Only exact/lexical comparison was available; inspect warnings and AI installation.'


def writing_flagged_share(pattern: dict | None, text: str) -> float | None:
    """Share of *analyzed* word positions covered by flagged windows.

    It is NOT the probability or percentage of AI authorship. Overlapping
    windows count their words once, and unexamined text is excluded.
    """
    if not pattern or pattern.get('status') != 'experimental' or not pattern.get('windows'):
        return None
    from core.extractor import TOKEN_RE
    windows=pattern['windows']
    analyzed=[]
    flagged=[]
    for w in windows:
        start,end=int(w['start']),int(w['end'])
        if 0 <= start < end <= len(text):
            analyzed.append((start,end))
            if w.get('review_flag'):
                flagged.append((start,end))
    denominator=numerator=0
    for word in TOKEN_RE.finditer(text):
        if any(a<=word.start() and word.end()<=b for a,b in analyzed):
            denominator+=1
            if any(a<=word.start() and word.end()<=b for a,b in flagged):
                numerator+=1
    return round(100*numerator/denominator,1) if denominator else None
