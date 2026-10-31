"""Pure presentation helpers for TraceLens v2 AI-writing comparison.

Do not reinterpret model scores as verified authorship probabilities.
"""
from __future__ import annotations


def writing_v2_snapshot(result: dict, text: str = '') -> dict:
    """Summarize saved v2 candidate outputs, preferring n-gram when available.

    Older analyses don't retroactively gain results from newly installed models.
    """
    comparator = result.get('ai_comparator_analysis') or {}
    models = comparator.get('models') or {}
    eligible = {name: value for name, value in models.items()
                if isinstance(value, dict) and value.get('status') == 'experimental'}
    if not eligible:
        return {'state': 'missing', 'models': models, 'primary': None,
                'message': 'No usable v2 candidate scores are saved in this analysis. Install a candidate, restart the app, and RUN A NEW analysis.'}

    preferred = 'ngram' if 'ngram' in eligible else 'minilm' if 'minilm' in eligible else next(iter(eligible))
    chosen = eligible[preferred]
    scores = chosen.get('scores') or []
    flags = chosen.get('flags') or []
    threshold = chosen.get('threshold')
    if not isinstance(threshold, (int, float)) or not scores or len(scores) != len(flags):
        return {'state': 'invalid', 'models': models, 'primary': preferred,
                'message': 'Saved candidate output is incomplete; rerun the analysis.'}
    try:
        from core.ai_patterns import make_windows
        offsets = make_windows(text) if text else []
    except Exception:
        offsets = []
    rows = []
    for idx, (score, flag) in enumerate(zip(scores, flags), start=1):
        window = offsets[idx-1] if idx-1 < len(offsets) else {}
        rows.append({'Window': idx, 'Model score': float(score), 'Threshold': float(threshold),
                     'Review signal': 'Review suggested' if flag else 'Not flagged',
                     'Characters': f"{window['start']}–{window['end']}" if window else 'Not available',
                     'text': window.get('text', '')})
    return {'state': 'ready', 'models': models, 'eligible': eligible, 'primary': preferred,
            'model': chosen, 'rows': rows, 'count': len(rows),
            'flags': sum(bool(f) for f in flags), 'threshold': float(threshold),
            'model_version': chosen.get('model_version', 'not recorded')}
