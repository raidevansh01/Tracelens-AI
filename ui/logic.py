"""Side-effect-free Streamlit view-models and action guards.

No external API calls or model execution is performed in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from core.presentation import source_summary
from core.v2_presentation import writing_v2_snapshot


@dataclass(frozen=True)
class Readiness:
    can_run: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: tuple[str, ...]


def check_readiness(*, has_document: bool, local_enabled: bool, reference_count: int,
                    online_provider: str, direct_url: str, writing_enabled: bool,
                    external_consent: bool, wikipedia_contact: str,
                    tavily_configured: bool) -> Readiness:
    """No submit unless the task has a meaningful, permitted analysis route."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    web = online_provider in ('wikipedia', 'tavily', 'both')
    url = direct_url.strip()
    wiki_url = url.startswith('https://en.wikipedia.org/wiki/')
    if not has_document:
        errors.append('Upload a document or paste text.')
    if local_enabled and reference_count:
        checks.append(f'{reference_count} local reference(s)')
    if web:
        checks.append('Public search: ' + ('Wikipedia + Tavily' if online_provider == 'both' else online_provider.title()))
    if url:
        checks.append('Direct URL comparison')
    if writing_enabled:
        checks.append('Experimental writing analysis')
    if not checks:
        errors.append('Choose a usable source or enable writing analysis. The local library is empty.' if local_enabled
                      else 'Enable a source comparison or experimental writing analysis.')
    if (web or url) and not external_consent:
        errors.append('Confirm authorization before sending text or fetching a public URL.')
    if (online_provider in ('wikipedia','both') or wiki_url) and not wikipedia_contact.strip():
        warnings.append('Wikipedia contact is not configured. Wikipedia retrieval may be unavailable.')
    if online_provider in ('tavily','both') and not tavily_configured:
        warnings.append('Tavily is not configured. Other enabled checks can still run.')
    if local_enabled and not reference_count:
        warnings.append('No local references are indexed. You can add the controlled demo collection with one click.')
    return Readiness(not errors, tuple(errors), tuple(warnings), tuple(checks))


def result_overview(result: dict, doc_text: str = '') -> dict:
    source = source_summary(result)
    writing = writing_v2_snapshot(result, doc_text)
    legacy = result.get('ai_pattern_analysis') or {}
    if writing['state'] == 'ready':
        ai_title = 'Review suggested' if writing['flags'] else 'No windows flagged'
        ai_detail = f"{writing['flags']} of {writing['count']} windows · {writing['primary'].upper()}"
    elif legacy.get('status') == 'experimental':
        ai_title = 'Legacy-only analysis'
        ai_detail = 'New v2 scores not stored in this run. Run a new analysis.'
    elif legacy.get('status') in ('unavailable','insufficient_text'):
        ai_title = 'Not available'
        ai_detail = legacy.get('message', 'Classifier did not run.')
    else:
        ai_title = 'Not run'
        ai_detail = 'Writing analysis was not requested.'
    return {'source': source, 'writing': writing, 'ai_title': ai_title,
            'ai_detail': ai_detail, 'source_title': source['headline']}


def safe_filename(name: str) -> str:
    """Avoid path fragments in downloadable report names."""
    text = Path(str(name or 'document')).name
    base = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in Path(text).stem)
    return (base.strip('_') or 'document')[:72] + '_tracelens_report.pdf'
