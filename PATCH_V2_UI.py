"""One-time, idempotent Streamlit UI repair for TraceLens Stage-10.

Backs up the installed app, does not touch SQLite, .env, training data or model artifacts.
Run from the project root after extracting this kit: python PATCH_V2_UI.py
"""
from __future__ import annotations
import ast
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app.py'
CORE = ROOT / 'core' / 'ai_comparator.py'
TAG = '# V2_WRITING_UI_PATCH_2026_10'

V2_UI = '''
# V2_WRITING_UI_PATCH_2026_10
def show_writing_result(result, submission_text: str = '', analysis_id: str = ''):
    """Show the saved v2 classifier first; keep legacy analysis available."""
    snapshot = writing_v2_snapshot(result, submission_text)
    st.markdown('#### AI-writing detector · installed research candidates')
    if snapshot['state'] == 'ready':
        method = snapshot['primary']
        model = snapshot['model']
        count = snapshot['count']
        flagged = snapshot['flags']
        label = ('N-gram' if method == 'ngram' else 'MiniLM' if method == 'minilm' else method)
        version = snapshot['model_version']
        st.caption(f'Primary displayed classifier: **{label}** · saved model version: {version}.'
                   ' These are saved analysis scores; installing another model does not update older records.')
        cols = st.columns(3)
        cols[0].metric('Classifier', label)
        cols[1].metric('Windows suggested for review', f'{flagged} / {count}')
        cols[2].metric('Review threshold', f"{snapshot['threshold']:.4f}")
        if flagged:
            st.warning(f'{flagged} writing window(s) crossed the {label} review threshold. '
                       'This is an experimental signal, not proof that AI wrote the text.')
        else:
            st.info(f'No {label} windows crossed the review threshold. This does not prove human authorship.')
        rows = [{key: value for key, value in row.items() if key != 'text'}
                for row in snapshot['rows']]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={'Model score': st.column_config.NumberColumn(format='%.4f'),
                           'Threshold': st.column_config.NumberColumn(format='%.4f')})
        with st.expander('Inspect the v2 candidate passages', expanded=False):
            for row in snapshot['rows']:
                st.markdown(f"**Window {row['Window']} · {row['Review signal']} · score {row['Model score']:.4f}**")
                if row['text']:
                    st.text(row['text'])
                else:
                    st.caption('Passage offsets unavailable in saved result.')
        other = {name: value for name, value in snapshot['models'].items() if name != method}
        if other:
            with st.expander('Secondary experimental classifier', expanded=False):
                for name, candidate in other.items():
                    st.markdown(f'**{name.upper()}**')
                    if candidate.get('status') == 'experimental':
                        st.write({'Review threshold': candidate['threshold'],
                                  'Window scores': candidate['scores'],
                                  'Suggested reviews': sum(candidate['flags']),
                                  'Model version': candidate.get('model_version', 'not recorded')})
                    else:
                        st.warning(candidate.get('message', 'Model unavailable'))
        st.caption('V2 and legacy models were trained differently and may disagree. '
                   'Scores are not calibrated probabilities. Human review is required for consequential decisions.')
    else:
        st.info(snapshot['message'])
        st.caption('To use the v2 n-gram candidate: run promote_v2_candidate.py --method ngram --install, '
                   'restart Streamlit, upload the text again and click Run analysis.')
    # Use a checkbox, NOT an outer expander: the legacy view itself contains expanders.
    if st.checkbox('Show legacy seven-feature classifier and previous reviewer records',
                   value=(snapshot['state'] != 'ready'), key='tl_show_legacy_ai'):
        st.caption('This older model is separate from the v2 classifier above. '
                   'Its low scores and out-of-distribution warnings are not v2 n-gram results. '
                   'Existing reviewer decisions on this screen refer to legacy windows only.')
        show_legacy_writing_result(result, submission_text, analysis_id)

'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f'Cannot safely patch {label}: expected one matching anchor, found {count}. No file changed.')
    return src.replace(old, new, 1)


def main() -> int:
    if not APP.exists() or not CORE.exists():
        print('ERROR: Extract the patch files beside app.py inside TraceLens-AI.'); return 2
    source = APP.read_text(encoding='utf-8')
    if TAG in source:
        print('ALREADY PATCHED: Streamlit v2 UI block exists. No change needed.'); return 0
    updated = replace_once(source,
        'from core.presentation import model_execution_label, source_summary, writing_summary, writing_flagged_share',
        'from core.presentation import model_execution_label, source_summary, writing_summary, writing_flagged_share\nfrom core.v2_presentation import writing_v2_snapshot',
        'v2 presentation import')
    updated = replace_once(updated,
        'def show_writing_result(result, submission_text: str = \'\', analysis_id: str = \'\'):',
        'def show_legacy_writing_result(result, submission_text: str = \'\', analysis_id: str = \'\'):',
        'legacy function rename')
    updated = replace_once(updated, '\ndef page_analyze():', '\n' + V2_UI + '\ndef page_analyze():', 'v2 display insertion')
    updated = replace_once(updated,
        '            st.caption(\'The old seven-feature classifier remains experimental. Optional n-gram / MiniLM comparison models can be trained separately with train_ai_comparator.py. Results may disagree.\')',
        '            st.caption(\'Installed n-gram and MiniLM writing candidates will be displayed first in Results & Review on NEW analyses. The seven-feature classifier remains available separately for historical comparison.\')',
        'Analyze help text')
    # Keep Analyze preview consistent: new candidate gets priority over legacy.
    updated = replace_once(updated,
        "            ai_info = writing_summary(preview.get('ai_pattern_analysis'))\n            if ai_info['state'] == 'experimental':",
        "            v2_preview = writing_v2_snapshot(preview)\n            if v2_preview['state'] == 'ready':\n                st.info(f\"**{v2_preview['primary'].upper()} candidate**: {v2_preview['flags']} of {v2_preview['count']} windows suggested for review. Open full results for saved scores.\")\n            ai_info = writing_summary(preview.get('ai_pattern_analysis'))\n            if ai_info['state'] == 'experimental':",
        'Analyze preview')
    updated = replace_once(updated,
        "                st.info(ai_info['headline'] + '. ' + ai_info['detail'])",
        "                st.caption('Legacy classifier: ' + ai_info['headline'] + '. ' + ai_info['detail'])",
        'legacy preview label')
    ast.parse(updated, filename='app.py')
    core_src = CORE.read_text(encoding='utf-8')
    # Append a stable v2-vs-legacy version tag in future saved candidate outputs.
    target = "                     'evaluation':item['evaluation'],"
    amended = core_src
    if "'model_version':item.get('training_metadata'" not in core_src:
        amended = replace_once(core_src, target,
            "                     'evaluation':item['evaluation'],\n                     'model_version':item.get('training_metadata',{}).get('candidate_release','original-candidate'),",
            'comparator model metadata')
        ast.parse(amended, filename='core/ai_comparator.py')
    backup_dir = ROOT / 'backups' / ('before_v2_ui_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(APP, backup_dir / 'app.py')
    shutil.copy2(CORE, backup_dir / 'ai_comparator.py')
    try:
        APP.write_text(updated, encoding='utf-8')
        if amended != core_src:
            CORE.write_text(amended, encoding='utf-8')
    except Exception:
        shutil.copy2(backup_dir/'app.py', APP)
        shutil.copy2(backup_dir/'ai_comparator.py', CORE)
        raise
    print('PATCHED: Streamlit now shows the saved v2 n-gram results before the legacy classifier.')
    print('Backup saved to:', backup_dir)
    print('Restart Streamlit and RUN A NEW analysis. Existing saved results are not recalculated.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
