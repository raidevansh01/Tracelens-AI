"""TraceLens AI — accessible, offline-first, evidence-first Streamlit workspace."""
from __future__ import annotations

import html
import importlib.util
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.ai_patterns import MODEL_FILE
from core.analyzer import analyze_document
from core.extractor import ExtractionError, extract_paste, extract_upload
from core.presentation import model_execution_label, source_summary, writing_summary, writing_flagged_share
from core.report import make_pdf
from db.store import Store

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='TraceLens AI | Review workspace', page_icon='🔎',
                   layout='wide', initial_sidebar_state='expanded')

# Native controls are used for every action; CSS adjusts appearance only.
st.markdown('''<style>
:root {--ink:#14283a;--soft:#52697a;--border:#dce5ee;--accent:#126c83}
.stApp{background:#f6f8fb;color:var(--ink)}
[data-testid="stSidebar"]{background:#eef3f7;border-right:1px solid var(--border)}
.block-container{padding-top:1.5rem;padding-bottom:4rem;max-width:1290px}
h1,h2,h3{letter-spacing:-.025em}
h1{color:#183449;font-size:2.1rem!important}
h2{color:#183449;font-size:1.5rem!important}
h3{color:#183449;font-size:1.12rem!important}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--border);border-radius:12px;padding:13px 18px;min-height:124px}
[data-testid="stMetricLabel"]{color:#52697a}
[data-testid="stMetricValue"]{color:#183449}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--border)!important}
.tl-hero{border-radius:16px;background:linear-gradient(115deg,#15394c,#245d6c);color:#fff;padding:26px 29px;margin:2px 0 18px}
.tl-hero small{color:#c1e4ea;letter-spacing:.12em;font-weight:750;font-size:.72rem}
.tl-hero h2{color:#fff!important;line-height:1.16;margin:9px 0 7px;font-size:1.65rem!important}
.tl-hero p{color:#e8f2f5;line-height:1.6;margin:0;max-width:860px}
.tl-step{display:inline-block;background:#e6f3f5;color:#146175;padding:4px 10px;border-radius:25px;font-weight:700;font-size:.75rem}
.tl-tag{display:inline-block;background:#eef4f7;color:#3a6072;border-radius:6px;padding:2px 7px;font-size:.8rem;margin-right:5px}
.tl-mark{background:#ffe5a1;color:#272b30;padding:1px 2px;border-radius:2px}
.tl-compare{border:1px solid var(--border);background:#fff;border-radius:11px;padding:17px;line-height:1.75;min-height:110px;overflow-wrap:anywhere;white-space:normal}
.tl-caption{color:#597386;font-size:.85rem}
.tl-overview{border:1px solid #d6e2eb;background:#fff;border-radius:12px;padding:14px 18px;margin:7px 0 18px}
.tl-flag{border-left:4px solid #b77c2a;background:#fff8e8;padding:12px 16px;border-radius:8px}
.tl-neutral{border-left:4px solid #247a8e;background:#eef7f9;padding:12px 16px;border-radius:8px}
@media(max-width:700px){.block-container{padding:1rem}.tl-hero{padding:18px}.tl-hero h2{font-size:1.35rem!important}h1{font-size:1.7rem!important}}
</style>''', unsafe_allow_html=True)

# Requests to change pages are applied BEFORE the navigation widget is created.
requested_page = st.session_state.pop('tl_next_page', None)
if requested_page:
    st.session_state['tl_nav'] = requested_page

store = Store()
NAV = ['Analyze', 'Results & Review', 'Reference Library', 'Model Comparison', 'How to Use']


def go_to(page: str):
    st.session_state['tl_next_page'] = page
    st.rerun()


def heading(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)


def highlight(text, spans):
    """Only actual verified character offsets are highlighted; source text is escaped."""
    normalized = sorted((max(0, int(a)), min(len(text), int(b))) for a, b in spans if b > a)
    merged = []
    for a, b in normalized:
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    parts = []
    last = 0
    for a, b in merged:
        parts.extend([html.escape(text[last:a]), '<mark class="tl-mark">',
                      html.escape(text[a:b]), '</mark>'])
        last = b
    parts.append(html.escape(text[last:]))
    return '<div class="tl-compare">' + ''.join(parts).replace('\n', '<br>') + '</div>'


def local_sources():
    result = []
    for item in store.list_documents('reference'):
        doc = store.get_document(item['id'])
        if doc:
            result.append({'id': doc['id'], 'title': doc['title'], 'text': doc['text'],
                           'pages': doc['pages'], 'type': 'local', 'url': None})
    return result


def health_panel():
    with st.sidebar.expander('System readiness', expanded=False):
        has_local = bool(store.list_documents('reference'))
        has_contact = bool(os.getenv('WIKIMEDIA_CONTACT', '').strip())
        has_tavily = bool(os.getenv('TAVILY_API_KEY', '').strip()) or os.getenv('TAVILY_KEYLESS', '0') == '1'
        has_transformer = importlib.util.find_spec('sentence_transformers') is not None
        st.write(('✅' if has_local else '○') + ' Local references: ' + ('ready' if has_local else 'none added'))
        st.write(('✅' if has_contact else '○') + ' Wikipedia: ' + ('contact configured' if has_contact else 'contact needed'))
        st.write(('✅' if has_tavily else '○') + ' Tavily: ' + ('configured' if has_tavily else 'optional'))
        st.write(('✅' if has_transformer else '○') + ' MiniLM: ' + ('package installed' if has_transformer else 'install requirements-ai.txt'))
        st.write(('✅' if MODEL_FILE.exists() else '○') + ' Writing classifier: ' + ('model file present' if MODEL_FILE.exists() else 'not trained'))
        st.caption('Readiness is a local configuration check, not proof that a network request or model inference succeeds. Keys are never displayed.')


def show_source_metrics(result, *, compact=False):
    summary = source_summary(result)
    if summary['state'] == 'not_assessed':
        st.info('**Source check not assessed.** No reference was compared. The percentage below cannot be interpreted as an originality result.')
    cols = st.columns(3)
    cols[0].metric('Exact matched-word coverage', summary['coverage_label'],
                   help='Percentage of eligible words in verified exact alignments. Not a plagiarism verdict. N/A means no references were compared.')
    cols[1].metric('Source findings', summary['findings'], help='Possible matches requiring contextual review.')
    cols[2].metric('Sources compared', summary['references'], help='Number of local documents and actually retrieved full web sources.')
    if not compact:
        st.caption(summary['detail'])
    return summary


def show_writing_result(result, submission_text: str = '', analysis_id: str = ''):
    pattern = result.get('ai_pattern_analysis') or {'status': 'disabled'}
    summary = writing_summary(pattern)
    st.subheader('AI-writing pattern experiment')
    if summary['state'] == 'disabled':
        st.info('Not run. Select **Experimental writing pattern analysis** on the Analyze page to evaluate writing style.')
        return
    if summary['state'] != 'experimental':
        st.warning(summary['headline'] + ': ' + summary['detail'])
        return
    flagged_share = writing_flagged_share(pattern, submission_text) if submission_text else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Text windows analyzed', summary['count'])
    c2.metric('Windows flagged', summary['flags'])
    c3.metric('Flagged analyzed-text share', 'N/A' if flagged_share is None else f'{flagged_share:.1f}%',
              help='Unique analyzed words covered by review-flagged windows, not an AI-authorship percentage.')
    c4.metric('Review threshold', f'{float(summary["threshold"]):.3f}')
    if any(w.get('distribution_warning') for w in pattern.get('windows', [])):
        st.warning('Some windows are **inconclusive** because their features differ substantially from the training corpus. A low model score is not evidence of human authorship.')
    if summary['flags']:
        st.warning('**Some writing windows crossed the experimental threshold.** This is not evidence establishing AI authorship.')
    else:
        st.info('**No reliable review flags were produced.** This may mean windows are below the threshold or that predictions are inconclusive because of distribution shift. It does not prove human authorship.')
    st.caption('Flagged text share counts analyzed words covered by flagged windows only. Scores are NOT calibrated probabilities or evidence proving AI use. Low scores do not prove human authorship.')
    if any(w.get('distribution_warning') for w in pattern.get('windows', [])):
        st.warning('Some passages differ substantially from the training feature ranges. Their predictions may be particularly unreliable; inspect the individual windows.')
    if not pattern.get('training_profile_available'):
        st.caption('This model predates training-distribution diagnostics. Retrain with the new script to enable shift warnings.')
    windows = pattern.get('windows') or []
    rows = [{'Window': i, 'Model score': float(w['score']),
             'Threshold': float(pattern.get('threshold', 0)),
             'Outcome': w.get('outcome', 'Review suggested' if w.get('review_flag') else 'Not flagged'),
             'Distribution shift': 'Possible' if w.get('distribution_warning') else '—',
             'Characters': f'{w["start"]}–{w["end"]}'}
            for i, w in enumerate(windows, 1)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={'Model score': st.column_config.NumberColumn(format='%.4f'),
                                'Threshold': st.column_config.NumberColumn(format='%.4f')})
    if submission_text:
        st.markdown('#### Review writing passages')
        st.caption('Review any window, including ones the model missed. A saved reviewer decision never changes the stored model score. Overlapping windows may repeat text.')
        decisions = ['Unreviewed', 'Discuss with author', 'AI assistance disclosed',
                     'No concern', 'Insufficient evidence']
        existing = store.writing_reviews_for(analysis_id) if analysis_id else {}
        review_count = sum(r.get('decision') != 'Unreviewed' for r in existing.values())
        st.caption(f'{review_count} of {len(windows)} windows have a human review decision.')
        for index, window in enumerate(windows):
            flagged = bool(window.get('review_flag'))
            old = existing.get(index, {})
            current = old.get('decision', 'Unreviewed')
            status_label = 'Model flagged for review' if flagged else 'Below model threshold (not a human-authorship verdict)'
            reviewed_label = (' · Human: ' + current) if current != 'Unreviewed' else ''
            with st.expander(f'Window {index+1} · score {float(window["score"]):.4f} · {status_label}{reviewed_label}',
                             expanded=bool(flagged and index < 2)):
                safe_text = html.escape(submission_text[int(window['start']):int(window['end'])])
                cls = 'tl-flag' if flagged else 'tl-neutral'
                st.markdown(f'<div class="{cls}" style="white-space:pre-wrap;overflow-wrap:anywhere">{safe_text}</div>',
                            unsafe_allow_html=True)
                st.caption(f'Characters {window["start"]}–{window["end"]} · {window.get("outcome","Experimental review signal")} · Not probability of AI authorship.')
                if window.get('distribution_warning'):
                    st.warning('Out-of-training-range features: ' + ', '.join(window.get('unusual_features', [])) + '. Treat this prediction as especially uncertain.')
                with st.form(f'writing_review_{analysis_id}_{index}'):
                    decision = st.selectbox('Human review outcome', decisions,
                        index=decisions.index(current) if current in decisions else 0)
                    note = st.text_area('Reviewer note / supporting evidence', value=old.get('note', ''),
                                        max_chars=2000, height=80,
                                        placeholder='Record contextual information, student disclosure, or why the signal is inconclusive.')
                    if st.form_submit_button('Save writing review'):
                        try:
                            store.update_writing_review(analysis_id, index, decision, note)
                            st.success('Writing review saved separately from the model prediction.')
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
    else:
        st.caption('Source text unavailable for window inspection.')
    comparator=result.get('ai_comparator_analysis') or {}
    if comparator.get('models'):
        st.markdown('#### Independent experimental model comparison')
        st.caption('Candidates are displayed separately; agreement is NOT proof of authorship. These models require their own independent evaluation.')
        for name, candidate in comparator['models'].items():
            with st.expander(f'{name.upper()} writing-pattern candidate'):
                if candidate.get('status')!='experimental':
                    st.warning(candidate.get('message','Candidate unavailable'))
                else:
                    st.write({'Review threshold':candidate['threshold'],
                              'Window scores':candidate['scores'],
                              'Suggested reviews':sum(candidate['flags'])})
                    st.caption(candidate['note'])
                    st.json(candidate.get('evaluation') or {})
    with st.expander('Validation metrics and limitations'):
        evaluation = pattern.get('evaluation') or {}
        if evaluation:
            st.json(evaluation)
        st.caption('Evaluation on a single held-out corpus does not establish reliability across new AI models, writing styles, or student populations. Never use this signal alone for accusations.')


def page_analyze():
    st.markdown('''<div class="tl-hero"><small>TRACELENS AI · DOCUMENT WORKSPACE</small>
    <h2>Find matching sources. Inspect the evidence. Understand the AI signals.</h2>
    <p>Two separate analyses: <b>source similarity</b> compares against documents and webpages; <b>writing patterns</b> is an experimental classifier. Neither is an automatic misconduct verdict.</p></div>''', unsafe_allow_html=True)
    st.markdown('**Simple workflow**  &nbsp; <span class="tl-step">1 · Add or select sources</span> &nbsp; <span class="tl-step">2 · Analyze a document</span> &nbsp; <span class="tl-step">3 · Review findings</span>', unsafe_allow_html=True)
    left, right = st.columns([1.12, 1], gap='large')
    with left:
        st.subheader('1. Choose your document')
        mode = st.radio('Input type', ['Upload a file', 'Paste text'], horizontal=True)
        uploaded = None
        pasted = ''
        if mode == 'Upload a file':
            uploaded = st.file_uploader('Upload PDF, DOCX or TXT', type=['pdf', 'docx', 'txt'],
                                        help='Up to 10 MB; PDF up to 30 pages. Scanned PDFs are not OCR-processed.')
        else:
            pasted = st.text_area('Paste the document text', height=230,
                                  placeholder='Paste an assignment or document excerpt...')
        st.caption('Original uploaded text is stored locally. External searching is OFF unless you opt in.')
        st.subheader('2. Choose what to check')
        do_local = st.checkbox('Compare against local reference documents', value=True)
        num_refs = len(store.list_documents('reference'))
        if do_local and not num_refs:
            st.warning('The local library is empty. Add documents in **Reference Library**, or enable web search. You can still run writing-pattern analysis without references.')
        elif do_local:
            st.caption(f'{num_refs} reference document(s) indexed locally.')
        do_patterns = st.checkbox('Analyze AI-writing patterns (experimental)', value=False,
                                  help='Requires your separately trained classifier. Does not prove AI authorship.')
        if do_patterns:
            if not MODEL_FILE.exists():
                st.warning('No trained writing-pattern model found. Train it before expecting writing scores.')
            st.caption('The old seven-feature classifier remains experimental. Optional n-gram / MiniLM comparison models can be trained separately with train_ai_comparator.py. Results may disagree.')
    with right:
        st.subheader('3. Source search options')
        st.caption('If your text comes from a known Wikipedia article, paste its URL below. Search engines are not guaranteed to rediscover every source.')
        choice = st.selectbox('Online search provider',
                              ['Off — local only', 'Wikipedia only · free', 'Tavily only · broader web', 'Wikipedia + Tavily · combined'],
                              help='Choose one or BOTH independent providers. Combined mode continues if either provider fails; duplicate URLs are only compared once.')
        online = choice != 'Off — local only'
        provider = ('both' if '+' in choice else 'wikipedia' if choice.startswith('Wikipedia') else 'tavily')
        deep_search=st.checkbox('Thorough online discovery (more sources and more API credits)', value=False,
                              help='Search more Wikipedia pages; Tavily uses advanced search/extraction when enabled. Slower and may cost more.') if online else False
        do_semantic = st.checkbox('Use MiniLM semantic matching (when sources are available)', value=True,
                                  help='Pretrained CPU-compatible model. This finds related passages, not AI authorship.')
        if do_semantic and importlib.util.find_spec('sentence_transformers') is None:
            st.caption('MiniLM not installed yet. The app will continue with exact and TF-IDF matching.')
        url = st.text_input('Optional direct public webpage URL',
                            placeholder='https://en.wikipedia.org/wiki/Machine_learning',
                            help='Compare directly against a permitted HTML article or English Wikipedia article.')
        wiki_contact = None
        if provider in ('wikipedia', 'both') and online or url.strip().startswith('https://en.wikipedia.org/wiki/'):
            wiki_contact = st.text_input('Wikimedia contact email or HTTPS project URL',
                                         value=os.getenv('WIKIMEDIA_CONTACT', ''),
                                         help='This is an identifying contact, not an API key.')
            if not wiki_contact.strip():
                st.caption('Enter a contact here or set WIKIMEDIA_CONTACT in .env before retrieving Wikipedia.')
        with st.expander('Preview web-search queries (no API calls)'):
            st.caption('These examples are computed locally and are NOT sent unless you consent and run analysis. Avoid web search for confidential student work.')
            if st.button('Preview selected phrases', key='preview_outgoing_queries'):
                try:
                    if mode == 'Upload a file' and uploaded is not None:
                        preview_doc = extract_upload(uploaded.name, uploaded.getvalue())
                    elif mode == 'Paste text' and pasted.strip():
                        preview_doc = extract_paste(pasted)
                    else:
                        preview_doc = None
                    if preview_doc:
                        from core.extractor import segment
                        from core.web_sources import select_search_queries
                        from core.wikipedia import wikipedia_queries
                        pieces=segment(preview_doc['text'],preview_doc['pages'])
                        if provider in ('tavily','both') and online:
                            st.write('Tavily query previews:',select_search_queries(pieces))
                        if provider in ('wikipedia','both') and online:
                            st.write('Wikipedia query previews:',wikipedia_queries(pieces))
                        if not online:
                            st.info('Online provider is off; no search queries will be sent.')
                    else:
                        st.info('Upload or paste a document first.')
                except ExtractionError as exc:
                    st.error(str(exc))
        allow_external = True
        if online or url.strip():
            allow_external = st.checkbox('I am authorized to send selected phrases / fetch this public URL', value=False)
            st.caption('Web requests can consume API credits. Do not submit private text without authorization.')
        if online:
            if provider == 'both':
                st.info('**Combined search:** Wikipedia and Tavily search independently. Accessible full-text sources are merged and deduplicated by URL. A failure in one provider will not cancel the other.')
            elif provider == 'wikipedia':
                st.caption('Wikipedia searches English Wikipedia only. No API key is required, but an identifying contact is needed.')
            else:
                st.caption('Tavily discovers publicly indexed web pages. Search/extraction may consume API credits.')
        with st.expander('Advanced · Matching thresholds and duplicate controls'):
            lex_threshold = st.slider('Lexical candidate threshold', .10, .80, .32, .02)
            sem_threshold = st.slider('Semantic candidate threshold', .45, .95, .70, .02)
            skip_identical = st.checkbox('Exclude identical local references — testing only', value=False)
            st.caption('A genuine duplicate submission can matter. Keep exclusion OFF for real comparisons. Thresholds are demonstration settings, not calibrated misconduct rules.')
    st.divider()
    if not do_patterns and not do_local and not online and not url.strip():
        st.info('Select at least one analysis mode to continue.')
    if st.button('Run analysis', type='primary', use_container_width=True):
        if not allow_external:
            st.error('Enable the external-sharing consent checkbox or switch off web and direct URL checking.')
            return
        if not do_patterns and not do_local and not online and not url.strip():
            st.error('Select at least one analysis mode.'); return
        if not do_patterns and not online and not url.strip() and (not do_local or not num_refs):
            st.error('No comparison sources are available. Add a reference document, enable web checking, or enable experimental writing analysis.'); return
        try:
            if mode == 'Upload a file':
                if uploaded is None:
                    st.error('Choose a PDF, DOCX or TXT document before running the analysis.'); return
                doc = extract_upload(uploaded.name, uploaded.getvalue())
            else:
                doc = extract_paste(pasted)
        except ExtractionError as exc:
            st.error(str(exc)); return
        with st.spinner('Extracting passages, checking selected sources, and preparing evidence…'):
            try:
                result = analyze_document(doc, local_sources() if do_local else [],
                                          semantic=do_semantic, online=online,
                                          online_provider=provider, wikipedia_contact=wiki_contact,
                                          direct_url=url, ai_patterns=do_patterns,
                                          lexical_threshold=lex_threshold,
                                          semantic_threshold=sem_threshold,
                                          exclude_identical_references=skip_identical,
                                          deep_search=deep_search)
                result['analysis_settings'] = {
                    'semantic_requested': do_semantic, 'writing_requested': do_patterns,
                    'local_requested': do_local, 'online_provider': provider if online else None,
                    'direct_url_requested': bool(url.strip()),
                    'deep_search_requested': deep_search,
                    'providers_requested': ([provider] if provider != 'both' else ['wikipedia','tavily']) if online else []}
                submission_id = store.add_document(doc, kind='submission')
                analysis_id = store.save_analysis(submission_id, result)
            except Exception as exc:
                st.error(f'Analysis could not finish ({type(exc).__name__}). Inspect the terminal and retry.'); return
        st.session_state['recent_analysis'] = analysis_id
        st.session_state['last_preview_id'] = analysis_id
        st.success('Analysis saved locally. You can now open the full evidence and writing-pattern results.')
    preview_id = st.session_state.get('last_preview_id')
    if preview_id:
        preview_record = store.get_analysis(preview_id)
        if preview_record:
            preview = preview_record['result']
            st.markdown('#### Latest analysis · Source similarity')
            show_source_metrics(preview, compact=True)
            st.markdown('#### Latest analysis · AI-writing patterns')
            ai_info = writing_summary(preview.get('ai_pattern_analysis'))
            if ai_info['state'] == 'experimental':
                st.info(ai_info['headline'] + '. ' + ai_info['detail'])
            elif ai_info['state'] in ('unavailable', 'insufficient_text'):
                st.warning(ai_info['headline'] + '. ' + ai_info['detail'])
            else:
                st.caption('Experimental writing-pattern analysis was not selected for this run.')
            if preview.get('provider_status'):
                with st.expander('Web retrieval breakdown'):
                    for name, detail in preview['provider_status'].items():
                        st.write(f'**{name.title()}** · {detail.get("retrieved", 0)} full-text source(s) · {detail.get("matched_sources", 0)} source(s) matched · {detail.get("state", "unknown")}')
            if preview.get('warnings'):
                with st.expander(f'Processing notes ({len(preview["warnings"])})'):
                    for note in preview['warnings']:
                        st.warning(note)
            if st.button('Open full results & AI-writing details', type='primary', use_container_width=True):
                go_to('Results & Review')


def page_library():
    heading('Reference Library', 'Manage the documents that source similarity is checked against.')
    st.info('A local reference library does not search the internet automatically. Wikipedia and Tavily are selected separately on Analyze.')
    with st.container(border=True):
        st.markdown('#### Add references')
        uploads = st.file_uploader('Choose PDF, DOCX or TXT reference documents',
                                   type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if st.button('Add selected documents', type='primary'):
            if not uploads:
                st.warning('Select at least one document.')
            else:
                old_ids = {r['id'] for r in store.list_documents('reference')}
                for upload in uploads:
                    try:
                        item = extract_upload(upload.name, upload.getvalue())
                        identity = store.add_document(item, 'reference')
                        st.success(f'{upload.name}: '+('already indexed' if identity in old_ids else 'added'))
                    except ExtractionError as exc:
                        st.error(f'{upload.name}: {exc}')
    c1,c2=st.columns(2)
    with c1:
        if st.button('Load two example reference files'):
            for name in ['library_ml.txt', 'library_energy.txt']:
                path = ROOT/'samples'/name
                store.add_document(extract_upload(name,path.read_bytes()),'reference')
            st.rerun()
    with c2:
        with st.expander('Controlled one-file QA test'):
            st.write('Load three known reference files, then upload **qa/ONE_FILE_FEATURE_TEST.txt** on Analyze. Do not index the uploaded test file as a reference.')
            if st.button('Load QA reference collection', key='qa_load'):
                for name in ['samples/library_ml.txt', 'samples/library_energy.txt', 'qa/qa_reference_common.txt']:
                    path=ROOT/name
                    store.add_document(extract_upload(path.name,path.read_bytes()), 'reference')
                st.rerun()
            st.caption('Controlled fixture: 26.07% exact coverage. The number of displayed evidence cards may increase with improved retrieval.')
    docs = store.list_documents('reference')
    st.subheader(f'Indexed reference documents ({len(docs)})')
    if not docs:
        st.info('No references yet. Upload reference documents, load the examples, or select public search on Analyze.')
    for item in docs:
        with st.expander(f'{item["title"]} · {item["text_length"]:,} characters'):
            st.caption(f'Type: {item["file_type"].upper()} · Added: {item["created_at"]}')
            confirmed=st.checkbox('I want to delete this reference',key='confirm_ref_'+item['id'])
            if st.button('Delete reference', key='delete_ref_'+item['id'],disabled=not confirmed):
                store.remove_reference(item['id'])
                st.rerun()


def get_selected_analysis():
    rows = store.list_analyses()
    if not rows:
        st.info('No saved analyses. Use **Analyze** to run your first check.')
        if st.button('Go to Analyze'):
            go_to('Analyze')
        return None, None, None
    ids = [r['id'] for r in rows]
    recent = st.session_state.get('recent_analysis')
    ix = ids.index(recent) if recent in ids else 0
    chosen = st.selectbox('Choose saved analysis', ids, index=ix,
                          format_func=lambda key: next((f'{r["title"]} · {r["created_at"][:16]}' for r in rows if r['id'] == key),key))
    record = store.get_analysis(chosen)
    document = store.get_document(record['document_id']) if record else None
    return chosen, record['result'] if record else None, document


def page_results():
    heading('Results & Review', 'Two independent reports: source similarity and experimental writing patterns.')
    aid, result, document = get_selected_analysis()
    if result is None:
        return
    settings=result.get('analysis_settings') or {}
    with st.container(border=True):
        st.markdown('### A · Source similarity')
        info=show_source_metrics(result)
        col1,col2=st.columns(2)
        col1.caption(f'Local sources: {info["local"]} · Retrieved full web sources: {info["web"]}')
        label, detail = model_execution_label(result, requested=settings.get('semantic_requested'))
        col2.caption(f'Semantic model: {label}')
        with st.expander('What does this source result mean?'):
            st.write(detail)
            st.write('A 0.0% score with sources checked means no eligible exact alignment was counted; semantic-only candidates can still appear separately. N/A means no sources were checked.')
        if result.get('warnings'):
            with st.expander(f'Warnings and retrieval limits ({len(result["warnings"])})'):
                for warning in result['warnings']:
                    st.warning(warning)
    providers = result.get('provider_status') or {}
    if providers:
        st.markdown('#### Public web search coverage')
        cols = st.columns(len(providers))
        for col, (name, state) in zip(cols, providers.items()):
            with col:
                st.metric(name.title() + ' pages retrieved', state.get('retrieved', 0))
                st.caption(f'{state.get("matched_sources", 0)} actually matched · '+state.get('state', 'unknown').replace('_', ' '))
        st.caption('Retrieved pages and pages with verified matches are different. Zero source matches does not establish originality. For a known Wikipedia excerpt, a direct article URL is the most reliable check.')
    inventory = result.get('source_inventory') or []
    if inventory:
        with st.expander('All sources actually compared ('+str(len(inventory))+')'):
            for entry in inventory:
                st.markdown('**'+html.escape(str(entry.get('title','Untitled')))+'** · '+html.escape(str(entry.get('provider','local'))))
                if entry.get('url'):
                    st.caption(entry['url'])
                if entry.get('discovery_query'):
                    st.caption('Found using query: '+entry['discovery_query'])
                if len(entry.get('discovered_via') or []) > 1:
                    st.caption('Found via: '+', '.join(entry['discovered_via']))
    st.divider()
    with st.container(border=True):
        st.markdown('### B · AI-writing analysis (independent)')
        show_writing_result(result, (document or {}).get('text',''), aid)
    st.divider()
    st.subheader('C · Matching source passages')
    st.caption('Yellow marks show verified equal-word runs only. Semantically related passages may not contain marked words.')
    matches = result.get('matches') or []
    if not matches:
        st.warning('No matching source passages were found among the specific sources retrieved. This is **not** a clean internet-plagiarism result. For text copied from Wikipedia, paste its original article URL into the direct URL field and rerun. This separates failed source discovery from failed matching.')
    else:
        category = st.selectbox('Filter source evidence', ['All'] + sorted({m['category'] for m in matches}))
        filtered = [m for m in matches if category == 'All' or m['category'] == category]
        reviews = store.reviews_for(aid)
        if not filtered:
            st.info('No matches in this evidence category.')
        for i, match in enumerate(filtered,1):
            label = f'{i}. {match["category"]} · {match["source_title"][:65]} · page {match["submitted_page"]}'
            with st.expander(label, expanded=i==1):
                a,b=st.columns(2,gap='medium')
                with a:
                    st.caption('SUBMITTED PASSAGE')
                    st.markdown(highlight(match['submitted_text'], match.get('input_spans') or []), unsafe_allow_html=True)
                with b:
                    st.caption('SOURCE PASSAGE')
                    st.markdown(highlight(match['source_text'], match.get('source_spans') or []), unsafe_allow_html=True)
                    if (match.get('source_url') or '').startswith(('https://','http://')):
                        st.link_button('Open source webpage',match['source_url'])
                x,y=st.columns(2)
                if match.get('exact_word_count'):
                    x.metric('Verified exact run', f'{match["exact_word_count"]} words')
                else:
                    x.metric('TF-IDF similarity', f'{match["lexical_score"]:.3f}')
                y.metric('MiniLM similarity', f'{match["semantic_score"]:.3f}' if match.get('semantic_score') is not None else 'Not run')
                st.caption(match.get('explanation') or 'Inspect the passages and source before deciding.')
                if match.get('citation_signals'):
                    st.info('Attribution signals: '+', '.join(match['citation_signals'])+'. Their presence does not prove adequate citation.')
                old=reviews.get(match['id'],{})
                decisions=['Unreviewed', 'Needs investigation', 'Properly attributed', 'Not relevant']
                decision=st.selectbox('Reviewer decision',decisions,
                                      index=decisions.index(old.get('decision','Unreviewed')),
                                      key='review_decision_'+match['id'])
                note=st.text_input('Review note (optional)',value=old.get('note',''),
                                   key='review_note_'+match['id'])
                if st.button('Save review decision', key='review_save_'+match['id']):
                    store.update_review(aid,match['id'],decision,note)
                    st.success('Review saved.')
                    st.rerun()
    st.divider()
    st.subheader('D · Export and manage this analysis')
    pdf=make_pdf(result,store.reviews_for(aid),store.writing_reviews_for(aid))
    st.download_button('Download complete evidence report (PDF)',data=pdf,
                       file_name='tracelens_evidence_report.pdf',mime='application/pdf',type='primary')
    with st.expander('Delete this saved analysis'):
        st.warning('Deletes saved matches, review notes and the associated submission when it is no longer used by another analysis. This cannot be undone.')
        confirmation=st.checkbox('I understand deletion is permanent',key='confirm_analysis_'+aid)
        if st.button('Permanently delete this analysis',disabled=not confirmation,key='delete_analysis_'+aid):
            store.delete_analysis(aid)
            if st.session_state.get('recent_analysis') == aid:
                del st.session_state['recent_analysis']
            st.rerun()


def page_comparison():
    heading('Algorithm Comparison', 'See exactly what TF-IDF measures and what MiniLM adds.')
    _,result,_ = get_selected_analysis()
    if result:
        if not result.get('references_checked'):
            st.info('No sources were compared in this analysis. Add references or enable public retrieval to generate a meaningful lexical-vs-semantic comparison.')
        elif not result.get('semantic_used'):
            st.info('MiniLM did not execute for this analysis. Install the optional AI dependencies, enable semantic matching, and run the document again.')
        if result.get('matches'):
            st.dataframe(pd.DataFrame([{'Source':m['source_title'], 'Category':m['category'],
                                       'TF-IDF cosine':m['lexical_score'],
                                       'MiniLM cosine':m['semantic_score']}
                                      for m in result['matches']]),hide_index=True,use_container_width=True)
        else:
            st.caption('No displayed passage matches to compare for this analysis.')
    st.subheader('Learning evaluation — sample pairs')
    st.caption('This educational toy benchmark is not evidence of real-world plagiarism or AI-authorship accuracy.')
    with_model=st.checkbox('Also evaluate MiniLM on the toy examples',value=False)
    if st.button('Run sample evaluation'):
        try:
            from evaluation.evaluate import compare_toy
            with st.spinner('Comparing the bundled example pairs...'):
                values=compare_toy(use_semantic=with_model)
            st.json({k:v for k,v in values.items() if k not in ('samples','labels','lexical','semantic')})
            st.dataframe(pd.DataFrame(values['samples'])[['submitted','source','related']],hide_index=True)
        except Exception as exc:
            st.error(f'Sample evaluation failed: {type(exc).__name__}. Check model installation and terminal output.')


def page_guide():
    heading('How to Use', 'A short guide to a meaningful, reproducible demonstration.')
    st.markdown('''
**1 · Add sources.** In **Reference Library**, index reference documents. For a quick controlled demo use **Load QA reference collection**.

**2 · Analyze.** In **Analyze**, upload `qa/ONE_FILE_FEATURE_TEST.txt`, leave web and MiniLM OFF for the reproducible QA baseline, and run. The expected controlled outcome is 26.07% exact coverage and six source findings if only those three references are indexed.

**3 · Inspect two different results.** **Results & Review** shows **Source similarity** first and **AI-writing analysis** second. Source similarity requires reference documents; the experimental writing-pattern classifier does not.

**4 · Test semantic AI.** Install `requirements-ai.txt`, select MiniLM and rerun the sample. Numeric MiniLM scores only appear if the model actually executed with eligible source passages.

**5 · Test public retrieval.** Choose **Wikipedia only**, **Tavily only**, or **Wikipedia + Tavily**. Both-provider mode records successes and failures independently and compares each URL once. Wikipedia is free but requires a **WIKIMEDIA_CONTACT** email or project URL. Enable Wikipedia, check external-sharing consent, and use a copied passage from a known accessible Wikipedia article. Tavily is optional and may consume credits.

**6 · Test experimental writing patterns.** Train the classifier first. Enable that option and upload `TEST_AI_GENERATED.txt`. A low score does not mean human authorship; this known AI-written test was previously missed by the experimental model.

**7 · Review and export.** Inspect each source pair and each writing window, save separate human review decisions and evidence notes, download the PDF, and run automated checks with `python -m pytest -q`.

**Interpretation:** No sources compared = **N/A**, not 0% originality. A similarity percentage is not a plagiarism verdict. AI-writing scores are experimental and cannot establish the author of a text.
''')
    with st.expander('Useful Windows commands'):
        st.code('''# Run from the extracted TraceLens-AI folder in PowerShell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-ai.txt
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m streamlit run app.py''',language='powershell')
    st.caption('Sensitive documents should stay local unless you have permission to send search queries to an external provider. Never post API keys or private student text in screenshots.')


with st.sidebar:
    st.markdown('## 🔎 TraceLens AI')
    st.caption('Explainable similarity · writing-pattern research')
    view=st.radio('Workspace', NAV, key='tl_nav', label_visibility='collapsed')
    st.divider()
    health_panel()
    st.caption('v1.7 · Local-first · English · Single-user prototype')
    st.caption('Source similarity ≠ plagiarism. Writing style ≠ proven AI authorship.')

if view == 'Analyze':
    page_analyze()
elif view == 'Results & Review':
    page_results()
elif view == 'Reference Library':
    page_library()
elif view == 'Model Comparison':
    page_comparison()
else:
    page_guide()
