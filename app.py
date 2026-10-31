"""TraceLens AI · Workspace redesign (v1.9 UI).

Frontend-only rebuild with explicit routes for source matching, writing research,
reference management, and human review. The detection backends remain unchanged.
"""
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
from core.presentation import model_execution_label, writing_summary
from core.report import make_pdf
from db.store import Store
from ui.logic import check_readiness, result_overview, safe_filename
from ui.theme import hero, init_theme, section

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='TraceLens | Document intelligence', page_icon='🔎',
                   layout='wide', initial_sidebar_state='expanded')
init_theme()

NAV = ['Analyze', 'Results & Review', 'Reference Library', 'Models & Diagnostics', 'Guide']
redirect = st.session_state.pop('_tl_route', None)
if redirect in NAV:
    st.session_state['tl_nav'] = redirect
if st.session_state.get('tl_nav', 'Analyze') not in NAV:
    st.session_state['tl_nav'] = 'Analyze'
store = Store()


def navigate(page: str):
    st.session_state['_tl_route'] = page
    st.rerun()


def safe(s) -> str:
    return html.escape(str(s if s is not None else ''))


def local_references() -> list[dict]:
    result = []
    for row in store.list_documents('reference'):
        d = store.get_document(row['id'])
        if d:
            result.append({'id': d['id'], 'title': d['title'], 'text': d['text'],
                           'pages': d['pages'], 'type': 'local', 'url': None})
    return result


def install_demo_sources():
    for name in ('samples/library_ml.txt', 'samples/library_energy.txt', 'qa/qa_reference_common.txt'):
        p = ROOT / name
        store.add_document(extract_upload(p.name, p.read_bytes()), 'reference')


def highlighted(text: str, spans: list | None) -> str:
    """Preserve original character offsets and escape source content."""
    size = len(text)
    pairs = []
    for item in (spans or []):
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        a, b = max(0, min(size, start)), max(0, min(size, end))
        if a < b:
            pairs.append((a,b))
    pairs.sort()
    merged = []
    for a,b in pairs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a,b))
    out, cursor = [], 0
    for a,b in merged:
        out.append(safe(text[cursor:a]))
        out.append('<mark>'+safe(text[a:b])+'</mark>')
        cursor = b
    out.append(safe(text[cursor:]))
    return '<div class="tl-compare">'+''.join(out)+'</div>'


def system_status():
    refs = len(store.list_documents('reference'))
    contact = bool(os.getenv('WIKIMEDIA_CONTACT', '').strip())
    tavily = bool(os.getenv('TAVILY_API_KEY', '').strip()) or os.getenv('TAVILY_KEYLESS') == '1'
    mini = importlib.util.find_spec('sentence_transformers') is not None
    ngram = (ROOT/'storage/ai_ngram_candidate.joblib').exists()
    return {'refs': refs, 'wiki': contact, 'tavily': tavily,
            'semantic': mini, 'ngram': ngram, 'legacy': MODEL_FILE.exists()}


def header_metrics(result: dict, document: dict | None = None):
    model = result_overview(result, (document or {}).get('text',''))
    source = model['source']
    cols = st.columns(4)
    cols[0].metric('Exact source coverage', source['coverage_label'],
                   help='Not a plagiarism verdict. N/A means zero actual sources compared.')
    cols[1].metric('Source evidence', source['findings'])
    cols[2].metric('Sources compared', source['references'])
    cols[3].metric('AI-writing signals',
                   f"{model['writing']['flags']} / {model['writing']['count']}" if model['writing']['state']=='ready' else 'Not assessed',
                   help='Experimental classifier windows for human review; not a percentage of AI-written text.')
    return model


def page_analyze():
    hero('01 / Analysis workspace', 'Check your document with confidence',
         'Choose your document, select exactly which sources to compare, and review plagiarism evidence and experimental writing signals separately.')
    status = system_status()
    st.markdown('<span class="tl-chip">Local-first</span><span class="tl-chip">Source-linked evidence</span>'
                '<span class="tl-chip">Human review</span>', unsafe_allow_html=True)

    left, right = st.columns([1.12, 1], gap='large')
    with left:
        section('Step 1', 'Add the document', 'Upload PDF, DOCX or TXT, or paste a passage directly.')
        mode = st.radio('Document input', ['Upload document','Paste text'], horizontal=True,
                        label_visibility='collapsed', key='tl_input_mode')
        upload, pasted = None, ''
        if mode == 'Upload document':
            upload = st.file_uploader('Document file', type=['pdf','docx','txt'],
                help='Maximum 10 MB. Text-based PDFs supported; scanned PDF OCR is not included.',
                key='tl_upload')
            if upload:
                st.caption(f'✓ Selected: {upload.name} · {len(upload.getvalue())/1024:.1f} KB')
        else:
            pasted = st.text_area('Document text', height=238,
                placeholder='Paste an assignment, essay, or an excerpt from a public article…', key='tl_pasted')
            st.caption(f'{len(pasted.split()):,} words entered')
        st.caption('Your uploaded text is saved locally. Text is sent externally only if you choose online search and consent.')

        section('Step 2', 'Choose detection methods', 'A source comparison needs reference documents or accessible webpages.')
        local_enabled = st.checkbox(f'Compare with local library · {status["refs"]} indexed',
                                    value=True, key='tl_local')
        if local_enabled and not status['refs']:
            with st.container(border=True):
                st.info('**Your reference library is empty.** Local plagiarism checking cannot run until you add sources.')
                if st.button('Load 3 sample reference documents', key='tl_quick_refs'):
                    install_demo_sources()
                    st.rerun()
                if st.button('Open Reference Library', key='tl_to_refs'):
                    navigate('Reference Library')
        writing_enabled = st.checkbox('Check AI-writing patterns · experimental',
             help='Uses your installed v2 n-gram / MiniLM candidates and legacy baseline where available. Results are not proof of authorship.',
             value=False, key='tl_writing')
        if writing_enabled:
            if status['ngram']:
                st.caption('✓ Installed n-gram candidate found. Results will lead with v2 predictions.')
            elif not status['legacy']:
                st.warning('No AI-writing model installed. Run training before expecting classifier scores.')
            else:
                st.warning('Only the legacy classifier is installed. The more recent candidate is not yet available.')
        semantic = st.checkbox('Use MiniLM semantic source similarity', value=status['semantic'],
                               help='Finds related passages against references, NOT AI authorship.', key='tl_semantic')
        if semantic and not status['semantic']:
            st.warning('Sentence Transformers is not installed. Exact and TF-IDF checks can still run.')

    with right:
        section('Step 3', 'Choose online sources', 'Search for public material or compare directly against a known source URL.')
        provider_label = st.selectbox('Web search provider',
            ['Off · keep text local', 'Wikipedia', 'Tavily', 'Wikipedia + Tavily'],
            key='tl_provider')
        provider = {'Off · keep text local':'off', 'Wikipedia':'wikipedia',
                    'Tavily':'tavily', 'Wikipedia + Tavily':'both'}[provider_label]
        online = provider != 'off'
        url = st.text_input('Optional public article URL',
            placeholder='https://en.wikipedia.org/wiki/Machine_learning', key='tl_direct_url',
            help='Useful if automatic discovery misses the original source.')
        thorough = st.checkbox('Thorough search · more pages and potential API credits',
                               value=False, key='tl_thorough') if online else False
        needs_contact = provider in ('wikipedia','both') or url.strip().startswith('https://en.wikipedia.org/wiki/')
        contact = st.text_input('Wikimedia contact', value=os.getenv('WIKIMEDIA_CONTACT',''),
                                placeholder='name@example.com', key='tl_contact',
                                help='Identifying contact for Wikimedia requests; not an API key.') if needs_contact else ''
        if online:
            st.caption('Wikipedia searches encyclopedia articles; Tavily searches broader public webpages. Both-provider mode deduplicates identical URLs.')
        if online or url.strip():
            consent = st.checkbox('I have authorization to submit search phrases and fetch these public sources',
                                  value=False, key='tl_consent')
        else:
            consent = False
        with st.expander('Search preview & advanced matching', expanded=False):
            st.caption('Outgoing phrases are previewed locally, not submitted by this action.')
            if online and st.button('Show proposed web queries', key='tl_queries'):
                try:
                    d = extract_upload(upload.name, upload.getvalue()) if mode=='Upload document' and upload else (
                        extract_paste(pasted) if mode=='Paste text' and pasted.strip() else None)
                    if not d:
                        st.info('Add a document first.')
                    else:
                        from core.extractor import segment
                        from core.web_sources import select_search_queries
                        from core.wikipedia import wikipedia_queries
                        chunks = segment(d['text'], d['pages'])
                        if provider in ('wikipedia','both'):
                            st.write('Wikipedia:', wikipedia_queries(chunks))
                        if provider in ('tavily','both'):
                            st.write('Tavily:', select_search_queries(chunks))
                except ExtractionError as exc:
                    st.warning(str(exc))
            lexical_threshold = st.slider('Lexical candidate threshold', .10, .80, .32, .02)
            semantic_threshold = st.slider('Semantic candidate threshold', .45, .95, .70, .02)
            exclude_identical = st.checkbox('Exclude identical local reference (test only)', value=False)
            st.caption('Keep identical-source exclusion OFF for real document comparison.')

    has_document = bool(upload) if mode=='Upload document' else bool(pasted.strip())
    readiness = check_readiness(has_document=has_document, local_enabled=local_enabled,
             reference_count=status['refs'], online_provider=provider, direct_url=url,
             writing_enabled=writing_enabled, external_consent=consent,
             wikipedia_contact=contact, tavily_configured=status['tavily'])
    section('Ready to analyze', 'Preflight check', 'Only selected sources and enabled detection methods will run.')
    with st.container(border=True):
        if readiness.checks:
            st.write('**Selected checks:** ' + ' · '.join(readiness.checks))
        for note in readiness.warnings:
            st.caption('⚠ ' + note)
        if readiness.reasons:
            for reason in readiness.reasons:
                st.info(reason)
        else:
            st.success('Your configuration is ready. Start analysis when you are ready.')
        run = st.button('Run document analysis', type='primary', use_container_width=True,
                        disabled=not readiness.can_run, key='tl_run')
    if run:
        try:
            doc = extract_upload(upload.name, upload.getvalue()) if mode=='Upload document' else extract_paste(pasted)
            with st.spinner('Reading the document, matching evidence, and scoring selected classifiers…'):
                result = analyze_document(doc, local_references() if local_enabled else [],
                        semantic=semantic, online=online, direct_url=url,
                        online_provider=provider if online else 'tavily',
                        wikipedia_contact=contact if needs_contact else None,
                        ai_patterns=writing_enabled, deep_search=thorough,
                        lexical_threshold=lexical_threshold,
                        semantic_threshold=semantic_threshold,
                        exclude_identical_references=exclude_identical)
                result['analysis_settings'] = {'semantic_requested':semantic,
                    'writing_requested':writing_enabled,'local_requested':local_enabled,
                    'online_provider':provider if online else None,
                    'providers_requested':(['wikipedia','tavily'] if provider=='both' else [provider] if online else []),
                    'direct_url_requested':bool(url.strip()), 'deep_search_requested':thorough}
                document_id = store.add_document(doc,'submission')
                analysis_id = store.save_analysis(document_id,result)
            st.session_state['tl_latest_analysis'] = analysis_id
            st.session_state['tl_selected_analysis'] = analysis_id
            st.success('Analysis completed and saved locally.')
            navigate('Results & Review')
        except ExtractionError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error('Analysis could not finish ('+type(exc).__name__+'). See the PowerShell terminal for diagnostics.')

    analyses = store.list_analyses(limit=3)
    if analyses:
        section('Recent work', 'Continue reviewing', 'Your three most recent saved analyses.')
        for row in analyses:
            a,b = st.columns([4,1])
            a.write('**'+safe(row['title'])+'** · '+safe(row['created_at'][:16]))
            if b.button('Review', key='recent_'+row['id']):
                st.session_state['tl_latest_analysis'] = row['id']
                st.session_state['tl_selected_analysis'] = row['id']
                navigate('Results & Review')


def page_library():
    hero('Reference management', 'Your reference library',
         'Index local documents to compare their passages with submitted work. Web search is configured separately in Analyze.')
    docs = store.list_documents('reference')
    c1,c2,c3=st.columns(3)
    c1.metric('Indexed documents', len(docs))
    c2.metric('Indexed characters', f'{sum(int(d.get("text_length",0)) for d in docs):,}')
    c3.metric('Availability', 'Ready' if docs else 'Empty')
    left,right=st.columns([1.45,1], gap='large')
    with left:
        section('Add sources', 'Upload reference files', 'Multiple PDF, DOCX, or TXT files can be added together.')
        with st.container(border=True):
            files=st.file_uploader('Choose reference files', type=['pdf','docx','txt'],
                                   accept_multiple_files=True, key='tl_ref_files')
            if st.button('Index selected files', type='primary', disabled=not bool(files), key='tl_index_refs'):
                added, failed=0,[]
                for f in files:
                    try:
                        extracted=extract_upload(f.name,f.getvalue())
                        store.add_document(extracted,'reference')
                        added+=1
                    except ExtractionError as exc:
                        failed.append(f'{f.name}: {exc}')
                if added: st.success(f'{added} reference file(s) processed.')
                for message in failed: st.error(message)
                if added: st.rerun()
    with right:
        section('Quick demo', 'Load known test references', 'Three small documents for reproducible local checking.')
        with st.container(border=True):
            st.write('**Controlled QA collection**')
            st.caption('Use with qa/ONE_FILE_FEATURE_TEST.txt. The controlled fixture produces 26.07% exact matched-word coverage when only these three sources are indexed.')
            if st.button('Load all three QA references', key='tl_load_qa', use_container_width=True):
                install_demo_sources()
                st.rerun()
            if st.button('Analyze a sample', key='tl_demo_go', use_container_width=True):
                navigate('Analyze')
    section('Your sources','Indexed documents', 'Remove individual sources using explicit confirmation.')
    if not docs:
        st.info('No indexed references yet. Upload files or load the controlled QA collection above.')
    else:
        query = st.text_input('Find a reference', placeholder='Search by filename…', key='tl_find_ref')
        shown=[x for x in docs if query.lower() in x['title'].lower()]
        st.caption(f'Showing {len(shown)} of {len(docs)} indexed documents.')
        for ref in shown:
            with st.expander(f'{ref["title"]} · {int(ref["text_length"]):,} characters'):
                st.caption('Added '+str(ref.get('created_at',''))+' · '+str(ref.get('file_type','')).upper())
                confirmed=st.checkbox('Confirm deletion of this local reference',key='ref_confirm_'+ref['id'])
                if st.button('Delete reference', key='ref_delete_'+ref['id'], disabled=not confirmed):
                    store.remove_reference(ref['id'])
                    st.rerun()


def selected_analysis():
    rows = store.list_analyses()
    if not rows:
        st.info('No analyses have been saved yet. Upload a document on Analyze.')
        if st.button('Start a new analysis',key='tl_first_analyze'): navigate('Analyze')
        return None,None,None
    ids=[r['id'] for r in rows]
    if st.session_state.get('tl_selected_analysis') not in (None, *ids):
        st.session_state.pop('tl_selected_analysis',None)
    recent=st.session_state.get('tl_latest_analysis')
    ix=ids.index(recent) if recent in ids else 0
    chosen=st.selectbox('Saved analysis', ids, index=ix,
        format_func=lambda k:next((f'{r["title"]} · {r["created_at"][:16]}' for r in rows if r['id']==k),k),
        key='tl_selected_analysis')
    record=store.get_analysis(chosen)
    doc=store.get_document(record['document_id']) if record else None
    return chosen,record['result'] if record else None,doc


def source_evidence(analysis_id: str, result: dict):
    section('Evidence review','Matching passages', 'Review the submitted passage alongside its originating reference.')
    matches=result.get('matches') or []
    if not matches:
        st.info('No matching passages met the selected thresholds among sources actually compared. This does not establish originality.')
        return
    category=st.selectbox('Evidence type', ['All']+sorted({m.get('category','Other') for m in matches}),key='tl_evidence_type')
    filtered=[m for m in matches if category=='All' or m.get('category','Other')==category]
    page_count=max(1,(len(filtered)+9)//10)
    page=st.number_input('Evidence page', min_value=1, max_value=page_count, value=1, step=1,
                         key='tl_evidence_page') if page_count>1 else 1
    start=(int(page)-1)*10
    st.caption(f'Displaying evidence {start+1}–{min(start+10,len(filtered))} of {len(filtered)}.')
    reviews=store.reviews_for(analysis_id)
    for i,m in enumerate(filtered[start:start+10],start+1):
        title=f"{i:02d}  ·  {m.get('category','Match')}  ·  {m.get('source_title','Unknown source')[:75]}"
        with st.expander(title, expanded=i==start+1):
            a,b=st.columns(2,gap='large')
            with a:
                st.caption('SUBMITTED TEXT')
                st.markdown(highlighted(str(m.get('submitted_text','')),m.get('input_spans')),
                            unsafe_allow_html=True)
            with b:
                st.caption('REFERENCE SOURCE')
                st.markdown(highlighted(str(m.get('source_text','')),m.get('source_spans')),
                            unsafe_allow_html=True)
                link=m.get('source_url') or ''
                if link.startswith(('https://','http://')):
                    st.link_button('Open original article', link)
            c,d=st.columns(2)
            c.metric('Longest verified word run', str(m.get('exact_word_count') or 'N/A'))
            d.metric('MiniLM cosine', f"{float(m['semantic_score']):.3f}" if m.get('semantic_score') is not None else 'Not run')
            st.caption(m.get('explanation') or 'Inspect the shared text and attribution before making a decision.')
            if m.get('citation_signals'):
                st.info('Citation/quotation signals: '+', '.join(m['citation_signals']))
            old=reviews.get(m['id'],{})
            options=['Unreviewed','Needs investigation','Properly attributed','Not relevant']
            with st.form('source_review_form_'+analysis_id+'_'+m['id']):
                decision=st.selectbox('Human review decision',options,index=options.index(old.get('decision','Unreviewed'))
                                        if old.get('decision','Unreviewed') in options else 0)
                note=st.text_area('Review note',value=old.get('note',''),height=85,max_chars=1000)
                if st.form_submit_button('Save source review'):
                    store.update_review(analysis_id,m['id'],decision,note)
                    st.success('Source review saved independently of the similarity scores.')
                    st.rerun()


def candidate_review_form(analysis_id: str, method: str, idx: int, text: str):
    decisions=['Unreviewed','Discuss with author','AI assistance disclosed','No concern','Insufficient evidence']
    previous=store.candidate_reviews_for(analysis_id,method).get(idx,{})
    value=previous.get('decision','Unreviewed')
    with st.form(f'candidate_review_{analysis_id}_{method}_{idx}'):
        decision=st.selectbox('Reviewer decision',decisions,index=decisions.index(value) if value in decisions else 0)
        note=st.text_area('Context / evidence note',value=previous.get('note',''),max_chars=2000,
                          height=90,placeholder='e.g. observed draft history, author disclosure, uncertainty…')
        if st.form_submit_button('Save writing review'):
            store.update_candidate_review(analysis_id,method,idx,decision,note)
            st.success('Human review saved separately from model predictions.')
            st.rerun()


def writing_evidence(analysis_id: str, result: dict, doc: dict | None):
    section('Writing research','Experimental AI-writing analysis',
            'Classifier scores are not authorship probabilities. Flagging a passage only suggests contextual human review.')
    model=result_overview(result,(doc or {}).get('text',''))
    snap=model['writing']
    legacy=result.get('ai_pattern_analysis') or {}
    if snap['state'] == 'ready':
        method=snap['primary']
        st.caption('PRIMARY RESEARCH MODEL · '+method.upper()+' · Saved scores from this analysis')
        c1,c2,c3=st.columns(3)
        c1.metric('Windows assessed',snap['count'])
        c2.metric('Review suggestions',f"{snap['flags']} / {snap['count']}")
        c3.metric('Review threshold',f"{snap['threshold']:.4f}")
        if snap['flags']:
            st.warning(f'{snap["flags"]} window(s) crossed this experimental threshold. Inspect passages; do not infer authorship.')
        else:
            st.info('No windows crossed the review threshold. This does not establish human authorship.')
        st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!='text'} for r in snap['rows']]),
            hide_index=True,use_container_width=True,
            column_config={'Model score':st.column_config.NumberColumn(format='%.4f'),
                           'Threshold':st.column_config.NumberColumn(format='%.4f')})
        text=(doc or {}).get('text','')
        section('Human review','Inspect and review writing windows', 'Each reviewer decision persists in a separate v2 table.')
        choice = st.selectbox('Select writing window to inspect',
            list(range(len(snap['rows']))), format_func=lambda i:
            f"Window {i+1} · {snap['rows'][i]['Review signal']} · score {snap['rows'][i]['Model score']:.4f}",
            key='tl_v2_window_choice')
        row=snap['rows'][choice]
        previous=store.candidate_reviews_for(analysis_id,method).get(choice,{})
        with st.container(border=True):
            st.markdown('**Window '+str(choice+1)+'** · '+row['Review signal'])
            st.markdown('<div class="tl-excerpt">'+safe(row.get('text') or 'Saved window excerpt unavailable.')+'</div>',
                        unsafe_allow_html=True)
            st.caption('Model: '+method.upper()+' · a review signal, not a validated AI-authorship probability.')
            if previous.get('decision','Unreviewed')!='Unreviewed':
                st.caption('Previous reviewer decision: '+previous['decision'])
            candidate_review_form(analysis_id,method,choice,text)
        other={k:v for k,v in snap['models'].items() if k!=method}
        if other:
            with st.expander('Compare with other installed writing classifiers'):
                for k,v in other.items():
                    st.write('**'+k.upper()+'**')
                    if v.get('status')=='experimental':
                        st.caption(f"Threshold: {float(v['threshold']):.4f} · {sum(v['flags'])}/{len(v['flags'])} review signals")
                        st.write('Window scores: '+', '.join(f'{float(x):.4f}' for x in v.get('scores',[])))
                    else:
                        st.caption(v.get('message','Unavailable'))
    else:
        st.info(snap.get('message','No v2 writing scores were saved in this analysis.'))
        if legacy.get('status')=='experimental':
            st.warning('This saved record contains legacy writing scores only. Run a NEW analysis after installing v2 to see its results.')
    with st.expander('Legacy seven-feature classifier · historical results'):
        summary=writing_summary(legacy)
        st.caption('This separate legacy baseline may disagree with v2. Its review notes are stored independently.')
        st.write('**'+summary['headline']+'**')
        if legacy.get('status')=='experimental':
            windows=legacy.get('windows') or []
            if any(w.get('distribution_warning') for w in windows):
                st.warning('Legacy model observed training-distribution shift. Treat these windows as inconclusive.')
            st.dataframe(pd.DataFrame([{'Window':i+1,'Score':float(w['score']),
                 'Flagged':bool(w.get('review_flag')),'Outcome':w.get('outcome','')}
                 for i,w in enumerate(windows)]),hide_index=True,use_container_width=True)
            st.caption('Legacy human reviews are preserved in the existing database and PDF export.')
        else:
            st.caption(summary['detail'])


def results_overview(analysis_id: str, result: dict, document: dict | None):
    overview=header_metrics(result,document)
    if overview['source']['state']=='not_assessed':
        st.info('Source matching not assessed: no actual local or public source was compared.')
    else:
        st.caption(overview['source']['detail'])
    if overview['writing']['state']=='ready':
        st.caption('Writing model: '+overview['writing']['primary'].upper()+' · '+overview['ai_detail'])
    elif (result.get('analysis_settings') or {}).get('writing_requested'):
        st.warning(overview['ai_detail'])
    st.divider()
    section('Search transparency','What was actually searched','Retrieved public sources are different from sources containing evidence.')
    states=result.get('provider_status') or {}
    if not states:
        st.caption('No automatic public source provider was requested for this analysis.')
    else:
        cols=st.columns(len(states))
        for col,(name,item) in zip(cols,states.items()):
            col.metric(name.title()+' pages',item.get('retrieved',0))
            col.caption(f"{item.get('matched_sources',0)} matched · {str(item.get('state','unknown')).replace('_',' ')}")
    if result.get('warnings'):
        with st.expander(f"Run notes and retrieval limits ({len(result['warnings'])})"):
            for warning in result['warnings']:
                st.warning(warning)
    with st.expander(f"All sources compared ({len(result.get('source_inventory') or [])})"):
        for source in result.get('source_inventory') or []:
            st.write('**'+safe(source.get('title','Untitled'))+'** · '+safe(source.get('provider','local')))
            if source.get('url'):
                st.caption(source['url'])


def page_results():
    hero('02 / Evidence workspace', 'Review what TraceLens found',
         'Source matches, writing-pattern signals and human decisions live in separate, traceable sections.')
    aid,result,document=selected_analysis()
    if result is None:
        return
    st.caption('Viewing a saved snapshot. Installing a new model will not change scores stored in earlier analyses.')
    overview_tab,sources_tab,writing_tab,export_tab=st.tabs(
        ['Overview','Source evidence','AI-writing review','Report & history'])
    with overview_tab:
        results_overview(aid,result,document)
        st.caption('Use the **Source evidence** and **AI-writing review** tabs above to inspect and review individual passages.')
    with sources_tab:
        source_evidence(aid,result)
    with writing_tab:
        writing_evidence(aid,result,document)
    with export_tab:
        section('Final step','Export an evidence report','Includes source evidence, model comparisons and saved human review decisions.')
        pdf=make_pdf(result,store.reviews_for(aid),store.writing_reviews_for(aid),
                     store.candidate_reviews_for(aid))
        st.download_button('Download PDF report',pdf,file_name=safe_filename((document or {}).get('title','document')),
                           mime='application/pdf',type='primary',use_container_width=True,
                           key='tl_export_pdf')
        st.caption('The report documents checked evidence and reviewer decisions. It is not an automated misconduct verdict.')
        with st.expander('Delete this saved analysis permanently'):
            st.warning('This also deletes all associated review notes. It cannot be undone.')
            confirm=st.checkbox('I understand this cannot be undone',key='tl_delete_confirm_'+aid)
            if st.button('Delete saved analysis',disabled=not confirm,key='tl_delete_analysis_'+aid):
                store.delete_analysis(aid)
                st.session_state.pop('tl_latest_analysis',None)
                st.session_state.pop('tl_selected_analysis',None)
                st.rerun()


def page_models():
    hero('Model laboratory', 'Models & diagnostics',
         'See what is installed, compare lexical/semantic source similarity, and keep experimental AI-writing detection evidence-based.')
    state=system_status()
    status=[('Local reference library',state['refs']>0,f"{state['refs']} documents"),
            ('Wikipedia',state['wiki'],'Identifying contact configured' if state['wiki'] else 'Contact missing'),
            ('Tavily',state['tavily'],'API configured' if state['tavily'] else 'Not configured'),
            ('MiniLM semantic package',state['semantic'],'Installed' if state['semantic'] else 'Install requirements-ai.txt'),
            ('V2 n-gram writing candidate',state['ngram'],'Installed' if state['ngram'] else 'Not installed'),
            ('Legacy writing classifier',state['legacy'],'Installed' if state['legacy'] else 'Not installed')]
    for name,ready,explain in status:
        a,b,c=st.columns([3,1,3])
        a.write('**'+name+'**')
        b.caption('● Ready' if ready else '○ Optional')
        c.caption(explain)
    st.caption('A ready indicator checks local configuration or file presence—not successful network service operation or model accuracy.')
    st.divider()
    section('Source model comparison','TF-IDF vs MiniLM','These methods compare text against sources; they do not determine who wrote the text.')
    rows=store.list_analyses()
    if rows:
        recent=store.get_analysis(rows[0]['id'])
        result=recent['result'] if recent else {}
        if result.get('matches'):
            st.dataframe(pd.DataFrame([{'Source':x.get('source_title',''),
                'Category':x.get('category',''),
                'TF-IDF cosine':x.get('lexical_score'),
                'MiniLM cosine':x.get('semantic_score')}
                for x in result['matches'][:75]]),hide_index=True,use_container_width=True)
        else:
            st.info('No eligible source pairs in the newest analysis.')
    with st.expander('Run bundled educational comparison'):
        compare=st.checkbox('Also run MiniLM on the toy example pairs',value=False,key='tl_toy_semantic')
        if st.button('Run toy evaluation',key='tl_run_toy'):
            try:
                from evaluation.evaluate import compare_toy
                with st.spinner('Evaluating bundled examples…'):
                    data=compare_toy(use_semantic=compare)
                st.json({k:v for k,v in data.items() if k not in ('samples','labels','lexical','semantic')})
            except Exception as exc:
                st.error('Toy comparison failed ('+type(exc).__name__+').')
    section('Writing model evidence','Experimental classifier performance',
            'Validation on a single essay corpus cannot establish general-purpose ChatGPT authorship detection.')
    benchmark=ROOT/'datasets/evaluation_reports/v2_external_comparison.json'
    if benchmark.exists():
        import json
        try:
            report=json.loads(benchmark.read_text(encoding='utf-8'))
            table=[]
            for method,rec in (report.get('results') or {}).items():
                stats=rec.get('external_test') or {}
                table.append({'Candidate':method.upper(),'Samples':stats.get('samples'),
                              'Accuracy':stats.get('accuracy'), 'AI recall':stats.get('recall'),
                              'Human FPR':stats.get('false_positive_rate')})
            if table:
                st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
            st.caption('Published train/test splits of the same essay corpus. This is research evaluation, not proof of authorship or accuracy on current ChatGPT outputs.')
        except Exception:
            st.info('Evaluation report is unreadable. Re-run the local model evaluation.')
    else:
        st.info('No v2 benchmark report installed in this project folder. Trained candidate presence and model accuracy are different checks.')


def page_guide():
    hero('Get started', 'A simple, reliable workflow',
         'Run a controlled test first, then explore live source discovery and experimental writing signals.')
    section('Quick start','The first five steps')
    st.markdown('''
1. **Reference Library:** Click **Load all three QA references** or upload your own reference documents.
2. **Analyze:** Upload `qa/ONE_FILE_FEATURE_TEST.txt` (do not index it as a source).
3. **Choose checks:** Enable local comparison; leave web and semantic AI off for a reproducible first test.
4. **Results & Review:** Expect around **26.07% exact matched-word coverage** with only the three controlled references, then inspect matching passages.
5. **Review and export:** Save a reviewer decision and download the evidence PDF.
''')
    with st.container(border=True):
        st.write('**Quick start test document**')
        sample=(ROOT/'qa/ONE_FILE_FEATURE_TEST.txt')
        st.download_button('Download controlled QA test',sample.read_bytes(),file_name=sample.name,
                           mime='text/plain',use_container_width=True)
        if st.button('Load the three controlled references',key='tl_guide_setup',use_container_width=True):
            install_demo_sources()
            navigate('Analyze')
    section('Beyond the demo','Wikipedia + Tavily')
    st.markdown('Search for public sources or supply the **exact article URL** when automatic discovery misses a page. Only accessible retrieved pages can be compared. Consent is required before sending search phrases externally.')
    section('Writing research','V2 AI-writing results')
    st.markdown('Enable **Check AI-writing patterns**, run a NEW analysis after installing a model, and open the **AI-writing review** tab. N-gram v2 is displayed first when available. Scores remain experimental; all consequential judgments require human context.')
    section('Windows','Run TraceLens locally')
    st.code(r'''cd C:\Code\Plagrism\Stage-10\TraceLens-AI
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\self_check.py --with-ai
.\.venv\Scripts\python.exe -m streamlit run app.py''',language='powershell')
    st.caption('Do not share .env, API keys, private student papers, or trusted local model files. Keep a backup before applying updates.')


with st.sidebar:
    st.markdown('<div class="tl-brand"><div class="name">◈ TraceLens AI</div>'
                '<small>Evidence-first document intelligence</small></div>',unsafe_allow_html=True)
    view=st.radio('Workspace',NAV,key='tl_nav',label_visibility='collapsed')
    st.divider()
    s=system_status()
    st.caption('WORKSPACE STATUS')
    st.caption(f'● {s["refs"]} local references')
    st.caption('● V2 n-gram ready' if s['ngram'] else '○ V2 n-gram not installed')
    with st.expander('System connections'):
        st.caption('Wikipedia: '+('configured' if s['wiki'] else 'contact needed'))
        st.caption('Tavily: '+('configured' if s['tavily'] else 'optional'))
        st.caption('MiniLM: '+('installed' if s['semantic'] else 'optional'))
    st.markdown('<div class="tl-foot">v1.9 · frontend rebuild<br>Local-first · Single-user prototype<br>Human review is essential</div>',unsafe_allow_html=True)

if view == 'Analyze': page_analyze()
elif view == 'Reference Library': page_library()
elif view == 'Results & Review': page_results()
elif view == 'Models & Diagnostics': page_models()
else: page_guide()
