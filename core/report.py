"""A compact ReportLab PDF with honest analysis provenance and limitations."""
from __future__ import annotations
import io
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

NAVY=colors.HexColor('#16324a')
INK=colors.HexColor('#233044')
MUTED=colors.HexColor('#5d6b79')

def _safe(s, limit=1200):
    return escape(str('' if s is None else s)[:limit]).replace('\n','<br/>')

def make_pdf(result: dict, reviews: dict | None=None, writing_reviews: dict | None=None,
             candidate_reviews: dict | None=None) -> bytes:
    reviews=reviews or {}
    writing_reviews=writing_reviews or {}
    candidate_reviews=candidate_reviews or {}
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=18*mm,bottomMargin=18*mm,title='TraceLens AI - Evidence Report')
    ss=getSampleStyleSheet()
    ss.add(ParagraphStyle(name='TitleTL',parent=ss['Title'],fontName='Helvetica-Bold',
                          textColor=NAVY,fontSize=19,leading=24,spaceAfter=10,alignment=TA_LEFT))
    ss.add(ParagraphStyle(name='BodyTL',parent=ss['BodyText'],fontSize=9.5,leading=14,
                          textColor=INK,spaceAfter=7,splitLongWords=True))
    ss.add(ParagraphStyle(name='SmallTL',parent=ss['BodyText'],fontSize=8.2,leading=11,
                          textColor=MUTED,spaceAfter=6,splitLongWords=True))
    ss.add(ParagraphStyle(name='HeadingTL',parent=ss['Heading2'],fontSize=11.5,
                          textColor=NAVY,spaceBefore=12,spaceAfter=6))
    reference_count=int(result.get('references_checked') or 0)
    coverage_display=(f'{float(result.get("coverage",0)):.2f}%' if reference_count else 'N/A - no reference sources checked')
    out=[Paragraph('TraceLens AI | Evidence report',ss['TitleTL']),
         Paragraph('Document: '+_safe(result.get('title')),ss['BodyTL']),
         Paragraph('Source similarity | Exact matched-word coverage: <b>'+_safe(coverage_display)+
                   '</b> (not a plagiarism or AI-authorship percentage)',ss['BodyTL'])]
    if not reference_count:
        out.append(Paragraph('SOURCE COMPARISON NOT ASSESSED: no local reference or retrieved webpage was compared. '+
                             'No conclusion about originality is possible.', ss['BodyTL']))
    rows=[['Words checked','References compared','Displayed findings'],
          [str(result.get('word_count',0)),str(result.get('references_checked',0)),
           str(len(result.get('matches',[])))]]
    table=Table(rows,colWidths=[55*mm,55*mm,55*mm],hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),
                               ('TEXTCOLOR',(0,0),(-1,0),NAVY),
                               ('BOX',(0,0),(-1,-1),.5,colors.HexColor('#d3dde6')),
                               ('INNERGRID',(0,0),(-1,-1),.3,colors.HexColor('#d3dde6')),
                               ('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),8),
                               ('TOPPADDING',(0,0),(-1,-1),8)]))
    out += [table,Spacer(1,5*mm),Paragraph('Methods and limitations',ss['HeadingTL']),
            Paragraph(_safe(result.get('method')),ss['BodyTL']),
            Paragraph(_safe(result.get('interpretation')),ss['SmallTL'])]
    provider_status = result.get('provider_status') or {}
    if provider_status:
        out.append(Paragraph('Public search provider status',ss['HeadingTL']))
        for name, info in provider_status.items():
            out.append(Paragraph(_safe(name.title()) + ': '+ _safe(info.get('state','unknown')) +
                                 ', full-text sources retrieved: '+str(info.get('retrieved',0)),ss['SmallTL']))
    if result.get('online_requested'):
        out.append(Paragraph('Web comparison covers only the accessible pages retrieved during this run; unavailable or unindexed sources were not checked.',ss['SmallTL']))
    for warning in result.get('warnings',[]):
        out.append(Paragraph('Note: '+_safe(warning),ss['SmallTL']))
    evaluation=(result.get('ai_pattern_analysis') or {}).get('evaluation') or {}
    if evaluation:
        out.append(Paragraph('Experimental AI-writing classifier, same-corpus held-out results: '
                             'precision='+_safe(evaluation.get('precision'))+
                             ', recall='+_safe(evaluation.get('recall'))+
                             ', false-positive rate='+_safe(evaluation.get('false_positive_rate'))+
                             '. These results may not generalize to other writing styles or generators.',ss['SmallTL']))
    out.append(Paragraph('Passage-level evidence',ss['HeadingTL']))
    if not result.get('matches'):
        detail=('Not assessed: no reference sources were available.' if not reference_count else
                'No source matches passed the configured thresholds in the searched collection. This does not establish originality.')
        out.append(Paragraph(detail,ss['BodyTL']))
    for i,m in enumerate(result.get('matches',[])[:65],1):
        review=reviews.get(m['id'],{})
        out.extend([
            HRFlowable(width='100%',thickness=.5,color=colors.HexColor('#d7e0e8')),
            Spacer(1,2*mm),
            Paragraph(f'<b>{i}. {_safe(m["category"])}</b> | Page {m["submitted_page"]}',ss['BodyTL']),
            Paragraph('Source: '+_safe(m['source_title'],200),ss['SmallTL']),
        ])
        if m.get('source_url'):
            out.append(Paragraph('URL: '+_safe(m['source_url'],350),ss['SmallTL']))
        out.extend([
            Paragraph('Submitted: '+_safe(m['submitted_text'],550),ss['BodyTL']),
            Paragraph('Reference: '+_safe(m['source_text'],550),ss['BodyTL']),
            Paragraph((f'Verified exact word sequence: {m["exact_word_count"]} words' if m.get('exact_word_count') else
                       f'Lexical cosine: {m["lexical_score"]:.3f}')+' | '+
                      (f'Semantic cosine: {m["semantic_score"]:.3f}' if m.get('semantic_score') is not None else 'Semantic: not run'),ss['SmallTL']),
            Paragraph(_safe(m['explanation'],500),ss['SmallTL']),
            Paragraph('Reviewer: '+_safe(review.get('decision','Unreviewed'))+
                      ((' | Note: '+_safe(review.get('note'),160)) if review.get('note') else ''),ss['SmallTL'])
        ])
    pattern=result.get('ai_pattern_analysis',{})
    out.append(Paragraph('Independent AI-writing pattern experiment',ss['HeadingTL']))
    if pattern.get('status')=='experimental':
        windows=pattern.get('windows',[])
        nflag=sum(bool(w.get('review_flag')) for w in windows)
        out.append(Paragraph(
            f'{nflag} of {len(windows)} writing windows exceeded the experimental review threshold '
            f'{float(pattern.get("threshold",0)):.3f}. No flags does not prove human authorship; '
            'flags do not prove AI authorship.',ss['BodyTL']))
        if windows:
            pattern_rows=[['Window', 'Model score', 'Model signal', 'Human review']]
            for i,w in enumerate(windows[:24],1):
                review=writing_reviews.get(i-1, {})
                pattern_rows.append([str(i), f'{float(w.get("score",0)):.4f}',
                                     w.get('outcome','Review suggested' if w.get('review_flag') else 'Not flagged'),
                                     review.get('decision', 'Unreviewed')])
            if len(windows)>24:
                pattern_rows.append(['...','...','...',f'{len(windows)-24} more windows'])
            wt=Table(pattern_rows, colWidths=[18*mm,29*mm,47*mm,71*mm],repeatRows=1,hAlign='LEFT')
            wt.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),
                ('BOX',(0,0),(-1,-1),.5,colors.HexColor('#d3dde6')),
                ('INNERGRID',(0,0),(-1,-1),.3,colors.HexColor('#d3dde6')),
                ('FONTSIZE',(0,0),(-1,-1),8.7),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ]))
            out.append(wt)
        reviewed=[(i,r) for i,r in sorted(writing_reviews.items()) if r.get('note') or r.get('decision') != 'Unreviewed']
        for i,review in reviewed[:30]:
            out.append(Paragraph(f'Window {int(i)+1} reviewer: '+_safe(review.get('decision','Unreviewed'))+
                                 ((' | Note: '+_safe(review.get('note'),350)) if review.get('note') else ''),
                                 ss['SmallTL']))
    elif pattern.get('status') in ('unavailable','insufficient_text'):
        out.append(Paragraph(_safe(pattern.get('message')),ss['BodyTL']))
    else:
        out.append(Paragraph('Not selected for this analysis.',ss['BodyTL']))
    comparator=result.get('ai_comparator_analysis') or {}
    if comparator.get('models'):
        out.append(Paragraph('Optional AI-writing model comparisons (experimental)',ss['HeadingTL']))
        for name,info in comparator['models'].items():
            if info.get('status')=='experimental':
                scores=', '.join(str(v) for v in info.get('scores',[])[:10])
                out.append(Paragraph(_safe(name.upper())+': review threshold '+_safe(info.get('threshold'))+
                                     ', scores '+_safe(scores,180)+
                                     '. Not an AI-authorship determination.', ss['SmallTL']))
            else:
                out.append(Paragraph(_safe(name.upper())+': unavailable for this analysis.',ss['SmallTL']))
    if candidate_reviews:
        out.append(Paragraph('V2 model: human-reviewed writing passages',ss['HeadingTL']))
        out.append(Paragraph('These decisions are independent from the legacy model and do not alter scores.',ss['SmallTL']))
        for (model,index),review in sorted(candidate_reviews.items()):
            out.append(Paragraph(_safe(model.upper()) + ' · window ' + str(index+1) +
                ' · reviewer: ' + _safe(review.get('decision','Unreviewed')) +
                (' · ' + _safe(review.get('note'),280) if review.get('note') else ''),ss['SmallTL']))
    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18*mm,11*mm,'TraceLens AI - Evidence only; human assessment required')
        canvas.drawRightString(192*mm,11*mm,f'Page {doc_.page}')
        canvas.restoreState()
    doc.build(out,onFirstPage=footer,onLaterPages=footer)
    return buf.getvalue()
