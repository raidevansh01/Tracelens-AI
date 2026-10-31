"""Safe local document ingestion and positional segmentation."""
from __future__ import annotations
import io
import re
from pathlib import Path
import pymupdf as fitz
from docx import Document

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PAGES = 30
MAX_CHARS = 180_000
SUPPORTED = {'.pdf', '.docx', '.txt'}
TOKEN_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", re.UNICODE)
SENTENCE_RE = re.compile(r'[^.!?\n]+(?:[.!?]+|$)', re.UNICODE)

class ExtractionError(ValueError):
    pass

def word_tokens(text: str) -> list[str]:
    return [m.group().casefold() for m in TOKEN_RE.finditer(text)]

def normalize(text: str) -> str:
    return ' '.join(word_tokens(text))

def _limit(text: str) -> str:
    if not text.strip():
        raise ExtractionError('No extractable text. A scanned PDF may need OCR (not included in v1).')
    if len(text) > MAX_CHARS:
        raise ExtractionError(f'Text exceeds the {MAX_CHARS:,}-character v1 limit.')
    return text

def extract_upload(filename: str, data: bytes) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ExtractionError('Unsupported format. Use PDF, DOCX, or TXT.')
    if not data:
        raise ExtractionError('The uploaded document is empty.')
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError('File exceeds the 10 MB limit.')
    pages: list[dict] = []
    try:
        if suffix == '.pdf':
            doc = fitz.open(stream=data, filetype='pdf')
            try:
                if doc.page_count > MAX_PAGES:
                    raise ExtractionError('PDF exceeds the 30-page v1 limit.')
                for idx, page in enumerate(doc):
                    pages.append({'page': idx + 1, 'text': page.get_text('text', sort=True)})
            finally:
                doc.close()
        elif suffix == '.docx':
            document = Document(io.BytesIO(data))
            blocks = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    blocks.append(' | '.join(c.text for c in row.cells))
            pages = [{'page': 1, 'text': '\n\n'.join(blocks)}]
        else:
            try:
                plain = data.decode('utf-8-sig')
            except UnicodeDecodeError as exc:
                raise ExtractionError('TXT file must be UTF-8 encoded.') from exc
            pages = [{'page': 1, 'text': plain}]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f'Could not read the document: {type(exc).__name__}') from exc
    text, page_ranges = '', []
    for page in pages:
        if text:
            text += '\n\n'
        start = len(text)
        text += page['text']
        page_ranges.append({'page': page['page'], 'start': start, 'end': len(text)})
    return {'title': Path(filename).name, 'text': _limit(text), 'pages': page_ranges, 'kind': suffix[1:]}

def extract_paste(text: str, title: str = 'Pasted text') -> dict:
    return {'title': title, 'text': _limit(text), 'pages': [{'page': 1, 'start': 0, 'end': len(text)}], 'kind': 'txt'}

def segment(text: str, pages: list[dict] | None = None, min_words: int = 5,
            max_passages: int = 350) -> list[dict]:
    """Segment text to offset-addressable sentences and 2-sentence windows.

    Offsets always index original text. Windows never span document pages.
    """
    if pages is None:
        pages = [{'page': 1, 'start': 0, 'end': len(text)}]
    result = []
    for pr in pages:
        chunk = text[pr['start']:pr['end']]
        single = []
        for m in SENTENCE_RE.finditer(chunk):
            raw = m.group()
            if not raw.strip():
                continue
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start = pr['start'] + m.start() + leading
            end = pr['start'] + m.start() + trailing
            if end <= start:
                continue
            if len(word_tokens(text[start:end])) >= min_words:
                single.append({'text': text[start:end], 'start': start,
                               'end': end, 'page': pr['page'], 'window': 1})
        for i, p in enumerate(single):
            result.append(p)
            # Do not create a window that crosses a double-newline paragraph break.
            if i + 1 < len(single) and '\n\n' not in text[p['end']:single[i+1]['start']]:
                q = single[i + 1]
                if len(word_tokens(text[p['start']:q['end']])) <= 110:
                    result.append({'text': text[p['start']:q['end']],
                                   'start': p['start'], 'end': q['end'],
                                   'page': pr['page'], 'window': 2})
            if len(result) >= max_passages:
                return result[:max_passages]
    return result
