"""SQLite persistence: local-only; no remote database required."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'storage' / 'tracelens.sqlite3'

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('reference','submission')),
                    file_type TEXT, content_hash TEXT NOT NULL,
                    text_content TEXT NOT NULL, pages_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, results_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    analysis_id TEXT NOT NULL, match_id TEXT NOT NULL,
                    decision TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(analysis_id,match_id),
                    FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS writing_reviews (
                    analysis_id TEXT NOT NULL,
                    window_index INTEGER NOT NULL CHECK(window_index >= 0),
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(analysis_id,window_index),
                    FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS candidate_reviews (
                    analysis_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    window_index INTEGER NOT NULL CHECK(window_index >= 0),
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(analysis_id,method,window_index),
                    FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );
            ''')

    @contextmanager
    def connect(self):
        """Provide a transactional connection and always release the SQLite handle.

        sqlite3.Connection.__exit__ commits or rolls back but does NOT close.
        Explicit close is essential for Windows TemporaryDirectory cleanup.
        """
        con = sqlite3.connect(str(self.path), timeout=8)
        try:
            con.row_factory = sqlite3.Row
            con.execute('PRAGMA foreign_keys=ON')
            with con:
                yield con
        finally:
            con.close()

    def add_document(self, item: dict, kind: str = 'reference') -> str:
        content_hash = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
        with self.connect() as con:
            if kind == 'reference':
                previous = con.execute('SELECT id FROM documents WHERE content_hash=? AND kind=?',
                                       (content_hash, kind)).fetchone()
                if previous:
                    return previous['id']
            id_ = uuid.uuid4().hex
            con.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',
                        (id_, item['title'], kind, item['kind'], content_hash,
                         item['text'], json.dumps(item['pages']), now()))
        return id_

    def list_documents(self, kind: str = 'reference') -> list[dict]:
        with self.connect() as con:
            rows = con.execute('SELECT id,title,file_type,created_at,length(text_content) text_length '
                               'FROM documents WHERE kind=? ORDER BY created_at DESC', (kind,)).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, id_: str) -> dict | None:
        with self.connect() as con:
            row = con.execute('SELECT * FROM documents WHERE id=?', (id_,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data['text'] = data.pop('text_content')
        data['pages'] = json.loads(data.pop('pages_json'))
        return data

    def remove_reference(self, id_: str) -> bool:
        with self.connect() as con:
            cur = con.execute("DELETE FROM documents WHERE id=? AND kind='reference'", (id_,))
            return cur.rowcount > 0

    def save_analysis(self, submission_id: str, result: dict) -> str:
        id_ = uuid.uuid4().hex
        with self.connect() as con:
            con.execute('INSERT INTO analyses VALUES (?,?,?,?)',
                        (id_, submission_id, now(), json.dumps(result, ensure_ascii=False)))
        return id_

    def get_analysis(self, id_: str) -> dict | None:
        with self.connect() as con:
            row = con.execute('SELECT * FROM analyses WHERE id=?', (id_,)).fetchone()
        if row is None:
            return None
        return {**dict(row), 'result': json.loads(row['results_json'])}

    def list_analyses(self, limit: int = 40) -> list[dict]:
        with self.connect() as con:
            rows = con.execute('''SELECT a.id, a.created_at, d.title,
                               json_extract(a.results_json, '$.coverage') coverage
                               FROM analyses a JOIN documents d ON d.id=a.document_id
                               ORDER BY a.created_at DESC LIMIT ?''', (limit,)).fetchall()
            return [dict(r) for r in rows]

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete saved findings and the submitted text if no longer referenced."""
        with self.connect() as con:
            row=con.execute('SELECT document_id FROM analyses WHERE id=?',(analysis_id,)).fetchone()
            if not row:
                return False
            con.execute('DELETE FROM analyses WHERE id=?',(analysis_id,))
            remaining=con.execute('SELECT COUNT(*) n FROM analyses WHERE document_id=?',
                                  (row['document_id'],)).fetchone()['n']
            if not remaining:
                con.execute("DELETE FROM documents WHERE id=? AND kind='submission'",
                            (row['document_id'],))
        return True

    def update_review(self, analysis_id: str, match_id: str, decision: str, note: str = ''):
        if decision not in ('Needs investigation','Properly attributed','Not relevant','Unreviewed'):
            raise ValueError('Unknown reviewer decision')
        with self.connect() as con:
            con.execute('''INSERT INTO reviews VALUES (?,?,?,?,?)
                           ON CONFLICT(analysis_id,match_id)
                           DO UPDATE SET decision=excluded.decision,note=excluded.note,
                           updated_at=excluded.updated_at''',
                        (analysis_id, match_id, decision, note[:1000], now()))

    def reviews_for(self, analysis_id: str) -> dict:
        with self.connect() as con:
            rows = con.execute('SELECT match_id,decision,note FROM reviews WHERE analysis_id=?',
                               (analysis_id,)).fetchall()
            return {r['match_id']: {'decision': r['decision'], 'note': r['note']} for r in rows}


    def update_writing_review(self, analysis_id: str, window_index: int,
                              decision: str, note: str = '') -> None:
        """Store a reviewer judgement separately from the immutable ML prediction."""
        allowed = ('Unreviewed', 'Discuss with author', 'AI assistance disclosed',
                   'No concern', 'Insufficient evidence')
        if decision not in allowed:
            raise ValueError('Invalid writing review decision')
        if not isinstance(window_index, int) or window_index < 0:
            raise ValueError('Invalid window index')
        with self.connect() as con:
            record = con.execute('SELECT results_json FROM analyses WHERE id=?',
                                 (analysis_id,)).fetchone()
            if record is None:
                raise ValueError('Analysis does not exist')
            pattern = json.loads(record['results_json']).get('ai_pattern_analysis', {})
            windows = pattern.get('windows') or []
            if pattern.get('status') != 'experimental' or window_index >= len(windows):
                raise ValueError('Window does not exist for this analysis')
            con.execute("""INSERT INTO writing_reviews VALUES (?,?,?,?,?)
                           ON CONFLICT(analysis_id,window_index)
                           DO UPDATE SET decision=excluded.decision,
                           note=excluded.note, updated_at=excluded.updated_at""",
                        (analysis_id, window_index, decision, note[:2000], now()))

    def writing_reviews_for(self, analysis_id: str) -> dict[int, dict]:
        with self.connect() as con:
            rows = con.execute("""SELECT window_index,decision,note,updated_at FROM writing_reviews
                                  WHERE analysis_id=? ORDER BY window_index""",
                               (analysis_id,)).fetchall()
            return {r['window_index']: {'decision': r['decision'], 'note': r['note'],
                                         'updated_at': r['updated_at']} for r in rows}

    def update_candidate_review(self, analysis_id: str, method: str, window_index: int,
                                decision: str, note: str = '') -> None:
        """V2 review is independent of legacy writing and source-matching reviews."""
        allowed = ('Unreviewed', 'Discuss with author', 'AI assistance disclosed',
                   'No concern', 'Insufficient evidence')
        if decision not in allowed:
            raise ValueError('Invalid writing review decision')
        if method not in ('ngram','minilm'):
            raise ValueError('Unknown writing model')
        if not isinstance(window_index, int) or window_index < 0:
            raise ValueError('Invalid window index')
        with self.connect() as con:
            row = con.execute('SELECT results_json FROM analyses WHERE id=?',
                              (analysis_id,)).fetchone()
            if row is None:
                raise ValueError('Analysis does not exist')
            models = (json.loads(row['results_json']).get('ai_comparator_analysis') or {}).get('models') or {}
            candidate = models.get(method) or {}
            if (candidate.get('status') != 'experimental'
                    or window_index >= len(candidate.get('scores') or [])):
                raise ValueError('Candidate window does not exist in saved analysis')
            con.execute("""INSERT INTO candidate_reviews VALUES (?,?,?,?,?,?)
                ON CONFLICT(analysis_id,method,window_index)
                DO UPDATE SET decision=excluded.decision,note=excluded.note,
                              updated_at=excluded.updated_at""",
                (analysis_id, method, window_index, decision, note[:2000], now()))

    def candidate_reviews_for(self, analysis_id: str, method: str | None = None) -> dict:
        with self.connect() as con:
            if method is None:
                rows = con.execute('SELECT method,window_index,decision,note,updated_at FROM '
                                   'candidate_reviews WHERE analysis_id=?', (analysis_id,)).fetchall()
                return {(r['method'],r['window_index']): {'decision':r['decision'],
                          'note':r['note'],'updated_at':r['updated_at']} for r in rows}
            rows = con.execute('SELECT window_index,decision,note,updated_at FROM '
                               'candidate_reviews WHERE analysis_id=? AND method=?',
                               (analysis_id,method)).fetchall()
            return {r['window_index']: {'decision':r['decision'],'note':r['note'],
                    'updated_at':r['updated_at']} for r in rows}
