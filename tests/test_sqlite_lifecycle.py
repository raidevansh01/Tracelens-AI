"""Guard against Windows [WinError 32] cleanup regression."""
from pathlib import Path
import sqlite3
import tempfile

import pytest

from db.store import Store


def test_sqlite_connection_is_actually_closed(tmp_path):
    store = Store(tmp_path / 'connections.sqlite3')
    with store.connect() as con:
        assert con.execute('SELECT 1').fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        con.execute('SELECT 1')


def test_sqlite_transaction_rollback_and_close(tmp_path):
    store = Store(tmp_path / 'rollback.sqlite3')
    with pytest.raises(RuntimeError, match='deliberate abort'):
        with store.connect() as con:
            con.execute("INSERT INTO documents VALUES ('1','x','reference','txt','hash','text','[]','now')")
            raise RuntimeError('deliberate abort')
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        con.execute('SELECT 1')
    with store.connect() as verification:
        assert verification.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_sqlite_file_can_be_removed_after_workflow():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'qa.sqlite3'
        store = Store(path)
        with store.connect() as con:
            assert con.execute('SELECT 1').fetchone()[0] == 1
        path.unlink()  # This would raise Windows PermissionError if handle remained open.
        assert not path.exists()
