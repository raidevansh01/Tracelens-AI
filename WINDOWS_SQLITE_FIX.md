# TraceLens AI v1.8.1 — Windows SQLite/PDF QA hotfix

## Confirmed symptom

On Windows, `python qa/self_check.py` fails with `[WinError 32] ... Temp\...\qa.sqlite3` in the `SQLite / PDF report workflow` check after successful SQLite and PDF operations.

## Cause and correction

`db/store.py` used `with self.connect() as con` where `connect()` returned a raw `sqlite3.Connection`. In Python, a SQLite connection's context manager handles transaction commit/rollback **but does not close the connection**. The resulting leaked handle prevents Windows from removing the temporary database.

The patch changes `Store.connect()` to a `@contextmanager`: it yields the connection inside a transaction and **always closes the connection in `finally`**. It also updates the PDF QA script to import the supported `pymupdf` module instead of deprecated `fitz`, and supplies three regression tests.

## Apply without losing project data

1. **Close Streamlit** (Ctrl+C in its terminal) and any other processes using the project.
2. Extract `TraceLens-AI-v1.8.1-WINDOWS-SQLITE-HOTFIX.zip` into a directory, e.g. your Downloads folder.
3. In PowerShell, run the following from the extracted patch folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\APPLY_WINDOWS_PATCH.ps1 -ProjectDir 'C:\Code\Plagrism\Stage-10\TraceLens-AI'
```

The installer backs up the existing `db/store.py` and `qa/self_check.py` in a `hotfix_backup_<timestamp>` directory inside your Stage-10 project. It copies the new `tests/test_sqlite_lifecycle.py` file. **No `.env`, trained model, dataset, reference document, or SQLite database is modified or replaced.**

Manual alternative: copy only `db/store.py`, `qa/self_check.py`, and `tests/test_sqlite_lifecycle.py` from the patch ZIP into the matching Stage-10 folders, preserving folder names.

## Verify

```powershell
cd C:\Code\Plagrism\Stage-10\TraceLens-AI
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\self_check.py
.\.venv\Scripts\python.exe qa\self_check.py --with-ai
```

Expected from the patched v1.8 codebase: **55 pytest tests**, **20/20 standard offline QA**, and **22/22 with MiniLM enabled** on a computer where MiniLM loads successfully. Optional live provider and browser checks remain separate. The patch does not improve the previously documented AI-authorship false negative.

## Verification here

- `python -m pytest -q`: 55 passed.
- `python qa/self_check.py`: 20 passed, 0 failed; PDF generated without the old import warning.
- `python qa/self_check.py --with-ai`: local container lacked MiniLM model/runtime; the isolated MiniLM check failed here, while the SQLite/PDF workflow passed. Your Windows output shows MiniLM works there.

If Windows still reports a lock, share the exact failing traceback and whether `qa.sqlite3` is held by antivirus, a second Python process, or another open connection. Do not delete your production `storage/tracelens.sqlite3`.
