# TraceLens AI v1.8 verification (current execution environment)

- Automated regression suite: **52 passed** (`python -m pytest -q`). Five new tests added for long Wikipedia-style references, altered punctuation/case, diversified search queries, optional n-gram candidate training+inference, and safe fallback when candidates aren't trained.
- Offline QA: **20 passed, 0 failed**, plus untested/live dependencies (`python qa/self_check.py`).
- Stress check: 133,289-character identical submission/reference produced **100% exact aligned-word coverage in ~0.6 seconds** on this environment. This is a controlled self-comparison, not an external recall benchmark.
- Controlled source fixture: **26.07% aligned-word coverage**, matched against the three known references. Earlier versions displayed six cards; v1.8 can show more thanks to independently retrieved exact runs.
- PDF report: generated and rendered to PNG for visual inspection.
- `train_ai_comparator.py` n-gram candidate tested with a small **synthetic fixture only**; this verifies operation, not realistic AI-authorship accuracy. MiniLM candidate inference not executed here.
- **Not run here:** live Tavily search, live Wikipedia direct/automatic source rediscovery, Streamlit UI, independent real-world classifier benchmark, and Windows integration. These require user network access, configuration and actual samples. They are not claimed to pass.

## Release boundaries

The program can now match long exact copied sequences *if the source is available*. It cannot guarantee discovering every public source. The seven-feature classifier's established false negative remains a model limitation. The new candidates remain experimental and should be retained only if they improve independent evaluation without excessive false positives.
