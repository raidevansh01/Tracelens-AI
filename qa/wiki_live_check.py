"""Manual live Wikimedia API smoke test; requires internet and operator contact.
Run from project root using the virtual environment Python interpreter.
"""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from core.wikipedia import wikipedia_url_source, WikipediaError

print('Wikimedia public API smoke test (no API key needed)')
try:
    src = wikipedia_url_source('https://en.wikipedia.org/wiki/Machine_learning')
    print('PASS: article fetched: ', src['title'])
    print('URL: ', src['url'])
    print('Characters extracted: ', len(src['text']))
    print('Text sample: ', src['text'][:120].replace('\n', ' '))
except WikipediaError as exc:
    print('FAIL: ', exc)
    sys.exit(1)
