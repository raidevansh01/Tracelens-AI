#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -f .venv/bin/python ]; then
 echo 'Python environment missing. Follow README.md setup.'
 exit 1
fi
exec .venv/bin/python -m streamlit run app.py
