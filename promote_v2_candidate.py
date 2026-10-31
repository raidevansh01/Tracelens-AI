"""Manually install a trained v2 candidate into v1.8 comparison UI.

Requires explicit --install and a previously completed external evaluation.
Makes a timestamped backup of the previous candidate; never edits stylometry.
"""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parent


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--method',required=True,choices=['ngram','minilm'])
    a.add_argument('--install',action='store_true',help='Explicitly install into Streamlit comparison interface')
    a.add_argument('--max-fpr',type=float,default=.10)
    args=a.parse_args()
    source=ROOT/'storage'/'v2_experiments'/f'ai_{args.method}_candidate.joblib'
    target=ROOT/'storage'/f'ai_{args.method}_candidate.joblib'
    report=ROOT/'datasets'/'evaluation_reports'/'v2_external_comparison.json'
    if not source.exists() or not report.exists():
        print('Candidate or external evaluation report missing. Train and evaluate first.')
        return 2
    result=json.loads(report.read_text(encoding='utf8'))
    entry=result.get('results',{}).get(args.method)
    if not entry:
        print('No external evaluation found for',args.method)
        return 2
    fpr=entry['external_test']['false_positive_rate']
    recall=entry['external_test']['recall']
    print(args.method,'external FPR',round(fpr,3),'recall',round(recall,3))
    if fpr>args.max_fpr:
        print(f'NOT PROMOTED: measured false-positive rate exceeds {args.max_fpr:.1%}.')
        return 2
    if not args.install:
        print('Dry run only. Add --install to install the candidate after review.')
        return 0
    # Trusted artifact only. Verify schema before installing, then backup existing.
    item=joblib.load(source)
    if item.get('schema')!='tracelens_comparator_1' or item.get('method')!=args.method:
        print('Incompatible model artifact; refusing installation.')
        return 2
    if target.exists():
        backup=target.with_suffix('.before_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.joblib')
        shutil.copy2(target,backup)
        print('BACKUP',backup)
    temporary=target.with_suffix('.installing')
    shutil.copy2(source,temporary)
    temporary.replace(target)
    print('INSTALLED',target)
    print('Restart Streamlit. This remains EXPERIMENTAL and cannot prove authorship.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
