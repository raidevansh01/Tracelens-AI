"""Inspect a known ChatGPT-written .txt with both new candidate models (research only)."""
from pathlib import Path
import argparse
from core.ai_comparator import inspect_candidates


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--file',default='TEST_AI_GENERATED.txt')
    args=p.parse_args()
    text=Path(args.file).read_text(encoding='utf-8-sig')
    root=Path(__file__).resolve().parent
    paths={m:root/'storage'/'v2_experiments'/f'ai_{m}_candidate.joblib' for m in ('ngram','minilm')}
    result=inspect_candidates(text,paths=paths)
    print('Document words:',len(text.split()), '| status:',result['status'])
    for name, value in result.get('models',{}).items():
        print(name,'status:',value['status'])
        if value.get('status')=='experimental':
            print('  threshold:',value['threshold'],'| window scores:',value['scores'],
                  '| review flags:',value['flags'])
    print('NOTE: A model flag is not proof that ChatGPT authored the document.')

if __name__=='__main__':
    main()
