from pathlib import Path
import sys, tempfile, csv, json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_v2_training as downloader
from train_v2_candidates import threshold_for_validation_humans, train
import numpy as np


def make_text(i, label):
    if label==0:
        base = 'During the semester my teacher asked us to think about our classroom activity and describe what I experienced. '
    else:
        base = 'Furthermore a systematic framework demonstrates comprehensive analytical methodologies and provides structured insights into pedagogical outcomes. '
    return f'Uniqueintro{i} ' + (base*15) + f' Uniqueending{i}.'


def test_threshold_has_bounded_validation_fpr():
    x=np.linspace(.01,.99,100)
    t=threshold_for_validation_humans(x,.05)
    assert sum(x>=t) <=5


def test_prepare_training_uses_train_split_and_excludes_external_test(tmp_path):
    (tmp_path/'app.py').write_text('')
    d=tmp_path/'datasets'; d.mkdir()
    with (d/'independent_test.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['text','label','group_id']);w.writeheader()
        w.writerow({'text':make_text(1,0),'label':'0','group_id':'external-1'})
        w.writerow({'text':make_text(2,1),'label':'1','group_id':'external-2'})
    for c in ('ai_writing.csv',):
        with (d/c).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=['text','label','group_id']);w.writeheader()
            w.writerow({'text':make_text(3,0),'label':'0','group_id':'hc3-1'})
            w.writerow({'text':make_text(4,1),'label':'1','group_id':'hc3-1'})
    def fetch(path,params):
        assert params['split']=='train'  # no test API calls
        assert path=='/rows'
        offset=params['offset']
        rows=[]
        for i in range(offset,min(offset+100,600)):
            label=i%2
            rows.append({'row_idx':i,'row':{'text':make_text(i,label),'generated':label}})
        return {'num_rows_total':600, 'rows':rows}
    target=downloader.build(tmp_path,per_class=50,max_pages=10,fetch=fetch)
    assert target.exists()
    rows=list(csv.DictReader(target.open(encoding='utf-8-sig')))
    assert len(rows)>=102
    assert make_text(1,0) not in [r['text'] for r in rows if r['group_id'].startswith('ai-human-train-')]
    assert all('test' not in r['group_id'] for r in rows)
    assert json.loads((d/'ai_v2_train_provenance.json').read_text())['test_split_never_downloaded_for_training']


def test_train_ngram_without_touching_old_model(tmp_path):
    f=tmp_path/'training.csv'
    with f.open('w',newline='',encoding='utf8') as out:
        w=csv.DictWriter(out,fieldnames=['text','label','group_id']);w.writeheader()
        for i in range(200):
            w.writerow({'text':make_text(i,0),'label':0,'group_id':f'pair-{i}'})
            w.writerow({'text':make_text(i+200,1),'label':1,'group_id':f'pair-{i}'})
    from train_v2_candidates import train
    result=train(f, methods=('ngram',), output_dir=tmp_path/'models')
    assert (tmp_path/'models'/'ai_ngram_candidate.joblib').exists()
    assert result['ngram']['validation']['human']>=40
    assert result['ngram']['validation']['false_positive_rate']<=.05

if __name__=='__main__':
    raise SystemExit(pytest.main([__file__,'-q']))
