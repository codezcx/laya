"""Matched source-split frozen REVE mean-feature logistic regression control."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from eeg_align.tuab_data import ROOT,readrows,selected_windows,metrics,write_json
OUT=Path('artifacts/tuab_jev/probe')


def features(split):
    rows=readrows(ROOT/f'records_{split}.jsonl')
    x=np.stack([np.load(ROOT/'tokens'/f"{r['id']}.npy")[selected_windows(r,0)].astype(np.float32).mean((0,1)) for r in rows])
    return x,np.array([r['label'] for r in rows]),rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.test:
        model=joblib.load(OUT/'model.joblib');x,y,rows=features('test');prob=model.predict_proba(x)[:,1]
        write_json(OUT/'test.json',dict(metrics=metrics(y,prob),predictions=[dict(id=r['id'],label=int(v),probability_abnormal=float(q)) for r,v,q in zip(rows,y,prob)]));return
    OUT.mkdir(exist_ok=False)
    x,y,_=features('train');v,vy,rows=features('validation');best=-1;results=[]
    for c in [.01,.1,1,10]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=42))
        model.fit(x,y);prob=model.predict_proba(v)[:,1];result=metrics(vy,prob);results.append(dict(C=c,metrics=result))
        if result['balanced_accuracy']>best:
            best=result['balanced_accuracy'];joblib.dump(model,OUT/'model.joblib')
            write_json(OUT/'validation.json',dict(C=c,metrics=result,predictions=[dict(id=r['id'],label=int(t),probability_abnormal=float(q)) for r,t,q in zip(rows,vy,prob)]))
    write_json(OUT/'selection.json',dict(candidates=results,selection='validation BA, earliest on ties',features='mean over EEG tokens and 8 windows; train epoch-0 selection; scaler fit only on train'))


if __name__=='__main__':main()
