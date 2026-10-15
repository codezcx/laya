"""Fixed-C source-only subject folds; excludes original source test and external data."""
import json,hashlib,warnings
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .source_probe import score


def main():
 root=Path('artifacts/emotion_source_probe_v4');out=root/'subject_cv.json'
 if out.exists():raise FileExistsError(out)
 assert (root/'results.json').exists(),'Feature extraction must complete first'
 rows=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()]
 assert all(r['dataset']=='SEED' and r['split'] in ('train','validation') for r in rows)
 subjects=np.random.default_rng(20261015).permutation(sorted({r['subject'] for r in rows}))
 folds=[set(x.tolist()) for x in np.array_split(subjects,3)];y=np.array([r['class_id'] for r in rows]);results={}
 for arm in ['3ch_relative_30','62ch_relative_30','62ch_absolute_30','62ch_absolute_45']:
  x=np.load(root/f'{arm}.npy');runs=[]
  for fold,held in enumerate(folds):
   vi=np.array([i for i,r in enumerate(rows) if r['subject'] in held]);ti=np.array([i for i,r in enumerate(rows) if r['subject'] not in held]);model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000,random_state=42))
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);model.fit(x[ti],y[ti])
   m=score([rows[i] for i in vi],y[vi],model.predict(x[vi]));m.update(fold=fold,held_subjects=sorted(held),convergence_warnings=[str(w.message) for w in caught]);runs.append(m)
   print(arm,fold,m['macro_dataset_accuracy'],flush=True)
  results[arm]={'folds':runs,'mean':float(np.mean([r['macro_dataset_accuracy'] for r in runs]))}
 out.write_text(json.dumps({'C':.01,'results':results,'rows_sha256':hashlib.sha256((root/'rows.jsonl').read_bytes()).hexdigest(),'note':'Exploratory source-only robustness check; reuses original source train and validation. Not fresh independent test.'},indent=2)+'\n')

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
