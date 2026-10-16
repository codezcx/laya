"""FACED event metadata readers used by the current raw-signal smoke."""
import csv
from pathlib import Path

def read_table(path):return list(csv.DictReader(Path(path).read_text(encoding='utf-8-sig').splitlines(),delimiter='\t'))


def video_events(rows):
 valid=[]
 for r in rows:
  label=r.get('binary_label')
  if label in ('n/a','',None):continue
  if label not in ['negative','neutral','positive']:raise ValueError(('Unknown valence',label))
  if float(r['duration'])<=0:raise ValueError('Video must have a duration')
  valid.append(r)
 assert len(valid)==28 and len({r['video_index'] for r in valid})==28
 assert {k:sum(r['binary_label']==k for r in valid) for k in ['negative','neutral','positive']}=={'negative':12,'neutral':4,'positive':12}
 return valid
