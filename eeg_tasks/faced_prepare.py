"""Verify one FACED BDF and extract fixed trial windows with explicit labels."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import mne
from .data import preprocess,normalized_window,subject_split


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


def prepare_window(window,names,quarantine_flat=False):
 if not np.isfinite(window).all():raise ValueError('Nonfinite EEG window')
 std=window.std(-1);bad=np.flatnonzero(std<1e-12)
 if len(bad) and quarantine_flat:
  return None,{'reason':'flat_channel_std_below_1e-12_volts','channels':[names[i] for i in bad],'std_volts':[float(std[i]) for i in bad]}
 return normalized_window(window),None


def main(subject, root=Path('artifacts/faced_source_v14'), quarantine_flat=False):
 root=Path(root);folder=root/'raw'/f'sub-{subject:03d}';prefix=f'sub-{subject:03d}_task-watchingVideoClips';done=json.loads((folder/'download.done.json').read_text());path=folder/f'{prefix}_eeg.bdf'
 assert path.stat().st_size==done['size']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==done['sha256']
 side=json.loads((folder/f'{prefix}_eeg.json').read_text());table=read_table(folder/f'{prefix}_channels.tsv');events=video_events(read_table(folder/f'{prefix}_events.tsv'));raw=mne.io.read_raw_bdf(path,preload=False,verbose=False);fs=raw.info['sfreq'];assert fs==side['SamplingFrequency'];assert abs(raw.n_times/fs-side['RecordingDuration'])<1
 eeg_names=[r['name'] for r in table if r['type']=='EEG'];assert len(eeg_names)==side['EEGChannelCount'];picks=[raw.ch_names.index(c) for c in eeg_names]
 cfg={'bandpass':[1,30]};signals=[];rows=[];split=subject_split(range(123),1401)[subject];statistics=[];quarantined=[]
 for e in events:
  begin=round(float(e['onset'])*fs);end=round((float(e['onset'])+float(e['duration']))*fs);assert 0<=begin<end<=raw.n_times
  x=raw.get_data(picks=picks,start=begin,stop=end);assert np.isfinite(x).all();statistics.append({'video':e['video_index'],'raw_std_volts_median':float(np.median(x.std(-1))),'raw_max_abs_volts':float(abs(x).max())});x=preprocess(x,fs,cfg);n=x.shape[-1]//600;assert n>=10
  for s in np.linspace(0,n-1,10,dtype=int)*600:
   meta={'dataset':'FACED-NEMAR','subject':str(subject),'split':split,'task':0,'class_id':{'negative':0,'neutral':1,'positive':2}[e['binary_label']],'path':str(path),'trial':str(e['video_index']),'start_sample':int(s),'event_onset_seconds':float(e['onset']),'label_source':'stimulus-assigned binary_label; not self-report'}
   signal,rejection=prepare_window(x[:,s:s+600],eeg_names,quarantine_flat)
   if rejection is not None:quarantined.append(dict(meta,**rejection));continue
   signals.append(signal);rows.append(dict(meta,signal_sha256=hashlib.sha256(signal.tobytes()).hexdigest()))
 assert len(signals)+len(quarantined)==280 and signals
 assert {r['class_id'] for r in rows}=={0,1,2}
 out=root/'prepared'/f'sub-{subject:03d}';out.mkdir(parents=True,exist_ok=False);np.save(out/'signals.npy',np.stack(signals));(out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));audit={'subject':subject,'split':split,'sampling_rate':fs,'n_times':int(raw.n_times),'channels':eeg_names,'mne_original_units':raw._orig_units,'sidecar_units':{r['name']:r['units'] for r in table},'event_count':len(events),'windows':len(signals),'candidate_windows':280,'quarantined_windows':quarantined,'quality_rule':'strict' if not quarantine_flat else 'quarantine flat windows; no replacement','trial_signal_statistics':statistics,'download_sha256':done['sha256'],'source_revision':done['revision']};(out/'audit.json').write_text(json.dumps(audit,indent=2));print({k:v for k,v in audit.items() if k not in ['trial_signal_statistics','sidecar_units','mne_original_units']},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=0);main(p.parse_args().subject)
