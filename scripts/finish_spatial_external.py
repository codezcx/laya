"""Wait for existing extraction workers and evaluate frozen v6 models once."""
import json,subprocess,time
from pathlib import Path
root=Path('artifacts/spatial_prototypes_v6');jobs=json.loads((root/'external_jobs.json').read_text());deadline=time.monotonic()+3600
while not all((root/'external_seed_vii'/f'{j["shard"]}.done.json').exists() for j in jobs):
 for j in jobs:
  if (root/'external_seed_vii'/f'{j["shard"]}.done.json').exists():continue
  p=Path(f'/proc/{j["pid"]}/stat')
  if not p.exists() or p.read_text().split()[2]=='Z':raise RuntimeError(f'Extractor failed: {j}; inspect logs, do not restart blindly')
 if time.monotonic()>deadline:raise TimeoutError('Extraction wait exceeded one hour')
 time.sleep(5)
subprocess.run(['laya/.venv-eeg/bin/python','-m','eeg_tasks.spatial_external','--evaluate'],check=True)
