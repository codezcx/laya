import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from eeg_align.cache import digest
root=Path('laya/JEV-27B-VL');mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
files=[root/n for n in sorted(set(mapping.values()))]+[root/n for n in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','adapter_vllm/adapter_model.safetensors','adapter_vllm/adapter_config.json','adapter_vllm/decision_head.json']]
with ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(zip(map(str,files),pool.map(digest,files)))
Path('artifacts/jev_eeg/checkpoint_audit.json').write_text(json.dumps(dict(files=hashes,checkpoint=str(root.resolve()),adapter_always_active=True),indent=2))
print('checkpoint audit complete',flush=True)
