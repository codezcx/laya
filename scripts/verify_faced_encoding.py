"""Check real FACED input through the exact frozen REVE/spatial interface."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from eeg_tasks.spatial_model import spatial_pool
root=Path('artifacts/faced_source_v14');folder=root/'prepared/sub-000';audit=json.loads((folder/'audit.json').read_text());base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());cfg=json.loads(Path('configs/spatial_prototypes_v6.json').read_text());torch.set_num_threads(4)
with torch.no_grad():
 model,info=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True,output_loading_info=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model=model.cuda().eval();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);positions=bank(audit['channels']).cuda();anchors=bank(cfg['anchors']).cuda();assert len(positions)==32;x=torch.from_numpy(np.load(folder/'signals.npy')[:2]).cuda();tail=(-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap);z=model(torch.nn.functional.pad(x,(0,tail),mode='replicate'),positions[None].expand(len(x),-1,-1));assert z.shape==(2,32,4,512) and torch.isfinite(z).all();p=spatial_pool(z.mean(2),positions,anchors,cfg['sigma']);assert p.shape==(2,8,512) and torch.isfinite(p).all()
 report={'subject':0,'input_shape':list(x.shape),'encoder_shape':list(z.shape),'pooled_shape':list(p.shape),'channels':audit['channels'],'positions_supported':True,'finite':True,'base_checkpoint':base['reve_checkpoint'],'source_sha256':audit['download_sha256'],'note':'Shape/loading check only, no task predictions or model fitting.'};(root/'encoder_smoke.json').write_text(json.dumps(report,indent=2));print(report)
