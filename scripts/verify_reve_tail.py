"""Numerical replay and gradient-scope audit against the real checkpoint."""
import json
from pathlib import Path
import torch
from transformers import AutoModel
from eeg_tasks.tail_model import TailClassifier

torch.set_num_threads(4);torch.manual_seed(42);cfg=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());reve=AutoModel.from_pretrained(cfg['reve_checkpoint'],local_files_only=True,trust_remote_code=True).cuda().eval().requires_grad_(False);bank=AutoModel.from_pretrained(cfg['positions_checkpoint'],local_files_only=True,trust_remote_code=True);names=json.loads(Path('artifacts/emotion_reve60_v5/0.done.json').read_text())['channels'];anchors=bank(['F3','F4','C3','Cz','C4','P3','P4','Oz']).cuda();positions=[bank(names).cuda(),bank(['C3','Cz','C4']).cuda()];tail=TailClassifier(reve,torch.randn(7,4096,device='cuda'),'learned_ce',anchors,positions,True).cuda().eval();capture={};hook=reve.transformer.layers[-2][0].register_forward_pre_hook(lambda m,args:capture.update(x=args[0].detach().clone()));audit={}
for task,c in [(0,60),(1,3)]:
 with torch.no_grad():
  full=reve(torch.randn(1,c,740,device='cuda'),positions[task][None]);prefix=capture['x'];replay=tail.encode_tail(prefix).reshape_as(full);torch.testing.assert_close(replay,full,atol=1e-5,rtol=1e-5);rounded=tail.encode_tail(prefix.half().float()).reshape_as(full);relative=float((rounded-full).norm()/full.norm());assert relative<.005
 audit[str(task)]={'fp32_max_abs':float((replay-full).abs().max()),'fp16_prefix_relative_l2':relative}
hook.remove();tail.train();tail.zero_grad();loss=tail(prefix.detach(),1).square().mean();loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tail.layers.parameters());assert all(p.grad is None for layer in reve.transformer.layers[:-2] for p in layer.parameters());assert any(p.grad is not None for p in tail.head.parameters());audit['gradient_scope']='last two blocks and head only';audit['tail_parameter_count']=sum(p.numel() for p in tail.layers.parameters());Path('artifacts/reve_prefix_v8/replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
