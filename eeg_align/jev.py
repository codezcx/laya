"""Actual JEV-27B-VL System-1 language backbone, pretrained adapter and decision rows."""
import json
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

CHECKPOINT=Path('laya/JEV-27B-VL')
PREFIX='[kind] choice\n[state] EEG recording begins:\n'
SUFFIX=('\nEEG recording ends.\n[question] Which task and class does this EEG recording represent?\n[options]\n'
        'A) Emotion recognition: negative emotional valence\n'
        'B) Emotion recognition: neutral emotional valence\n'
        'C) Emotion recognition: positive emotional valence\n'
        'D) Motor imagery recognition: imagined left-hand movement\n'
        'E) Motor imagery recognition: imagined right-hand movement\n[decision]:')

class LoadedLoRA(nn.Module):
    def __init__(self,base,a,b,scale):
        super().__init__();self.base=base.requires_grad_(False)
        self.lora_A=nn.Parameter(a.float());self.lora_B=nn.Parameter(b.float());self.scale=scale
    def forward(self,x):
        return self.base(x)+(F.linear(F.linear(x.float(),self.lora_A),self.lora_B)*self.scale).to(x.dtype)

def assemble_embeddings(prefix,eeg,suffix,valid,segment_splits=None,segment_markers=None):
    suffixes = [suffix] * len(eeg) if isinstance(suffix, torch.Tensor) and suffix.ndim == 2 else suffix
    if len(suffixes) != len(eeg):raise ValueError('One suffix required per EEG sample')
    if segment_splits is None:segment_splits=[0]*len(eeg)
    seqs=[]
    for x,m,s,cut in zip(eeg,valid,suffixes,segment_splits):
        z=x[m]
        if cut:
            if segment_markers is None or not 0 < cut < len(z):raise ValueError('Invalid paired segment boundary')
            baseline,task=segment_markers
            seqs.append(torch.cat([prefix,baseline,z[:cut],task,z[cut:],s]))
        else:seqs.append(torch.cat([prefix,z,s]))
    lengths=torch.tensor([len(s) for s in seqs],device=eeg.device)
    inputs=nn.utils.rnn.pad_sequence(seqs,batch_first=True)
    mask=torch.arange(inputs.shape[1],device=eeg.device)[None]<lengths[:,None]
    return inputs,mask,(mask.long().cumsum(-1)-1).clamp_min(0),lengths-1

def load_jev(device='cuda'):
    cfg=AutoConfig.from_pretrained(CHECKPOINT,local_files_only=True).text_config
    cfg.use_cache=False;cfg._attn_implementation='sdpa'
    with init_empty_weights():text=Qwen3_5TextModel(cfg)
    mapping=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
    prefix='model.language_model.'
    selected={k[len(prefix):]:(k,v) for k,v in mapping.items() if k.startswith(prefix)}
    expected=text.state_dict();assert set(selected)==set(expected),(set(expected)-set(selected),set(selected)-set(expected))
    shards=defaultdict(list)
    for key,(source,shard) in selected.items():shards[shard].append((key,source))
    for shard,keys in sorted(shards.items()):
        with safe_open(CHECKPOINT/shard,framework='pt',device='cpu') as f:
            for key,source in keys:
                value=f.get_tensor(source);assert value.shape==expected[key].shape
                dtype=torch.float32 if any(part in key for part in (text._keep_in_fp32_modules or [])) else torch.bfloat16
                set_module_tensor_to_device(text,key,device,value=value,dtype=dtype)
        print('Loaded JEV language shard',shard,flush=True)
    text.to(device=device).requires_grad_(False)
    adapter=CHECKPOINT/'adapter_vllm';ac=json.loads((adapter/'adapter_config.json').read_text())
    assert not ac['use_dora'] and not ac['use_rslora'] and not ac['rank_pattern'] and not ac['alpha_pattern']
    scale=ac['lora_alpha']/ac['r'];head=json.loads((adapter/'decision_head.json').read_text())
    ids=head['verbalizer_ids'][8:13];assert ids==[32,33,34,35,36]
    with safe_open(CHECKPOINT/mapping['lm_head.weight'],framework='pt',device='cpu') as f:
        view=f.get_slice('lm_head.weight');weight=torch.cat([view[i:i+1] for i in ids]).float()
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=set(f.keys());used=set();ap='base_model.model.model.language_model.'
        for name,module in list(text.named_modules()):
            a=ap+name+'.lora_A.weight';b=ap+name+'.lora_B.weight'
            if a not in keys:continue
            assert isinstance(module,nn.Linear) and b in keys
            aa=f.get_tensor(a);bb=f.get_tensor(b)
            assert aa.shape==(ac['r'],module.in_features) and bb.shape==(module.out_features,ac['r'])
            parent,_,child=name.rpartition('.')
            setattr(text.get_submodule(parent),child,LoadedLoRA(module,aa.to(device),bb.to(device),scale))
            used.update([a,b])
        a='base_model.model.lm_head.lora_A.weight';b='base_model.model.lm_head.lora_B.weight'
        weight+=scale*(f.get_tensor(b)[ids].float()@f.get_tensor(a).float());used.update([a,b])
        assert used==keys,keys-used
    assert not any(p.is_meta for p in text.parameters())
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
    return text,weight.to(device),torch.tensor(head['bias'][8:13],device=device),tokenizer

class JEVBridge(nn.Module):
    def __init__(self,text,weight,bias,tokenizer,train_lora=False,train_head=False):
        super().__init__();self.text=text;self.train_lora=train_lora
        self.text.requires_grad_(False)
        for name,p in self.text.named_parameters():
            if name.endswith(('lora_A','lora_B')):p.requires_grad_(train_lora)
        self.connector=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,5120),nn.GELU(),nn.Linear(5120,5120)).to(weight.device)
        if train_head:
            self.head_weight=nn.Parameter(weight.clone());self.head_bias=nn.Parameter(bias.clone())
        else:
            self.register_buffer('head_weight',weight);self.register_buffer('head_bias',bias)
        self.register_buffer('prefix_ids',torch.tensor(tokenizer.encode(PREFIX,add_special_tokens=False),device=weight.device))
        self.register_buffer('suffix_ids',torch.tensor(tokenizer.encode(SUFFIX,add_special_tokens=False),device=weight.device))
        self.register_buffer('baseline_ids',torch.tensor(tokenizer.encode('\nPre-cue fixation baseline [-2.5,-0.5] s:\n',add_special_tokens=False),device=weight.device),persistent=False)
        self.register_buffer('task_ids',torch.tensor(tokenizer.encode('\nMotor imagery task [0.5,2.5] s:\n',add_special_tokens=False),device=weight.device),persistent=False)
        self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def train(self,mode=True):
        super().train(mode)
        # Train mode activates checkpointing even for a frozen decoder: gradients must reach connector.
        self.text.train(mode)
        return self
    def forward(self,tokens,valid,labels=None,suffix_ids=None,num_choices=5,segment_splits=None):
        if valid.dtype!=torch.bool or tokens.shape[:2]!=valid.shape or not valid.any(1).all():raise ValueError('Invalid EEG mask')
        z=self.connector(tokens.masked_fill(~valid[...,None],0).float())
        embed=self.text.get_input_embeddings();prefix=embed(self.prefix_ids)
        suffix=embed(self.suffix_ids) if suffix_ids is None else [embed(s) for s in suffix_ids]
        x,mask,positions,last=assemble_embeddings(prefix,z.to(prefix.dtype),suffix,valid,segment_splits,
            (embed(self.baseline_ids),embed(self.task_ids)))
        hidden=self.text(inputs_embeds=x,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state
        state=hidden[torch.arange(len(hidden),device=hidden.device),last]
        if not 1 <= num_choices <= len(self.head_weight):raise ValueError('Invalid choice count')
        logits=F.linear(state.float(),self.head_weight[:num_choices],self.head_bias[:num_choices])
        return dict(logits=logits,loss=None if labels is None else F.cross_entropy(logits,labels))
