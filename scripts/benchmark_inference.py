"""Warm batch-1 latency of LEAF -> JEV, small-LM control, and native LEAF.

No training, quantization, LoRA merging, compilation, or model source changes.
Long inputs repeat a cached MI trial for shape/latency testing, not accuracy.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from eeg_align.benchmark_data import catalog, segment
from eeg_align.leaf_data import LEAFSignalAdapter
from eeg_align.multitask_data import write_json
from eeg_align.personalization import (CONFIG, DATA, ResidualAdapter,
    configure_personal, copy_trainable, load_encoder, load_parent)
from eeg_align.run_personalization import load_server, suffix_for
from eeg_align.runtime import digest


def load_small_language(path):
    """Read only the built-in Qwen3-VL language weights, not ECG/custom code.

    This is the local ECG-R1 derivative, not untouched Qwen or an EEG-trained
    JEV. Native A-P vocabulary rows supply a speed-only choice readout.
    """
    from collections import defaultdict
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    path = Path(path)
    cfg = Qwen3VLTextConfig(**json.loads((path/'config.json').read_text())['text_config'])
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'
    with init_empty_weights():
        text = Qwen3VLTextModel(cfg)
    mapping = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.'
    selected = {k.removeprefix(prefix):(k,v) for k,v in mapping.items() if k.startswith(prefix)}
    expected = text.state_dict()
    if set(selected) != set(expected):
        raise ValueError('Local small model does not exactly match built-in Qwen3VLTextModel')
    shards = defaultdict(list)
    for name,(source,shard) in selected.items():
        shards[shard].append((name,source))
    for shard,keys in sorted(shards.items()):
        with safe_open(path/shard, framework='pt', device='cpu') as f:
            for name,source in keys:
                value = f.get_tensor(source)
                if value.shape != expected[name].shape:
                    raise ValueError(f'Wrong shape: {name}')
                dtype = (torch.float32 if any(part in name for part in (text._keep_in_fp32_modules or []))
                         else torch.bfloat16)
                set_module_tensor_to_device(text,name,'cuda',value=value,dtype=dtype)
        print('Loaded small Qwen language shard',shard,flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    ids = [tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(16)]
    if any(len(i)!=1 for i in ids):
        raise ValueError('Choice verbalizers must each be a single token')
    with safe_open(path/mapping['lm_head.weight'],framework='pt',device='cpu') as f:
        head = f.get_slice('lm_head.weight')
        weight = torch.cat([head[i[0]:i[0]+1] for i in ids]).float().cuda()
    if any(p.is_meta for p in text.parameters()):
        raise ValueError('Unloaded small-model parameter')
    return text.to(device='cuda').eval().requires_grad_(False), weight, tokenizer


class SpeedOnlyBridge(torch.nn.Module):
    """Same decision forward, dimension-matched RANDOM connector, no EEG claim."""
    from eeg_align.jev import JEVBridge as _Reference
    forward = _Reference.forward

    def __init__(self,text,weight,tokenizer):
        super().__init__()
        from eeg_align.jev import PREFIX, SUFFIX
        self.text = text
        width = text.config.hidden_size
        self.connector = torch.nn.Sequential(torch.nn.LayerNorm(256),torch.nn.Linear(256,width),
            torch.nn.GELU(),torch.nn.Linear(width,width)).to(weight.device)
        self.register_buffer('head_weight',weight)
        self.register_buffer('head_bias',torch.zeros(len(weight),device=weight.device))
        self.register_buffer('prefix_ids',torch.tensor(tokenizer.encode(PREFIX,add_special_tokens=False),device=weight.device))
        self.register_buffer('suffix_ids',torch.tensor(tokenizer.encode(SUFFIX,add_special_tokens=False),device=weight.device))
        self.requires_grad_(False).eval()


def duration_input(trial, seconds, hz=200):
    samples = round(seconds * hz)
    if seconds < .5 or not math.isclose(samples / hz, seconds):
        raise ValueError('Duration must be >=0.5s and exactly representable at 200 Hz')
    return np.ascontiguousarray(np.tile(trial, (1, 1, math.ceil(samples / trial.shape[-1])))[:, :, :samples])


def preprocess(raw, spatial):
    """In-memory form of LEAFSignalAdapter.batch for this 200 Hz unipolar case.

    Equivalence to the production adapter is checked before timing. Input is
    already filtered/resampled by the dataset cache; EDF loading is excluded.
    """
    limits = np.percentile(raw, [.1, 25, 50, 75, 99.9], axis=(1, 2))
    scale = limits[3] - limits[1]
    fallback = scale <= 0
    scale[fallback] = raw[fallback].std(axis=(1, 2))
    if (scale <= 0).any() or not np.isfinite(raw).all():
        raise ValueError('Nonfinite or constant input')
    x = (np.clip(raw, limits[0, :, None, None], limits[4, :, None, None])
         - limits[2, :, None, None]) / scale[:, None, None]
    x = np.einsum('ec,bct->bet', spatial, x).astype(np.float32)
    return [np.ascontiguousarray(np.pad(v, ((0, 0), (0, 0), (0, (-v.shape[-1]) % 100))))
            for v in segment(x)]


def summary(samples):
    values = np.asarray(samples, dtype=float)
    return dict(n=len(values), mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(values.min()),
                max_ms=float(values.max()), std_ms=float(values.std()), samples_ms=values.tolist())


def measure(fn, warmup, repeats, cuda=True):
    for _ in range(warmup):
        value = fn()
        del value
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall, device = [], []
    start = torch.cuda.Event(enable_timing=True) if cuda else None
    end = torch.cuda.Event(enable_timing=True) if cuda else None
    for _ in range(repeats):
        torch.cuda.synchronize()
        if cuda:
            start.record()
        before = time.perf_counter_ns()
        value = fn()
        if cuda:
            end.record()
            torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - before) / 1e6)
        if cuda:
            device.append(start.elapsed_time(end))
        del value
    return dict(wall=summary(wall), cuda_stream=summary(device) if cuda else None,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def gpu_status():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()


def load_latency_fixture():
    """Same cached real 4-second trial and preprocessing in every speed arm."""
    tasks = catalog()
    tid = next(i for i,t in enumerate(tasks) if t['task_id'] == 'bcic_2a')
    path = Path(DATA['cache_root'])/'bcic_2a'/'index.csv'
    with path.open() as handle:
        index = list(csv.DictReader(handle))
    offset, row = next((i,r) for i,r in enumerate(index) if r['record_id'] == 'A08E' and r['row'] == '0')
    row = dict(row, task=tid, id=f'bcic_2a:{offset}', hz=int(row['hz']), samples=int(row['samples']))
    trial = np.array(np.load(path.parent/row['array'], mmap_mode='r')[0:1], copy=True)
    signal_adapter = LEAFSignalAdapter()
    channels = json.loads(row['channels'])
    keep, spatial = signal_adapter.spatial(channels)
    if row['hz'] != 200 or keep != list(range(len(channels))):
        raise ValueError('Benchmark assumes current unipolar 2a cache at 200 Hz')
    expected = signal_adapter.batch([row])[0][2]
    actual = preprocess(trial, spatial)
    if len(expected) != len(actual) or not all(np.array_equal(a,b) for a,b in zip(actual,expected)):
        raise ValueError('Benchmark preprocessing differs from production adapter')
    return tasks[tid], path, row, trial, channels, spatial


def leaf_cosine_readout(embeddings, prototypes, lengths):
    """Upstream CPU cosine scoring; duration-weighted scores for >10s only.

    A single chunk has exactly the upstream readout. Multi-chunk aggregation
    is a benchmark wrapper, not an original LEAF full-sequence capability.
    """
    if not embeddings or len(embeddings) != len(lengths) or any(n <= 0 for n in lengths):
        raise ValueError('Need one positive length per embedding')
    scores = torch.stack([emb.float().cpu() @ prototypes.T for emb in embeddings])
    if len(embeddings) == 1:
        return scores[0]
    weights = torch.tensor(lengths, dtype=scores.dtype)
    return (scores * (weights/weights.sum())[:,None,None]).sum(0)


def write_leaf_report(root, result):
    lines = ['# 原生 LEAF 推理速度', '',
        '_2026-11-24；本地发布版 MPNet 指令微调权重，未追加训练。_', '',
        '## 📋 测量对象', '',
        'LEAF不是生成式语言模型：MPNet预先编码任务指令与类别；完整LEAF用Tower、FiLM和4层Q-Former产生768维向量，再与类别原型做余弦匹配。官方直接推理使用缓存文本向量[^1][^2]。',
        f"本次严格载入发布权重，共{result['model_parameters']:,}参数；前向有效参数{result['forward_parameters']:,}，其中Tower {result['tower_forward_parameters']:,}、指令Q-Former {result['qformer_parameters']:,}。不包括离线MPNet。",
        '没有JEV、个人adapter或随机连接器；使用官方原有的8个query及训练好的Q-Former。', '',
        f"{result['device']}；batch=1；CPU线程{result['cpu_threads']}；每项预热{result['warmup']}次、测量{result['repeats']}次。CUDA同步墙钟中位数与P95，原始样本见[results.json](results.json)。", '',
        '## 📊 完整路径延迟', '',
        '| EEG时长（秒） | 精度 | 分块数 | 完整路径中位数（ms） | P95（ms） |',
        '| ---: | --- | ---: | ---: | ---: |']
    for row in sorted(result['results'],key=lambda r:(r['seconds'],r['precision'])):
        m = row['measurements']['end_to_end']['wall']
        lines.append(f"| {row['seconds']:g} | {row['precision']} | {len(row['chunk_seconds'])} | {m['median_ms']:.2f} | {m['p95_ms']:.2f} |")
    lines += ['', '## 🔍 阶段耗时', '',
        '| EEG时长/精度 | Tower（ms） | FiLM＋Q-Former（ms） | 完整LEAF前向（ms） | 峰值allocated（GiB） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in sorted(result['results'],key=lambda r:(r['seconds'],r['precision'])):
        m = row['measurements']
        med = lambda key: m[key]['wall']['median_ms']
        lines.append(f"| {row['seconds']:g}s / {row['precision']} | {med('tower'):.2f} | {med('qformer'):.2f} | {med('model_forward'):.2f} | {m['end_to_end']['peak_allocated_gib']:.3f} |")
    lines += ['', '## ⚠️ 比较边界', '',
        '- FP32是本地官方c_inference.py的默认推理精度。bf16_autocast保留FP32权重，只作为与既有信号侧精度接近的计算对照；未验证完整测试集精度变化。',
        '- 完整路径：内存中已滤波/重采样的200Hz EEG，归一化、映射65通道、分块、H2D、完整LEAF、768维向量D2H、CPU类别匹配及argmax。',
        '- 指令和类别文本向量在计时前从发布缓存载入；未测在线MPNet编码新指令的耗时，也不含读取EDF、滤波/重采样、模型加载、网络、排队和采集等待。',
        '- 单块输入上限10秒；16/30秒分别按8+8和10+10+10分块，各块分别执行整个LEAF，按时长加权平均余弦分数。本地添加的是测速汇总规则，不是官方原生超长输入或经过验证的分类方法。',
        '- 0.5–4秒截取同一个真实4秒trial前缀，更长输入重复该trial，仅测计算形状，不报告分类准确率。',
        '- 与JEV速度可作系统级比较，但输出机制、参数量、权重、文本缓存、精度不同，不是等价能力或纯参数量消融。阶段独立计时的中位数不能简单相加。', '',
        '## 🔗 可复现性', '',
        f"- 发布权重：`{result['checkpoint']}`；SHA256 `{result['checkpoint_sha256']}`。",
        f"- 脚本SHA256：`{result['script_sha256']}`。上游源码、配置、文本缓存、输入哈希见JSON。",
        '- 加载strict=True；单块拆分Tower/Q-Former与LEAF.forward输出逐值相同；无训练、量化、编译或上游源码修改。',
        '- 资源检测/Transformers技能用于安全加载与推理设置，Markdown技能用于记录测量边界；过程引用沿用[既有测速报告](../inference_latency_qwen8b_20261124/summary.md)。', '',
        '[^1]: Jiang et al. LEAF official repository. https://github.com/Jiang-Muyun/LEAF',
        '[^2]: LEAF direct inference. https://github.com/Jiang-Muyun/LEAF/blob/main/c_direct_inference.py', '']
    (root/'summary.md').write_text('\n'.join(lines))


def run_native_leaf(args, root):
    """Original release + cached text, no JEV or personal fine-tuning weights."""
    from eeg_align.runtime import CONFIG as leaf_config
    leaf_root = Path(leaf_config['leaf_root'])
    sys.path.insert(0,str(leaf_root))
    from LEAF import LEAF
    from load_config import build_model_config, load_yaml, load_targets, load_instructions
    config = build_model_config(load_yaml(leaf_root/'configs/LEAF_mpnet.yaml'))
    checkpoint = Path(leaf_config['checkpoint'])
    if digest(checkpoint) != leaf_config['checkpoint_sha256']:
        raise ValueError('Released LEAF checkpoint changed')
    model = LEAF(config)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.cuda().requires_grad_(False).eval()
    cache = leaf_root/'text_embeddings/mpnet-base'
    texts = cache.with_suffix('.txt').read_text().splitlines()
    vectors = np.load(cache.with_suffix('.npy'),allow_pickle=False).astype(np.float32)
    if vectors.shape != (len(texts),768) or len(set(texts)) != len(texts):
        raise ValueError('Invalid text cache')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Text cache is not normalized/finite')
    lookup = dict(zip(texts,vectors))
    instruction_text = load_instructions()['Motor'][0]
    labels = load_targets()['MI_BCIC_IV2a']
    prototypes = torch.from_numpy(np.stack([lookup[label] for label in labels]))
    # Official level 1: task-specific instruction, no candidate-label blending.
    instruction = torch.from_numpy(lookup[instruction_text].copy())[None].cuda()
    _, path, fixture, trial, channels, spatial = load_latency_fixture()
    count = lambda module: sum(p.numel() for p in module.parameters())
    dormant = sum(p.numel() for name,p in model.tower.named_parameters()
                  if name=='mask_token' or name.startswith(('mask_outProj.','next_outProj.')))
    sources = ['LEAF.py','load_config.py','c_inference.py','init_text_embeddings.py',
               'configs/LEAF_mpnet.yaml','configs/env.yaml','configs/tasks.yaml',
               'text_embeddings/mpnet-base.txt','text_embeddings/mpnet-base.npy']
    result = dict(status='running',backbone='leaf-native',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu_before=gpu_status(),batch=1,warmup=args.warmup,repeats=args.repeats,
        cpu_threads=torch.get_num_threads(),script_sha256=digest(__file__),
        source_sha256={name:digest(leaf_root/name) for name in sources},
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        model_parameters=count(model),forward_parameters=count(model)-dormant,
        tower_forward_parameters=count(model.tower)-dormant,qformer_parameters=count(model.qformer),
        text_encoding='offline published MPNet cache; no online text encoder loaded or timed',
        instruction_level=1,instruction_text=instruction_text,labels=labels,
        instruction_embedding=instruction.cpu().tolist(),
        fixture_id=fixture['id'],fixture_sha256=digest(path.parent/fixture['array']),
        input_hz=200,input_channels=channels,preprocessing_reference_equal=True,
        single_chunk_limit_seconds=10,long_aggregation='duration-weighted mean of CPU cosine scores (benchmark wrapper)',
        quantization=False,compile=False,personal_adapter=False,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,results=[])
    order = list(args.seconds)
    np.random.default_rng(42).shuffle(order)
    result['measurement_order_seconds'] = order
    precisions = ['fp32','bf16_autocast'] if args.leaf_precision=='both' else [args.leaf_precision]
    write_json(root/'results.json',result)
    with torch.inference_mode():
        for precision in precisions:
            for seconds in order:
                raw = duration_input(trial,seconds)
                pieces = preprocess(raw,spatial)
                gpu_pieces = [torch.from_numpy(p).cuda() for p in pieces]
                lengths = [p.shape[-1] for p in pieces]
                def tower_forward():
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16_autocast'):
                        return [model.tower(p) for p in gpu_pieces]
                tokens = tower_forward()
                def qformer_forward():
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16_autocast'):
                        return [torch.nn.functional.normalize(model.qformer(t,instruction),dim=-1) for t in tokens]
                def model_forward():
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16_autocast'):
                        return [model(p,instruction)[0] for p in gpu_pieces]
                embs = model_forward()
                if not all(torch.equal(a,b) for a,b in zip(embs,qformer_forward())):
                    raise ValueError('Decomposed LEAF forward differs from upstream')
                if not all(torch.isfinite(e).all() for e in embs):
                    raise ValueError('Nonfinite LEAF output')
                def readout():
                    return leaf_cosine_readout(embs,prototypes,lengths).argmax(-1).numpy()
                def end_to_end():
                    chunks = preprocess(raw,spatial)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16_autocast'):
                        outputs = [model(torch.from_numpy(p).cuda(),instruction)[0] for p in chunks]
                    return leaf_cosine_readout(outputs,prototypes,[p.shape[-1] for p in chunks]).argmax(-1).numpy()
                if not np.array_equal(end_to_end(),readout()):
                    raise ValueError('End-to-end readout differs from decomposed path')
                measures = {}
                for name,fn,cuda in [('end_to_end',end_to_end,True),
                    ('preprocess',lambda:preprocess(raw,spatial),False),('model_forward',model_forward,True),
                    ('tower',tower_forward,True),('qformer',qformer_forward,True),('readout',readout,True)]:
                    measures[name] = measure(fn,args.warmup,args.repeats,cuda)
                record = dict(seconds=seconds,precision=precision,input_shape=list(raw.shape),
                    chunk_seconds=[n/200 for n in lengths],eeg_tokens_per_chunk=[t.shape[1] for t in tokens],
                    qformer_queries_per_chunk=config.num_q,embedding_dimension=768,
                    native_single_chunk=len(pieces)==1,decomposed_forward_equal=True,
                    cosine_scores=leaf_cosine_readout(embs,prototypes,lengths).tolist(),measurements=measures)
                result['results'].append(record)
                write_json(root/'results.json',result)
                write_leaf_report(root,result)
                print(json.dumps(dict(seconds=seconds,precision=precision,
                    total_ms=measures['end_to_end']['wall']['median_ms'],
                    tower_ms=measures['tower']['wall']['median_ms'],
                    qformer_ms=measures['qformer']['wall']['median_ms'])),flush=True)
    if any(digest(leaf_root/name)!=sha for name,sha in result['source_sha256'].items()):
        raise ValueError('Upstream files changed during benchmark')
    result.update(status='complete',gpu_after=gpu_status(),
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(root/'results.json',result)
    write_leaf_report(root,result)
    print('COMPLETE',root/'summary.md',flush=True)


def write_report(root, result):
    lines = [f"# LEAF Tower → {result['backbone_label']} 推理延迟", '',
        f"单卡{result['device']}，batch=1；信号侧固定父模型InfoNCE→JEV step1360和2a被试8 joint_lora个人权重。",
        '保留原信号侧精度与未合并LoRA，LM使用bf16/SDPA；eval + inference_mode，不编译、不量化，不修改训练实现。', '',
        f"连接器状态：{result['connector_status']}。语言主干参数：{result['language_parameters']:,}；隐藏维度：{result['hidden_size']}。",
        '小模型是计算速度对照，不是EEG分类效果对照；模型架构、语言LoRA路径和接口也发生变化，不能把速度差异全部归因于参数量。' if result['speed_control_only'] else '本组使用已训练的JEV与原连接器。', '',
        '| EEG 时长（秒） | 分块长度（秒） | EEG tokens | 总输入 tokens | CPU 预处理（ms） | Tower（ms） | adapter＋连接器（ms） | LM bridge（ms） | 完整路径中位数（ms） | 完整路径 P95（ms） | 峰值分配显存（GiB） |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(result['results'], key=lambda r:r['seconds']):
        m = row['measurements']
        med = lambda key: m[key]['wall']['median_ms']
        lines.append(f"| {row['seconds']:g} | {row['chunk_seconds']} | {row['eeg_tokens']} | {row['lm_tokens']} | "
            f"{med('preprocess'):.2f} | {med('tower'):.2f} | {med('adapter_connector'):.2f} | "
            f"{med('jev_bridge'):.2f} | {med('end_to_end'):.2f} | "
            f"{m['end_to_end']['wall']['p95_ms']:.2f} | {m['end_to_end']['peak_allocated_gib']:.2f} |")
    lines += ['', '## 📋 测量口径', '',
        f"- 每个长度/阶段预热 {result['warmup']} 次，测量 {result['repeats']} 次；顺序固定随机化，原始样本保存在 results.json。",
        '- 表内为 CUDA 同步后的主机墙钟时间，包含 Python 调度；另存 CUDA stream 时间。',
        '- 完整路径单独测量：内存中200 Hz EEG → 归一化/65通道映射/分块 → CPU到GPU → Tower → 个人adapter → 对应connector → LM → 概率回CPU。',
        '- 不含信号采集等待、EDF读取、上游滤波/重采样、模型加载、服务器排队、网络传输。固定提示词预先tokenize。',
        '- LM bridge（JSON字段jev_bridge）包含connector、输入拼接、LM和choice head；与adapter＋连接器列有重叠，不能直接相加。',
        '- lm_body 是准备好 inputs_embeds 后仅语言主干的独立计时，详细分解见 JSON。',
        '- 0.5–4秒取同一真实4秒trial的前缀；更长片段重复该trial，仅评估计算形状，不是连续真实MI记录，也没有评估准确率。',
        '- 采用现有最长10秒分块策略，16秒分为8+8秒，30秒分为10+10+10秒；块内双分支tokens按原顺序拼接。',
        '- 峰值为 PyTorch allocated 显存（含模型），不是整卡 nvidia-smi 占用；GPU并发状态记录在JSON。', '',
        '## 🔗 可复现性', '',
        f"- 脚本 SHA256：`{result['script_sha256']}`。",
        f"- 父 checkpoint SHA256：`{result['parent_sha256']}`。",
        f"- 个人 checkpoint SHA256：`{result['personal_sha256']}`。",
        f"- 语言权重来源：`{result['language_checkpoint']}`；索引/配置哈希及分片大小/mtime见JSON。",
        '- 预处理与现有 LEAFSignalAdapter.batch 逐值一致；拆分 LM 路径与原 JEVBridge logits 核对一致。',
        '- 资源检测技能用于确认单卡余量；模型技能用于保持本地权重、eval及无梯度推理。', '']
    (root/'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, nargs='+', default=[.5, 1, 2, 4, 8, 16, 30])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--backbone',choices=['jev','qwen3vl-8b','leaf-native'],default='jev')
    parser.add_argument('--small-checkpoint',default='third_party/ECG-R1/ECG-R1-8B-RL')
    parser.add_argument('--leaf-precision',choices=['fp32','bf16_autocast','both'],default='both')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError('Need warmup >=1 and repeats >=2')
    os.chdir(REPO)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(42)
    if args.backbone == 'leaf-native':
        return run_native_leaf(args,root)
    print('Loading current trained Tower and',args.backbone,flush=True)
    loaded_at = time.perf_counter()
    saved = load_parent()
    encoder = load_encoder(saved)
    if args.backbone == 'jev':
        server, tokenizer = load_server(saved, encoder)
        language_path = Path('laya/JEV-27B-VL')
    else:
        text, weight, tokenizer = load_small_language(args.small_checkpoint)
        server = SpeedOnlyBridge(text,weight,tokenizer)
        language_path = Path(args.small_checkpoint)
    parent_sha = digest(Path(CONFIG['parent'])/'best.pt')
    del saved
    personal_path = (Path('artifacts/benchmark')/CONFIG['run_name']/'episodes'
                     /'bcic_2a_s8_joint_lora_k5_seed42'/'adapter.pt')
    saved_personal = torch.load(personal_path, map_location='cpu', weights_only=False, mmap=True)
    if saved_personal['parent_checkpoint_sha256'] != parent_sha:
        raise ValueError('Personal and shared checkpoints do not match')
    adapter = ResidualAdapter().cuda()
    personal, _ = configure_personal(encoder, adapter, 'lora')
    copy_trainable(personal, saved_personal['personal'])
    del saved_personal
    personal.requires_grad_(False).eval()
    server.requires_grad_(False).eval()
    server.text.gradient_checkpointing_disable()
    task, path, row, trial, channels, spatial = load_latency_fixture()
    suffix = suffix_for(task, tokenizer)
    result = dict(status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(), gpu_before=gpu_status(), batch=1,
        warmup=args.warmup, repeats=args.repeats, cpu_threads=torch.get_num_threads(),
        script_sha256=digest(__file__), parent_sha256=parent_sha, personal_sha256=digest(personal_path),
        model_loading_seconds=time.perf_counter()-loaded_at, fixture_id=row['id'],
        fixture_sha256=digest(path.parent/row['array']), input_hz=200, input_channels=channels,
        prefix_text_tokens=server.prefix_ids.numel(), suffix_text_tokens=suffix.numel(),
        precision='Tower fp32 weights/bf16 autocast; adapter/connector fp32; LM bf16 with native fp32 modules; existing Tower/JEV LoRA unchanged',
        lora_merged=False, compile=False, quantization=False, kv_cache=False,
        backbone=args.backbone,backbone_label=('JEV-27B' if args.backbone=='jev' else 'Qwen3-VL-8B (ECG-R1 derivative)'),
        speed_control_only=args.backbone!='jev',
        connector_status=('已训练256→5120→5120' if args.backbone=='jev' else '随机初始化256→4096→4096，仅测速，不代表EEG对齐'),
        language_parameters=sum(p.numel() for p in server.text.parameters()),
        connector_parameters=sum(p.numel() for p in server.connector.parameters()),
        hidden_size=server.text.config.hidden_size, language_checkpoint=str(language_path),
        language_index_sha256=digest(language_path/'model.safetensors.index.json'),
        language_config_sha256=digest(language_path/'config.json'),
        language_shard_stats={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                              for p in sorted(language_path.glob('model-*.safetensors'))},
        preprocessing_reference_equal=True, results=[])
    write_json(root/'results.json', result)
    order = list(args.seconds)
    np.random.default_rng(42).shuffle(order)
    result['measurement_order_seconds'] = order
    from eeg_align.jev import assemble_embeddings, choice_logits
    with torch.inference_mode():
        for seconds in order:
            raw = duration_input(trial, seconds)
            pieces = preprocess(raw, spatial)
            gpu_pieces = [torch.from_numpy(p).cuda() for p in pieces]
            def tower_forward():
                return torch.cat([encoder(p) for p in gpu_pieces], dim=1)
            tokens = tower_forward()
            adapted = adapter(tokens)
            valid = torch.ones(adapted.shape[:2], device='cuda', dtype=torch.bool)
            def bridge_forward():
                return server(adapted, valid, suffix_ids=[suffix], num_choices=4)['logits']
            embed = server.text.get_input_embeddings()
            x, mask, positions, last = assemble_embeddings(embed(server.prefix_ids),
                server.connector(adapted).to(embed.weight.dtype), [embed(suffix)], valid)
            def lm_forward():
                return server.text(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                    use_cache=False, return_dict=True).last_hidden_state
            split_logits = choice_logits(lm_forward()[torch.arange(1,device='cuda'),last].float(),
                                          server.head_weight, server.head_bias, 4)
            original_logits = bridge_forward()
            if not torch.equal(split_logits, original_logits):
                raise ValueError('Decomposed forward differs from current JEVBridge')
            def end_to_end():
                chunks = preprocess(raw, spatial)
                z = adapter(torch.cat([encoder(torch.from_numpy(p)) for p in chunks], dim=1))
                m = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
                logits = server(z, m, suffix_ids=[suffix], num_choices=4)['logits']
                return logits.softmax(-1).cpu()
            m = {}
            # Main number has no module hooks and no per-stage sync barriers.
            for key, fn, cuda in [('end_to_end',end_to_end,True), ('preprocess',lambda:preprocess(raw,spatial),False),
                                 ('tower',tower_forward,True), ('adapter_connector',lambda:server.connector(adapter(tokens)),True),
                                 ('jev_bridge',bridge_forward,True), ('lm_body',lm_forward,True)]:
                m[key] = measure(fn, args.warmup, args.repeats, cuda)
            record = dict(seconds=seconds, input_shape=list(raw.shape),
                chunk_seconds=[p.shape[-1]/200 for p in pieces], eeg_tokens=tokens.shape[1],
                lm_tokens=x.shape[1], logits_equal=True, measurements=m)
            result['results'].append(record)
            write_json(root/'results.json', result)
            write_report(root, result)
            print(json.dumps(dict(seconds=seconds,tokens=tokens.shape[1],
                total_ms=m['end_to_end']['wall']['median_ms'],tower_ms=m['tower']['wall']['median_ms'],
                lm_ms=m['lm_body']['wall']['median_ms'])), flush=True)
    result['status'] = 'complete'
    result['gpu_after'] = gpu_status()
    result['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write_json(root/'results.json',result)
    write_report(root,result)
    print('COMPLETE',root/'summary.md',flush=True)


if __name__ == '__main__':
    main()
