#!/usr/bin/env python3
"""Read-only, validation-only representation audit of completed BSR/JEV arms.

No optimization, test access, or checkpoint selection. Uses the existing selected
best.pt and canonical evaluation options. CPU arrays retain full precision.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from eeg_align.bsr_data import BSRSignalAdapter, encode_batch
from eeg_align.checkpoint import restore_model, seed_rng, tensor_fingerprint
from eeg_align.evaluation import validation_rows
from eeg_align.jev import JEVBridge, load_jev, assemble_embeddings, choice_logits
from eeg_align.multitask_data import TrialStore, write_json
from eeg_align.online_bsr import OnlineBSR
from eeg_align.runtime import CONFIG, digest, arm_settings
from eeg_align.aligned_checkpoint import load_aligned_encoder
from eeg_align.task_options import make_variant
from eeg_align.token_resampler import TokenReadout


@torch.no_grad()
def decision(model, reduced, valid, suffix_ids, choices):
    """Exact JEVBridge.forward computation, also returning internal states."""
    z = model.connector(reduced.masked_fill(~valid[..., None], 0).float())
    embed = model.text.get_input_embeddings()
    x, mask, positions, last = assemble_embeddings(
        embed(model.prefix_ids), z.to(embed.weight.dtype),
        [embed(s) for s in suffix_ids], valid)
    h = model.text(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                   use_cache=False, return_dict=True).last_hidden_state
    state = h[torch.arange(len(h), device=h.device), last].float()
    logits = choice_logits(state, model.head_weight, model.head_bias, choices)
    return z, state, logits


@torch.no_grad()
def text_state(model, tokenizer, strings):
    """Last contextual state of exact text; not a trained CLIP sentence embedding."""
    ids = [torch.tensor(tokenizer.encode(s, add_special_tokens=False), device='cuda') for s in strings]
    n = torch.tensor([len(s) for s in ids], device='cuda')
    tokens = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = torch.arange(tokens.shape[1], device='cuda')[None] < n[:, None]
    hidden = model.text(input_ids=tokens, attention_mask=mask,
        position_ids=(mask.long().cumsum(-1)-1).clamp_min(0), use_cache=False,
        return_dict=True).last_hidden_state
    return hidden[torch.arange(len(ids), device='cuda'), n-1].float().cpu().numpy()


def verify_identity(identity, store):
    if identity['config'] != CONFIG:
        raise ValueError('Current configuration differs from trained model')
    for path, expected in identity['code_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Training source changed: {path}')
    if identity['cache_indexes'] != {t['task_id']: store.meta[i]['index_sha256']
                                    for i, t in enumerate(store.tasks)}:
        raise ValueError('Data indexes differ from training')
    root = Path('laya/JEV-27B-VL')
    for path, expected in identity['jev_config_sha256'].items():
        if digest(root/path) != expected:
            raise ValueError(f'JEV release metadata changed: {path}')
    for name, expected in identity['jev_shard_stats'].items():
        stat = (root/name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f'JEV shard changed: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--arms', choices=CONFIG['arms'], nargs='+', required=True)
    p.add_argument('--cap', type=int, default=256)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    if args.cap < 128 or args.batch < 1:
        raise ValueError('Require cap >= 128 and positive batch size')
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    seed_rng(42)
    store, adapter = TrialStore(), BSRSignalAdapter()
    selection = {i: validation_rows(store, i, args.cap) for i, t in enumerate(store.tasks)
                 if t['role'] == 'benchmark'}
    if any(r['split'] != 'valid' for rows in selection.values() for r in rows):
        raise RuntimeError('Only validation windows may be loaded')
    root = args.run/'representation_analysis'
    root.mkdir(exist_ok=True)
    for arm in args.arms:
        if (root/arm).exists():
            raise FileExistsError(f'Preserve existing diagnostic output: {root/arm}')
    text, weight, bias, tokenizer = load_jev('cuda:0')
    model = JEVBridge(text, weight, bias, tokenizer, True, True, 16, 768)
    release_hash = digest(CONFIG['checkpoint'])
    for arm in args.arms:
        started = time.monotonic()
        folder = root/arm
        folder.mkdir()
        checkpoint = args.run/'train'/arm/'best.pt'
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        verify_identity(saved['identity'], store)
        if saved['identity']['checkpoint_sha256'] != release_hash:
            raise ValueError('BSR release checksum mismatch')
        setting = arm_settings(arm)
        encoder = OnlineBSR(CONFIG, trainable=setting['encoder_trainable'])
        load_aligned_encoder(encoder, setting, CONFIG, saved['identity']['cache_indexes'])
        model.add_module('eeg_encoder', encoder.cuda())
        model.add_module('readout', TokenReadout(setting['readout']).cuda())
        restore_model(model, saved)
        initial_bsr_hash = tensor_fingerprint(model.eeg_encoder.state_dict())
        model.eval()
        model.text.gradient_checkpointing_disable()
        selected_step = saved['step']
        del saved
        manifests, summaries, text_conditions, text_candidates = {}, {}, [], []
        candidate_labels, condition_prompts, candidate_prompts = [], [], []
        for tid, rows in selection.items():
            task = store.tasks[tid]
            name = task['task_id']
            classes = len(task['classes'])
            q = make_variant(task, 0, 'geometry', 0, 42, 0)
            suffix = torch.tensor(tokenizer.encode(q['suffix'], add_special_tokens=False), device='cuda')
            manifests[name] = [dict(id=r['id'], subject=r['subject'], label=r['label'], split=r['split']) for r in rows]
            # Null slots preserve token count and positions, but are an OOD control.
            with torch.no_grad():
                zeros = torch.zeros(1, model.readout.output_tokens, 768, device='cuda')
                valid_zero = torch.ones(zeros.shape[:2], device='cuda', dtype=torch.bool)
                _, null_state, null_logits = decision(model, zeros, valid_zero, [suffix], classes)
            arrays = defaultdict(list)
            reference = np.load(args.run/'train'/arm/f'valid_{name}_predictions.npz')
            ref_index = {key: i for i, key in enumerate(reference['ids'])}
            max_probability_error = 0.
            for offset in range(0, len(rows), args.batch):
                chunk = rows[offset:offset+args.batch]
                with torch.no_grad():
                    tokens, valid = encode_batch(model.eeg_encoder, adapter.batch(chunk), len(chunk))
                    bsr_mean = (tokens.float()*valid[..., None]).sum(1)/valid.sum(1)[:, None]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        reduced, readout_valid = model.readout(tokens, valid)
                    z, state, logits = decision(model, reduced, readout_valid, [suffix]*len(chunk), classes)
                    if offset == 0:
                        expected = model(reduced, readout_valid, suffix_ids=[suffix]*len(chunk),
                                         num_choices=classes)['logits']
                        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                    reduced = reduced.float()
                    normed = torch.nn.functional.normalize(reduced, dim=-1)
                    if reduced.shape[1] > 1:
                        count = reduced.shape[1]
                        token_cosine = ((normed@normed.transpose(1, 2)).sum((1, 2))-count)/(count*(count-1))
                    else:
                        token_cosine = torch.full((len(chunk),), float('nan'), device='cuda')
                    values = dict(bsr_mean=bsr_mean, readout_mean=reduced.mean(1),
                        readout_flat=reduced.flatten(1), connector_mean=z.mean(1), decision=state,
                        probabilities=logits.softmax(-1), token_cosine=token_cosine)
                    for key, value in values.items():
                        arrays[key].append(value.float().cpu().numpy())
                old = reference['probabilities'][[ref_index[r['id']] for r in chunk]]
                max_probability_error = max(max_probability_error, float(np.abs(old-arrays['probabilities'][-1]).max()))
                if offset % 64 == 0:
                    print(json.dumps(dict(arm=arm, dataset=name, windows=offset+len(chunk),
                        total=len(rows), seconds=round(time.monotonic()-started))), flush=True)
            arrays = {k: np.concatenate(v) for k, v in arrays.items()}
            if max_probability_error > .01:
                raise ValueError(f'Saved prediction parity failed {name}: {max_probability_error}')
            arrays.update(labels=np.array([r['label'] for r in rows]),
                          subjects=np.array([r['subject'] for r in rows]),
                          ids=np.array([r['id'] for r in rows]),
                          null_decision=null_state.cpu().numpy(), null_probabilities=null_logits.softmax(-1).cpu().numpy())
            np.savez_compressed(folder/f'{name}.npz', **arrays)
            from eeg_align.jev import PREFIX
            condition_prompts.append(PREFIX + q['suffix'])
            text_conditions.append(text_state(model, tokenizer, [condition_prompts[-1]])[0])
            candidates = [f"[task] {task['instruction']}\n[question] {task['question']}\n[candidate] {c['option']}\n[representation]:"
                          for c in task['classes']]
            candidate_prompts.extend(candidates)
            text_candidates.extend(text_state(model, tokenizer, candidates))
            candidate_labels.extend([dict(dataset=name, **c) for c in task['classes']])
            summaries[name] = dict(n=len(rows), subjects=len(set(arrays['subjects'])),
                class_counts=np.bincount(arrays['labels'], minlength=classes).tolist(),
                max_saved_probability_error=max_probability_error,
                accuracy=float((arrays['probabilities'].argmax(1)==arrays['labels']).mean()))
            write_json(folder/'progress.json', dict(status='extracting', arm=arm, datasets=summaries))
        np.savez_compressed(folder/'text.npz', condition=np.array(text_conditions),
                            candidate=np.array(text_candidates))
        write_json(folder/'manifest.json', dict(samples=manifests, tasks=[store.tasks[i] for i in selection],
            condition_prompts=condition_prompts, candidate_prompts=candidate_prompts,
            candidate_labels=candidate_labels, sampling='existing validation_rows(cap); legacy 128 + class/subject queues',
            cap=args.cap, option_order='canonical fixed as original validation', seed=42))
        verify_identity(json.loads((args.run/'train'/arm/'config.json').read_text()), store)
        if initial_bsr_hash != tensor_fingerprint(model.eeg_encoder.state_dict()):
            raise RuntimeError('Frozen BSR changed')
        import transformers
        write_json(folder/'complete.json', dict(status='complete', arm=arm, selected_step=selected_step,
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            extractor_sha256=digest(__file__), datasets=summaries, test_evaluations=0,
            weights_updated=False, frozen_bsr_sha256=initial_bsr_hash,
            visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), elapsed_seconds=time.monotonic()-started,
            versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__)))
        print('COMPLETE', arm, time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
