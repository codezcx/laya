"""Load the finished shared checkpoint once, then reuse it for independent users."""
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from jev_train.checkpoint import restore_model, seed_rng
from jev_train.jev import JEVBridge, choice_logits, load_jev
from jev_train.online_leaf import OnlineLEAFTower, encode_batch
from jev_train.runtime import ROOT, digest
from jev_train.task_options import make_variant
from .common import CONFIG, PARENT_CONFIG
from .data import PreparedAdapter
from .modules import canonical_logits


class PersonalServer:
    def __init__(self, parent_digest):
        parent = ROOT / CONFIG['parent']
        done = json.loads((parent / 'complete.json').read_text())
        if done['smoke'] or not done['checkpoint_reload_verified']:
            raise ValueError('Personalization requires a verified completed shared parent')
        checkpoint = parent / 'best.pt'
        if digest(checkpoint) != parent_digest:
            raise ValueError('Shared checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if saved['step'] != done['selected_step']:
            raise ValueError('Selected shared checkpoint mismatch')
        metadata = json.loads((parent / 'config.json').read_text())
        for name, expected in metadata['jev_shard_stats'].items():
            st = (ROOT / 'laya/JEV-27B-VL' / name).stat()
            if [st.st_size, st.st_mtime_ns] != expected:
                raise RuntimeError('Native JEV shard differs from historical parent')
        seed_rng(CONFIG['seed'])
        text, weight, bias, self.tokenizer = load_jev('cuda:0')
        model = JEVBridge(text, weight, bias, self.tokenizer, train_lora=True,
                          train_head=True, head_choices=16, input_dim=256)
        model.add_module('eeg_encoder', OnlineLEAFTower(PARENT_CONFIG).cuda())
        restore_model(model, saved)
        self.parent_step = saved['step']
        del saved
        model.requires_grad_(False).eval()
        self.model = model
        self.prompt_cache = {}
        self.frozen_versions = {name: p._version for name, p in model.named_parameters()}
        self.frozen_buffers = {name: (b._version, tuple(b.shape)) for name, b in model.named_buffers()}

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Shared model was made trainable or received parameter gradients')
        if self.frozen_versions != {n: p._version for n, p in self.model.named_parameters()}:
            raise RuntimeError('A frozen shared parameter was modified')
        if self.frozen_buffers != {n: (b._version, tuple(b.shape)) for n, b in self.model.named_buffers()}:
            raise RuntimeError('A frozen shared buffer was modified')

    def text(self, value):
        if value not in self.prompt_cache:
            self.prompt_cache[value] = torch.tensor(self.tokenizer.encode(value, add_special_tokens=False), device='cuda')
        return self.model.text.get_input_embeddings()(self.prompt_cache[value])

    @torch.no_grad()
    def encode(self, store, rows):
        self.model.eval()
        output = []
        adapter = PreparedAdapter(store)
        for offset in range(0, len(rows), CONFIG['encode_batch']):
            batch = rows[offset:offset + CONFIG['encode_batch']]
            tokens, valid = encode_batch(self.model.eeg_encoder, adapter.batch(batch), len(batch))
            if not valid.all():
                raise ValueError('Personal cohort unexpectedly mixes signal lengths')
            output.append(tokens.detach().cpu())
        return torch.cat(output)

    def project(self, tokens, user):
        c = self.model.connector
        z = c[2](c[1](c[0](tokens.float())))
        result = c[3](z)
        if user is not None and hasattr(user, 'delta'):
            result = result + user.delta(z)
        return result

    def forward(self, tokens, rows, task, user=None, step=0, training=False, variant=0):
        self.model.text.train(training)
        self.model.eeg_encoder.eval()
        z = self.project(tokens.cuda(), user)
        prefix = self.model.text.get_input_embeddings()(self.model.prefix_ids)
        questions = [make_variant(task, r['label'], r['id'], step, CONFIG['seed'],
                                  None if training else variant) for r in rows]
        sequences = []
        for x, question in zip(z, questions):
            parts = [prefix]
            parts.append(x.to(prefix.dtype))
            parts.append(self.text(question['suffix']))
            sequences.append(torch.cat(parts))
        lengths = torch.tensor([len(s) for s in sequences], device='cuda')
        inputs = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(inputs.shape[1], device='cuda')[None] < lengths[:, None]
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.model.text(inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
                                 use_cache=False, return_dict=True).last_hidden_state
        state = hidden[torch.arange(len(rows), device='cuda'), lengths - 1]
        logits = choice_logits(state.float(), self.model.head_weight, self.model.head_bias, len(task['classes']))
        return canonical_logits(logits, [[int(c) for c in q['option_class_ids']] for q in questions])

    @torch.no_grad()
    def predict(self, tokens, rows, task, user=None, variant=0):
        batch = CONFIG['eval_batch']
        return torch.cat([self.forward(tokens[i:i + batch], rows[i:i + batch], task, user,
                                      variant=variant).cpu()
                          for i in range(0, len(rows), batch)])
