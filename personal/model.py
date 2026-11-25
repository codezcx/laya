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
from .data import PreparedAdapter, SignalBatch
from .modules import canonical_logits, install_tower_lora, tower_lora_parameters


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
        self.tower_lora_manifest = install_tower_lora(model.eeg_encoder,
            CONFIG['tower_lora_rank'],CONFIG['tower_lora_scale'],CONFIG['tower_lora_seed'])
        model.requires_grad_(False)
        self.tower_enabled = False
        self.initial_tower = self.tower_state()
        self.prompt_cache = {}
        self.personal_names = {'eeg_encoder.'+name for name in tower_lora_parameters(model.eeg_encoder)}
        self.frozen_versions = {name: p._version for name, p in model.named_parameters()
                                if name not in self.personal_names}
        self.frozen_buffers = {name: (b._version, tuple(b.shape)) for name, b in model.named_buffers()}

    def assert_frozen(self):
        frozen = {n: p for n, p in self.model.named_parameters() if n not in self.personal_names}
        if any(p.requires_grad or p.grad is not None for p in frozen.values()):
            raise RuntimeError('Shared model was made trainable or received parameter gradients')
        if any(self.frozen_versions[n] != p._version for n, p in frozen.items()):
            raise RuntimeError('A frozen shared parameter was modified')
        if self.frozen_buffers != {n: (b._version, tuple(b.shape)) for n, b in self.model.named_buffers()}:
            raise RuntimeError('A frozen shared buffer was modified')
        for p in tower_lora_parameters(self.model.eeg_encoder).values():
            if p.requires_grad != self.tower_enabled or (not self.tower_enabled and p.grad is not None):
                raise RuntimeError('Tower LoRA gradient ownership differs from the assigned arm')

    def tower_state(self):
        return {n: p.detach().cpu().clone() for n, p in tower_lora_parameters(self.model.eeg_encoder).items()}

    def load_tower(self, state):
        self.assert_frozen()
        parameters=tower_lora_parameters(self.model.eeg_encoder)
        if parameters.keys()!=state.keys() or any(p.shape!=state[n].shape for n,p in parameters.items()):
            raise RuntimeError('Tower LoRA state shape/key mismatch')
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(state[name])

    def configure_arm(self, arm):
        self.assert_frozen()
        self.model.eeg_encoder.zero_grad(set_to_none=True)
        self.load_tower(self.initial_tower)
        self.tower_enabled = arm == 'tower_both'
        for p in tower_lora_parameters(self.model.eeg_encoder).values():
            p.requires_grad_(self.tower_enabled)
        self.model.eeg_encoder.eval()
        self.assert_frozen()

    def tower_parameters(self):
        return list(tower_lora_parameters(self.model.eeg_encoder).values()) if self.tower_enabled else []

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
        normalized=c[0](tokens.float())
        z=c[1](normalized)
        if user is not None and hasattr(user,'first'):
            z=z+user.first(normalized)
        z=c[2](z)
        result = c[3](z)
        if user is not None and hasattr(user, 'delta'):
            result = result + user.delta(z)
        return result

    def forward(self, tokens, rows, task, user=None, step=0, training=False, variant=0):
        self.model.text.train(training)
        self.model.eeg_encoder.eval()
        if isinstance(tokens, SignalBatch):
            tokens = torch.cat([self.model.eeg_encoder(piece) for piece in tokens.pieces], dim=1)
        elif self.tower_enabled:
            raise RuntimeError('Trainable Tower cannot consume frozen embedding caches')
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
