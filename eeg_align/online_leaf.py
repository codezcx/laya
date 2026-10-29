"""Released LEAF task-conditioned query path; no old downstream weights."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .runtime import CONFIG
from .benchmark_data import catalog
from .normalization import apply_bn_policy

sys.path.insert(0, CONFIG['leaf_root'])
from LEAF import LEAF
from load_config import build_model_config, load_yaml


class OnlineLEAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = LEAF(build_model_config(load_yaml(Path(CONFIG['leaf_root']) / 'configs/LEAF_mpnet.yaml')))
        self.leaf.load_state_dict(torch.load(CONFIG['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        for name, parameter in self.leaf.named_parameters():
            if name == 'tower.mask_token' or name.startswith(('tower.mask_outProj.', 'tower.next_outProj.', 'qformer.output_embed.')):
                parameter.requires_grad_(False)
        root = Path(CONFIG['text_root'])
        info = json.loads((root / 'instructions.json').read_text())
        values = np.load(root / 'instructions.npy')
        self.register_buffer('instructions', torch.from_numpy(np.stack([values[info['texts'].index(t['instruction'])] for t in catalog()])))
        # Published running statistics remain fixed across heterogeneous montages.
        self.leaf.eval()
        apply_bn_policy(self.leaf, 'frozen')

    def forward(self, batch, count, unfrozen=False):
        # Eval mode disables dropout, NOT gradients. All query-path weights can SFT.
        self.leaf.eval()
        apply_bn_policy(self.leaf, 'frozen')
        outputs = [None] * count
        for tid, indices, pieces in batch:
            instruction = self.instructions[tid:tid + 1].expand(len(indices), -1)
            queries = []
            with torch.set_grad_enabled(torch.is_grad_enabled() and unfrozen):
                for piece in pieces:
                    x = torch.from_numpy(piece.copy()).to(self.instructions.device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        # Avoid unused reconstruction and text-output projection.
                        tokens = self.leaf.tower(x)
                        _, history = self.leaf.qformer(tokens, instruction, return_all_q=True)
                        q = self.leaf.qformer.final_norm(history[-1])
                    queries.append(q.float())
                q = torch.cat(queries, dim=1)
            for i, index in enumerate(indices):
                outputs[index] = q[i]
        if any(x is None for x in outputs):
            raise ValueError('Missing encoded samples')
        lengths = torch.tensor([len(q) for q in outputs], device=self.instructions.device)
        width = int(lengths.max())
        tokens = torch.stack([F.pad(q, (0, 0, 0, width - len(q))) for q in outputs])
        valid = torch.arange(width, device=tokens.device)[None] < lengths[:, None]
        return tokens, valid
