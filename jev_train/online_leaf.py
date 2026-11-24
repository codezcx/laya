"""Released LEAF signal Tower only: no text encoder, FiLM or Q-Former."""
from pathlib import Path
import sys
import torch
from .runtime import CONFIG, digest
from .checkpoint import tensor_fingerprint


class OnlineLEAFTower(torch.nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        if digest(config['checkpoint']) != config['checkpoint_sha256']:
            raise ValueError('Released LEAF checkpoint changed')
        root = Path(config['leaf_root'])
        sys.path.insert(0, str(root))
        from LEAF import Tower
        from load_config import build_model_config, load_yaml
        self.tower = Tower(build_model_config(load_yaml(root / 'configs/LEAF_mpnet.yaml')))
        state = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
        tower_state = {k.removeprefix('tower.'): v for k, v in state.items() if k.startswith('tower.')}
        if not tower_state:
            raise ValueError('Expected instruction-tuned checkpoint with tower.* parameters')
        self.tower.load_state_dict(tower_state, strict=True)
        self.release_audit = dict(checkpoint_sha256=config['checkpoint_sha256'],
            released_tower_sha256=tensor_fingerprint(tower_state),
            ignored_keys=sorted(k for k in state if not k.startswith('tower.')))
        # These heads exist only for self-supervised reconstruction, not forward().
        del self.tower.mask_token, self.tower.mask_outProj, self.tower.next_outProj
        self.release_audit['forward_tower_sha256'] = tensor_fingerprint(self.tower.state_dict())
        self.set_adaptation_enabled(True)

    def set_adaptation_enabled(self, enabled):
        self.tower.requires_grad_(enabled)
        # Preserve the historical LEAF inference/dropout and BatchNorm policy.
        self.tower.eval()

    def train(self, mode=True):
        super().train(mode)
        self.tower.eval()
        return self

    def forward(self, x):
        device = next(self.tower.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            return self.tower(x.to(device=device, dtype=torch.float32)).float()


def encode_batch(encoder, adapted, size):
    """Keep both branches and every ordered chunk, without a learned resampler."""
    samples = [None] * size
    for _, indices, pieces in adapted:
        joined = torch.cat([encoder(torch.from_numpy(p)) for p in pieces], dim=1)
        for index, value in zip(indices, joined):
            samples[index] = value
    if any(s is None for s in samples):
        raise ValueError('Missing encoded row')
    tokens = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
    lengths = torch.tensor([len(s) for s in samples], device=tokens.device)
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
    return tokens, valid
