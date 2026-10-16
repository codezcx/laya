"""Cached REVE tokens -> real Qwen decoder; no task ID is supplied."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import assemble_embeddings

CLASSES = ('negative', 'neutral', 'positive', 'left_hand', 'right_hand')
PREFIX = 'Identify the task and class represented by this EEG recording.\nEEG begins:\n'
SUFFIX = ('\nEEG ends.\nOptions: A. emotion recognition, negative; '
          'B. emotion recognition, neutral; C. emotion recognition, positive; '
          'D. motor imagery, left hand; E. motor imagery, right hand.\nAnswer:')


def contrastive_pair_loss(signal, text, signal_labels, text_labels, temperature=.1):
    """Balanced sigmoid pair supervision; all same-class descriptions are positive.

    This helper is not yet wired into the archived CE trainer. Repeated class
    descriptions must not become off-diagonal negatives as in one-to-one CLIP.
    Every row and column needs both positive and negative pairs.
    """
    if temperature<=0 or signal.ndim!=2 or text.ndim!=2 or signal.shape[1]!=text.shape[1]:
        raise ValueError('Invalid embeddings/temperature')
    if signal_labels.shape!=(len(signal),) or text_labels.shape!=(len(text),):
        raise ValueError('Invalid label shape')
    positive=signal_labels[:,None]==text_labels[None,:]
    negative=~positive
    if not all(mask.any(axis).all() for mask in [positive,negative] for axis in [0,1]):
        raise ValueError('Both pair types required for every signal and text anchor')
    scores=F.normalize(signal.float(),dim=-1)@F.normalize(text.float(),dim=-1).T/temperature
    pos=F.softplus(-scores)*positive;neg=F.softplus(scores)*negative
    row=.5*(pos.sum(1)/positive.sum(1)+neg.sum(1)/negative.sum(1))
    col=.5*(pos.sum(0)/positive.sum(0)+neg.sum(0)/negative.sum(0))
    return .5*(row.mean()+col.mean())


class MultitaskBridge(nn.Module):
    """Frozen decoder and token readout, trainable bridge, unmasked five-class CE.

    Input tokens must be actual REVE embeddings (B,K,512), never spectral
    features or a flattened channel vector. Encoder caching is external.
    """
    def __init__(self, text, head_weight, prefix_ids, suffix_ids, input_dim=512):
        super().__init__()
        if head_weight.ndim != 2 or head_weight.shape[0] != len(CLASSES):
            raise ValueError('Expected A-E readout weights')
        self.text = text.requires_grad_(False).eval()
        hidden = head_weight.shape[1]
        self.connector = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                                       nn.GELU(), nn.Linear(hidden, hidden))
        self.register_buffer('head_weight', head_weight.detach().float().clone())
        self.register_buffer('prefix_ids', prefix_ids)
        self.register_buffer('suffix_ids', suffix_ids)

    def train(self, mode=True):
        super().train(mode)
        self.text.eval()
        return self

    def forward(self, tokens, valid, labels=None):
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Expected B,K,D tokens and boolean B,K validity mask')
        if not valid.any(1).all() or not torch.isfinite(tokens[valid]).all():
            raise ValueError('Empty or nonfinite valid EEG tokens')
        # Sanitize padding before the MLP: NaN padding must not poison gradients.
        projected = self.connector(tokens.masked_fill(~valid[..., None], 0).float())
        embed = self.text.get_input_embeddings()
        prefix, suffix = embed(self.prefix_ids), embed(self.suffix_ids)
        inputs, mask, positions, last = assemble_embeddings(
            prefix, projected.to(prefix.dtype), suffix, valid)
        hidden = self.text(inputs_embeds=inputs, attention_mask=mask,
                           position_ids=positions, use_cache=False,
                           return_dict=True).last_hidden_state
        state = hidden[torch.arange(len(hidden), device=hidden.device), last]
        logits = F.linear(state.float(), self.head_weight)
        return {'logits': logits, 'loss': None if labels is None else F.cross_entropy(logits, labels)}
