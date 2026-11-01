"""One-way multi-positive losses over the SAME fixed candidate text bank.

Not original CLIP/SigLIP reproductions. No EEG batch member is a negative;
microbatch accumulation is therefore exact for the specified objectives.
"""
import torch
import torch.nn.functional as F


def positive_mask(labels, text_labels):
    if labels.ndim != 1 or text_labels.ndim != 1:
        raise ValueError('Labels must be one-dimensional')
    mask = labels[:, None] == text_labels[None, :]
    if not mask.any(1).all() or mask.all(1).any():
        raise ValueError('Every EEG needs positives AND known exclusive negatives')
    return mask


def per_sample_loss(logits, mask, kind):
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Invalid semantic positive mask')
    if not mask.any(1).all() or mask.all(1).any():
        raise ValueError('Missing positive or negative')
    logits = logits.float()
    if kind == 'infonce':
        # Average log probabilities, not log of the sum of positive mass:
        # every positive paraphrase must align (SupCon-style).
        return -(F.log_softmax(logits, dim=-1) * mask).sum(1) / mask.sum(1)
    if kind == 'sigmoid':
        pos = (F.softplus(-logits) * mask).sum(1) / mask.sum(1)
        neg = (F.softplus(logits) * ~mask).sum(1) / (~mask).sum(1)
        return .5 * (pos + neg)
    raise ValueError(kind)


def class_scores(similarities, text_labels, classes):
    """Mean cosine over templates, identical evaluation rule for both arms.

No sigmoid output is claimed to be a calibrated class probability.
"""
    if set(text_labels.tolist()) != set(range(classes)):
        raise ValueError('Candidate bank must contain exactly the allowed classes')
    return torch.stack([similarities[:, text_labels == c].mean(1) for c in range(classes)], 1)
