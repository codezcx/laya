"""Task-specific readouts with matched initialization for the shared-head control."""
import copy
import torch


class SplitContrastHeads(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        # Copying consumes no RNG: preserve encoder dropout and sampling streams.
        self.mi = initial
        self.auxiliary = copy.deepcopy(initial)

    def forward(self, features, tasks):
        if len(tasks) != len(features) or any(t not in (0, 1) for t in tasks):
            raise ValueError('Invalid task routing')
        return torch.cat([
            (self.mi if task == 0 else self.auxiliary)(row[None])
            for row, task in zip(features, tasks)
        ], dim=0)
