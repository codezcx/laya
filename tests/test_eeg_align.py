"""Small CPU checks for variable-length prompt composition (no checkpoint loading)."""
import unittest
import torch
from eeg_align.model import assemble_embeddings


class PromptAssemblyTest(unittest.TestCase):
    def test_padding_and_gradient(self):
        prefix = torch.ones(2, 4)
        suffix = torch.ones(3, 4) * 2
        eeg = torch.randn(2, 5, 4, requires_grad=True)
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        embeds, mask, positions, last = assemble_embeddings(prefix, eeg, suffix, valid)
        self.assertEqual(last.tolist(), [9, 6])
        self.assertEqual(mask.sum(1).tolist(), [10, 7])
        torch.testing.assert_close(embeds[1, 4:7], suffix)
        self.assertEqual(positions[1, :7].tolist(), list(range(7)))
        embeds.sum().backward()
        self.assertTrue((eeg.grad[1, 2:] == 0).all())
        self.assertTrue((eeg.grad[1, :2] == 1).all())


if __name__ == "__main__":
    unittest.main()
