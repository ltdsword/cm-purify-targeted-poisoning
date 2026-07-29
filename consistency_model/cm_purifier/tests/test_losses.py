"""Tests for the LPIPS-augmented Algorithm 2 objective."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from consistency_model.cm_purifier.losses import LossWeights, compute_loss_dict, linear_warmup_factor


class FakeLPIPS(torch.nn.Module):
    """Small differentiable stand-in that follows the LPIPS call interface."""

    # Purpose: Initialize one frozen parameter for gradient-isolation checks.
    # Input: no arguments beyond the module instance.
    # Output: initialized fake perceptual module.
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    # Purpose: Produce a per-image perceptual distance with LPIPS-compatible shape.
    # Input: prediction, target, and unused normalization flag.
    # Output: Bx1x1x1 squared-distance tensor.
    def forward(self, prediction, target, normalize=False):
        del normalize
        return self.scale * (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)


class LossTests(unittest.TestCase):
    # Purpose: Confirm disabling LPIPS preserves the original weighted objective exactly.
    # Input: synthetic prediction, teacher, clean target, labels, and clean mask.
    # Output: assertion that computed and manually assembled losses match.
    def test_disabled_lpips_preserves_original_objective(self):
        prediction = torch.tensor([[[[0.25]]], [[[0.50]]]], requires_grad=True)
        teacher = torch.tensor([[[[0.0]]], [[[1.0]]]])
        clean = torch.tensor([[[[0.0]]], [[[0.25]]]])
        labels = torch.tensor([0, 1])
        clean_mask = torch.tensor([True, False])
        weights = LossWeights(distill=1.5, reconstruction=2.0, identity=0.5)

        losses = compute_loss_dict(
            prediction,
            teacher,
            clean,
            labels,
            clean_mask,
            classifier=None,
            weights=weights,
        )
        expected = (
            weights.distill * F.mse_loss(prediction, teacher)
            + weights.reconstruction * F.l1_loss(prediction, clean)
            + weights.identity * F.l1_loss(prediction[clean_mask], clean[clean_mask])
        )

        self.assertTrue(torch.allclose(losses["loss"], expected))
        self.assertEqual(float(losses["loss_lpips"]), 0.0)
        self.assertEqual(float(losses["loss_lpips_weighted"]), 0.0)

    # Purpose: Confirm LPIPS gradients reach predictions but not the frozen feature network.
    # Input: random student prediction and clean target tensors.
    # Output: assertions over student and LPIPS parameter gradients.
    def test_lpips_gradient_reaches_student_only(self):
        torch.manual_seed(7)
        prediction = torch.randn(2, 3, 32, 32, requires_grad=True)
        clean = torch.randn_like(prediction)
        fake_lpips = FakeLPIPS()
        weights = LossWeights(distill=0.0, reconstruction=0.0, identity=0.0, lpips=0.1)

        losses = compute_loss_dict(
            prediction,
            teacher_target=clean,
            clean_target=clean,
            labels=torch.zeros(2, dtype=torch.long),
            clean_mask=torch.zeros(2, dtype=torch.bool),
            classifier=None,
            weights=weights,
            lpips_model=fake_lpips,
            lpips_image_size=64,
            lpips_warmup_factor=1.0,
        )
        losses["loss"].backward()

        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertIsNone(fake_lpips.scale.grad)
        self.assertTrue(torch.allclose(losses["loss_lpips_weighted"], 0.1 * losses["loss_lpips"]))

    # Purpose: Confirm LPIPS warmup is bounded and resume-safe at representative steps.
    # Input: steps before, during, and after a 2,000-step warmup.
    # Output: exact expected linear factors.
    def test_linear_warmup_factor(self):
        self.assertEqual(linear_warmup_factor(0, 2000), 0.0)
        self.assertEqual(linear_warmup_factor(1000, 2000), 0.5)
        self.assertEqual(linear_warmup_factor(2000, 2000), 1.0)
        self.assertEqual(linear_warmup_factor(4000, 2000), 1.0)
        self.assertEqual(linear_warmup_factor(1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
