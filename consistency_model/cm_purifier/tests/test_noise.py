"""Tests for private and reproducible Algorithm 3 noise generation."""

from __future__ import annotations

import unittest

import torch

from consistency_model.cm_purifier.infer import make_noise_generator, timestep_statistics
from purify.purifier import CMPurifier


class EchoModel(torch.nn.Module):
    """Return the noised input so tests can recover the sampled epsilon."""

    # Purpose: Expose the purifier's noised tensor without denoising it.
    # Input: noised image, timestep, and unused schedules.
    # Output: unchanged noised image tensor.
    def forward(self, x_t, timesteps, alpha_schedule, sigma_schedule):
        del timesteps, alpha_schedule, sigma_schedule
        return x_t


class NoiseTests(unittest.TestCase):
    # Purpose: Confirm equal private seeds reproduce noise despite global RNG activity.
    # Input: two private generators and unrelated draws from the global generator.
    # Output: identical private random tensors.
    def test_private_generator_is_independent_from_global_rng(self):
        first, first_seed = make_noise_generator(torch.device("cpu"), seed=2026)
        torch.manual_seed(999)
        torch.randn(100)
        second, second_seed = make_noise_generator(torch.device("cpu"), seed=2026)

        first_noise = torch.randn((4, 3, 8, 8), generator=first)
        torch.randn(250)
        second_noise = torch.randn((4, 3, 8, 8), generator=second)

        self.assertEqual(first_seed, second_seed)
        self.assertTrue(torch.equal(first_noise, second_noise))

    # Purpose: Confirm two timestep runs apply different schedules to the same private noise.
    # Input: zero images, equal private seeds, and two different alpha/sigma indices.
    # Output: equal recovered epsilon tensors despite unrelated global RNG draws.
    def test_timestep_runs_share_noise_sequence(self):
        device = torch.device("cpu")
        alpha = torch.tensor([0.9, 0.6])
        sigma = torch.tensor([0.1, 0.8])
        first_generator, first_seed = make_noise_generator(device, seed=2026)
        second_generator, second_seed = make_noise_generator(device, seed=2026)
        common = {
            "model": EchoModel(),
            "alpha_schedule": alpha,
            "sigma_schedule": sigma,
            "train_args": {},
            "image_size": 8,
            "device": device,
        }
        first = CMPurifier(
            **common,
            t_star=0,
            noise_seed=first_seed,
            noise_generator=first_generator,
        )
        second = CMPurifier(
            **common,
            t_star=1,
            noise_seed=second_seed,
            noise_generator=second_generator,
        )
        images = torch.zeros(2, 3, 8, 8)

        first_x_t = first.purify_tensor(images, output_range="minus_one_to_one")
        torch.manual_seed(55)
        torch.randn(300)
        second_x_t = second.purify_tensor(images, output_range="minus_one_to_one")

        first_epsilon = first_x_t / sigma[0]
        second_epsilon = second_x_t / sigma[1]
        self.assertTrue(torch.allclose(first_epsilon, second_epsilon, atol=1e-6))

    # Purpose: Confirm timestep diagnostics report the exact schedule coefficients and SNR.
    # Input: small alpha/sigma schedules and one selected timestep.
    # Output: expected alpha, sigma, and squared signal-to-noise ratio.
    def test_timestep_statistics(self):
        alpha = torch.tensor([0.9, 0.6])
        sigma = torch.tensor([0.1, 0.8])
        statistics = timestep_statistics(1, alpha, sigma)

        self.assertAlmostEqual(statistics["alpha"], 0.6, places=6)
        self.assertAlmostEqual(statistics["sigma"], 0.8, places=6)
        self.assertAlmostEqual(statistics["snr"], (0.6**2) / (0.8**2), places=6)


if __name__ == "__main__":
    unittest.main()
