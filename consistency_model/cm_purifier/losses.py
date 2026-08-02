"""Loss functions for poison-aware consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .schedules import minus_one_to_one_to_zero_one


@dataclass(frozen=True)
class LossWeights:
    distill: float = 1.0
    reconstruction: float = 1.0
    identity: float = 1.0
    lpips: float = 0.0
    classifier: float = 0.0


# Purpose: Compute a configurable pixel distance between two tensors.
# Input: prediction tensor, target tensor, and loss type.
# Output: scalar loss tensor.
def distance_loss(prediction, target, loss_type: str = "l2"):
    if loss_type == "l2":
        return F.mse_loss(prediction.float(), target.float())
    if loss_type == "l1":
        return F.l1_loss(prediction.float(), target.float())
    if loss_type == "huber":
        return F.smooth_l1_loss(prediction.float(), target.float())
    raise ValueError(f"Unsupported loss_type: {loss_type}")


# Purpose: Compute identity loss only over clean identity samples.
# Input: prediction tensor, clean target tensor, and boolean clean mask.
# Output: scalar loss tensor, or zero when the batch has no clean samples.
def identity_loss(prediction, clean_target, clean_mask):
    if clean_mask is None or clean_mask.sum().item() == 0:
        return prediction.new_tensor(0.0)
    return F.l1_loss(prediction[clean_mask].float(), clean_target[clean_mask].float())


# Purpose: Compute optional frozen-classifier semantic preservation loss.
# Input: classifier module, predicted image tensor in [-1, 1], and labels.
# Output: scalar cross-entropy tensor, or zero when classifier is disabled.
def classifier_loss(classifier: Optional[torch.nn.Module], prediction, labels):
    if classifier is None:
        return prediction.new_tensor(0.0)
    logits = classifier(minus_one_to_one_to_zero_one(prediction))
    return F.cross_entropy(logits, labels)


# Purpose: Compute frozen-network LPIPS distance from prediction to the clean target.
# Input: optional LPIPS model, image tensors in [-1, 1], and LPIPS evaluation size.
# Output: scalar perceptual loss, or zero when LPIPS is disabled.
def lpips_loss(
    lpips_model: Optional[torch.nn.Module],
    prediction,
    clean_target,
    image_size: int = 64,
):
    if lpips_model is None:
        return prediction.new_tensor(0.0)
    if image_size <= 0:
        raise ValueError(f"LPIPS image_size must be positive, got {image_size}")
    if prediction.ndim != 4 or clean_target.ndim != 4:
        raise ValueError("LPIPS inputs must be BCHW tensors")
    if (
        prediction.shape[0] != clean_target.shape[0]
        or prediction.shape[1] != 3
        or clean_target.shape[1] != 3
    ):
        raise ValueError("LPIPS prediction and target must have equal batch size and three channels")

    prediction = prediction.float().clamp(-1.0, 1.0)
    clean_target = clean_target.float().clamp(-1.0, 1.0)
    if prediction.shape[-2:] != (image_size, image_size):
        prediction = F.interpolate(
            prediction,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if clean_target.shape[-2:] != (image_size, image_size):
        clean_target = F.interpolate(
            clean_target,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return lpips_model(prediction, clean_target, normalize=False).mean()


# Purpose: Compute a bounded linear warmup multiplier for an auxiliary loss.
# Input: one-based optimization step and number of warmup steps.
# Output: scalar multiplier in [0, 1].
def linear_warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(max(float(step) / float(warmup_steps), 0.0), 1.0)


# Purpose: Combine Algorithm 2 losses into a single optimization objective.
# Input: predictions, clean target, optional frozen models, weights, LPIPS settings, and distance type.
# Output: dictionary containing total loss and individual detached metrics.
def compute_loss_dict(
    student_prediction,
    teacher_target,
    clean_target,
    labels,
    clean_mask,
    classifier: Optional[torch.nn.Module],
    weights: LossWeights,
    distill_loss_type: str = "l2",
    lpips_model: Optional[torch.nn.Module] = None,
    lpips_image_size: int = 64,
    lpips_warmup_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= lpips_warmup_factor <= 1.0:
        raise ValueError(f"lpips_warmup_factor must be in [0, 1], got {lpips_warmup_factor}")
    if weights.lpips > 0.0 and lpips_model is None:
        raise ValueError("LPIPS weight is positive but no LPIPS model was provided")

    loss_distill = distance_loss(student_prediction, teacher_target.detach(), distill_loss_type)
    loss_reconstruction = F.l1_loss(student_prediction.float(), clean_target.float())
    loss_identity = identity_loss(student_prediction, clean_target, clean_mask)
    loss_lpips = lpips_loss(lpips_model, student_prediction, clean_target, image_size=lpips_image_size)
    loss_classifier = classifier_loss(classifier, student_prediction, labels)
    effective_lpips_weight = weights.lpips * lpips_warmup_factor
    weighted_lpips = effective_lpips_weight * loss_lpips
    total = (
        weights.distill * loss_distill
        + weights.reconstruction * loss_reconstruction
        + weights.identity * loss_identity
        + weighted_lpips
        + weights.classifier * loss_classifier
    )
    return {
        "loss": total,
        "loss_distill": loss_distill.detach(),
        "loss_reconstruction": loss_reconstruction.detach(),
        "loss_identity": loss_identity.detach(),
        "loss_lpips": loss_lpips.detach(),
        "loss_lpips_weighted": weighted_lpips.detach(),
        "lpips_effective_weight": student_prediction.new_tensor(effective_lpips_weight),
        "loss_classifier": loss_classifier.detach(),
    }
