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


# Purpose: Combine Algorithm 2 losses into a single optimization objective.
# Input: student prediction, teacher target, clean image, labels, clean mask, classifier, weights, and distance type.
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
) -> Dict[str, torch.Tensor]:
    loss_distill = distance_loss(student_prediction, teacher_target.detach(), distill_loss_type)
    loss_reconstruction = F.l1_loss(student_prediction.float(), clean_target.float())
    loss_identity = identity_loss(student_prediction, clean_target, clean_mask)
    loss_classifier = classifier_loss(classifier, student_prediction, labels)
    total = (
        weights.distill * loss_distill
        + weights.reconstruction * loss_reconstruction
        + weights.identity * loss_identity
        + weights.classifier * loss_classifier
    )
    return {
        "loss": total,
        "loss_distill": loss_distill.detach(),
        "loss_reconstruction": loss_reconstruction.detach(),
        "loss_identity": loss_identity.detach(),
        "loss_classifier": loss_classifier.detach(),
    }

