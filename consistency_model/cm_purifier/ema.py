"""EMA teacher helpers for consistency distillation."""

from __future__ import annotations

import copy


# Purpose: Make a detached teacher copy from a student model.
# Input: trainable student torch module.
# Output: eval-mode teacher module with gradients disabled.
def create_ema_model(student):
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


# Purpose: Update EMA teacher parameters from the current student parameters.
# Input: teacher module, student module, and EMA decay coefficient.
# Output: none; teacher parameters are updated in place.
def update_ema(teacher, student, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if teacher_value.dtype.is_floating_point:
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


# Purpose: Copy student parameters into teacher exactly.
# Input: teacher module and student module.
# Output: none; teacher parameters are overwritten in place.
def sync_ema(teacher, student) -> None:
    teacher.load_state_dict(student.state_dict())


# Purpose: Extract EMA state for checkpointing.
# Input: teacher module.
# Output: serializable state dictionary.
def ema_state_dict(teacher):
    return teacher.state_dict()


# Purpose: Load EMA state from a checkpoint into a teacher module.
# Input: teacher module and serialized state dictionary.
# Output: none; teacher state is restored in place.
def load_ema_state_dict(teacher, state_dict) -> None:
    teacher.load_state_dict(state_dict)

