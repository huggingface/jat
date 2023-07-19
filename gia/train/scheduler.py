import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def _get_cosine_schedule_with_linear_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int = 15_000, num_decay_steps: int = 1_000_000, final_value: float = 0.1
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif num_warmup_steps <= current_step < num_warmup_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps))
        return final_value + (1.0 - final_value) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif num_warmup_steps + num_decay_steps <= current_step:
        return final_value


def get_cosine_schedule_with_linear_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 15_000,
    num_decay_steps: int = 1_000_000,
    final_value: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that warms up linearly, then decays using the cosine schedule.

    Args:
        optimizer (`Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`, optional):
            The number of steps for the warmup phase. Defaults to 15_000.
        num_decay_steps (`int`, optional):
            The number of steps for the decay phase. Defaults to 1_000_000.
        final_value (`float`, optional):
            The final learning rate (as a fraction of the initial learning rate). Defaults to 0.1.
        last_epoch (`int`, optional):
            The index of the last epoch. Defaults to -1.
    """

    def lr_lambda(current_step: int) -> float:
        return _get_cosine_schedule_with_linear_warmup_lr_lambda(
            current_step, num_warmup_steps=num_warmup_steps, num_decay_steps=num_decay_steps, final_value=final_value
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
