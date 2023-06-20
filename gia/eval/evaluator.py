import torch

from gia.model import GiaModel
from gia.config.arguments import Arguments


class Evaluator:
    def __init__(self, args: Arguments, task: str) -> None:
        self.args = args
        self.task = task

    @torch.no_grad()
    def evaluate(self, model: GiaModel):
        return self._evaluate(model)

    def _evaluate(self):
        raise NotImplementedError
