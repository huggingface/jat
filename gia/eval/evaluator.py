import torch

from gia.config.arguments import Arguments
from gia.model import GiaModel


class Evaluator:
    def __init__(self, args: Arguments, task_name: str) -> None:
        self.args = args
        self.task_name = task_name

    @torch.no_grad()
    def evaluate(self, model: GiaModel) -> float:
        return self._evaluate(model)

    def _evaluate(self, model: GiaModel) -> float:
        raise NotImplementedError
