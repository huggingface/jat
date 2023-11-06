import torch

from gato.config.arguments import Arguments
from gato.model import GatoModel


class Evaluator:
    def __init__(self, args: Arguments, task_name: str) -> None:
        self.args = args
        self.task_name = task_name

    @torch.no_grad()
    def evaluate(self, model: GatoModel) -> float:
        return self._evaluate(model)

    def _evaluate(self, model: GatoModel) -> float:
        raise NotImplementedError
