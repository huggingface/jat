from .base_language_modeling_evaluator import BaseLanguageModelingEvaluator
from gia.config.arguments import Arguments


class OscarEvaluator(BaseLanguageModelingEvaluator):
    def __init__(self, args: Arguments):
        self.task = "oscar"
        super().__init__(args)