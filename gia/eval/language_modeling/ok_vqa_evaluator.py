from .base_language_modeling_evaluator import BaseLanguageModelingEvaluator
from gia.config.arguments import Arguments


class OKVQAEvaluator(BaseLanguageModelingEvaluator):
    def __init__(self, args: Arguments):
        self.task = "ok-vqa"
        super().__init__(args)
