from gia.config.arguments import Arguments

from .base_language_modeling_evaluator import BaseLanguageModelingEvaluator


class OKVQAEvaluator(BaseLanguageModelingEvaluator):
    def __init__(self, args: Arguments):
        self.task = "ok-vqa"
        super().__init__(args)
