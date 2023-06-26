from gia.config.arguments import Arguments

from .base_language_modeling_evaluator import BaseLanguageModelingEvaluator


class ConceptualCaptionsEvaluator(BaseLanguageModelingEvaluator):
    def __init__(self, args: Arguments):
        self.task = "conceptual-captions"
        super().__init__(args)
