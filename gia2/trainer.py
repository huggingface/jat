from typing import Any, Callable, Dict, List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Sampler
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class MyTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | Module = None,
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        train_sampler: Sampler[int] | None = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.train_sampler = train_sampler

    def _get_train_sampler(self):
        return self.train_sampler if self.train_sampler is not None else super()._get_train_sampler()
