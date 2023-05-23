import torch

from gia.config.arguments import Arguments
from gia.datasets import load_task_dataset, collate_fn
from gia.model.gia_model import GiaModel

from torch.utils.data import DataLoader

from gia.eval.evaluator import Evaluator

class BaseLanguageModelingEvaluator(Evaluator):
    def __init__(self, args: Arguments):
        self.args = args

    def evaluate(self, model: GiaModel):
        model.eval()
        losses = []
        dataset = load_task_dataset(self.task, split="test")
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=collate_fn, shuffle=True)
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            losses.append(outputs.loss)
            if 0 < step == self.args.max_eval_steps:
                break

        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return perplexity.item()