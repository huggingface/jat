import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from gia.datasets import GiaDataCollator
from gia.eval.evaluator import Evaluator
from gia.model import GiaModel
from gia.processing import GiaProcessor


class LanguageModelingEvaluator(Evaluator):
    def _evaluate(self, model: GiaModel):
        model.eval()
        losses = []
        processor = GiaProcessor(self.args)
        dataset = load_dataset("gia-project/gia-dataset", self.task, split="test")
        dataset = dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=GiaDataCollator(), shuffle=True)
        for step, batch in enumerate(dataloader):
            for key, value in batch.items():
                batch[key] = value.to(self.args.device)
            outputs = model(**batch)
            losses.append(outputs.loss)
            if 0 < step == self.args.max_eval_steps:
                break

        loss = torch.mean(torch.stack(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return perplexity.item()
