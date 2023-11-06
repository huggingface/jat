import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from gato.datasets import GatoDataCollator
from gato.eval.evaluator import Evaluator
from gato.model import GatoModel
from gato.processing import GatoProcessor


class LanguageModelingEvaluator(Evaluator):
    def _evaluate(self, model: GatoModel) -> float:
        losses = []
        processor = GatoProcessor()
        dataset = load_dataset("gia-project/gia-dataset", self.task_name, split=self.args.test_split)
        dataset = dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=GatoDataCollator(), shuffle=True)
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
