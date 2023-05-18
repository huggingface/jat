import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import Trainer

from gia.config import Arguments
from gia.datasets import collate_fn, load_gia_dataset
from gia.datasets.utils import DatasetDict
from gia.model import GiaModel
from gia.processing import GiaProcessor


def random_patch_positions(size):  # Ensure that min < max
    t1 = torch.rand(*size, 1, 2)  # Create a random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t2 = torch.rand(*size, 1, 2)  # Create another random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t_min = torch.min(t1, t2)  # Element-wise minimum of t1 and t2
    t_max = torch.max(t1, t2)  # Element-wise maximum of t1 and t2
    return torch.cat((t_min, t_max), dim=-2)  # Concatenate t_min and t_max along the proper dimension


@pytest.mark.parametrize("use_accelerate", [True, False])
def test_gia_accelerate(use_accelerate):
    args = Arguments(task_names=["mujoco-ant"], embed_dim=48, seq_len=256, output_dir="./")
    dataset = load_gia_dataset(args.task_names)
    processor = GiaProcessor(args)
    dataset = DatasetDict(processor(**dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = GiaModel(args)
    if use_accelerate:
        accelerator = Accelerator()
        model, dataloader = accelerator.prepare(model, dataloader)
    batch = next(iter(dataloader))
    output = model(**batch)
    assert output.loss.item() > 0.0
    assert output.logits.shape == (args.batch_size, args.seq_len, 30_000 + 1024 + 1)


def test_model():
    module = GiaModel(Arguments(output_dir="./"))
    input_ids = torch.randint(0, 256, (2, 32))
    local_positions = torch.arange(32).repeat(2, 1)
    patches = torch.rand(2, 32, 4, 16, 16)
    patch_positions = random_patch_positions((2, 32))
    input_types = torch.randint(0, 2, (2, 32))
    loss_mask = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    attention_mask = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    output = module(input_ids, local_positions, patches, patch_positions, input_types, loss_mask, attention_mask)
    assert output.loss.item() > 0.0
    assert output.logits.shape == (2, 32, 30_000 + 1024 + 1)


def test_trainer():
    pytest.skip("test-trainer skipped until #57 is solved")
    args = Arguments(task_names=["mujoco-ant"], output_dir="./")
    dataset = load_gia_dataset(args.task_names)
    processor = GiaProcessor(args)
    dataset = Datatset(processor(**dataset))  # line skipped unti
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = GiaModel(args)
    trainer = Trainer(model=model, train_dataloader=dataloader)
    trainer.train()
