import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithPast

from gia import GiaConfig, GiaModel


def random_patch_positions(size):
    t1 = torch.rand(*size, 1, 2)  # Create a random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t2 = torch.rand(*size, 1, 2)  # Create another random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t_min = torch.min(t1, t2)  # Element-wise minimum of t1 and t2
    t_max = torch.max(t1, t2)  # Element-wise maximum of t1 and t2
    return torch.cat((t_min, t_max), dim=-2)  # Concatenate t_min and t_max along the proper dimension


@pytest.mark.parametrize("test_mode", ["train", "eval"])
@pytest.mark.parametrize("input_mode", ["input_ids", "patches", "both"])
def test_gia_model(test_mode, input_mode):
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)
    if test_mode == "train":
        model.train()
    else:  # 'eval'
        model.eval()

    input_ids = torch.randint(0, 256, (2, 32), dtype=torch.long) if input_mode in ["input_ids", "both"] else None
    patches = (
        torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8) if input_mode in ["patches", "both"] else None
    )
    patch_positions = random_patch_positions((2, 32)) if input_mode in ["patches", "both"] else None
    input_types = torch.randint(0, 2, (2, 32), dtype=torch.long) if input_mode in ["both"] else None
    output = model(
        input_ids=input_ids,
        patches=patches,
        patch_positions=patch_positions,
        input_types=input_types,
        use_cache=True,
    )
    assert isinstance(output, CausalLMOutputWithPast)
    # Check the loss. If only patches are provided, the loss should be None.
    if input_mode in ["input_ids", "both"]:
        assert output.loss is not None
        assert output.loss.item() > 0.0
    else:  # 'patches'
        assert output.loss is None
    assert output.logits.shape == (2, 32, 30_000 + 1024 + 1)
    assert output.past_key_values is not None


def test_gia_model_local_positions():
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)
    input_ids = torch.randint(0, 256, (2, 32), dtype=torch.long)
    local_positions = torch.randint(0, 256, (2, 32), dtype=torch.long)
    output_wo_local_positions = model(input_ids=input_ids)
    output_w_local_positions = model(input_ids=input_ids, local_positions=local_positions)
    assert isinstance(output_wo_local_positions, CausalLMOutputWithPast)
    assert isinstance(output_w_local_positions, CausalLMOutputWithPast)
    assert not torch.allclose(output_wo_local_positions.logits, output_w_local_positions.logits)


def test_gia_model_attention_mask():
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)
    attention_mask = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    input_ids_1 = torch.randint(0, 256, (2, 32), dtype=torch.long)
    input_ids_2 = input_ids_1.clone()
    input_ids_2.masked_fill_(~attention_mask, 1.0)

    output_1 = model(input_ids=input_ids_1, attention_mask=attention_mask)
    output_2 = model(input_ids=input_ids_2, attention_mask=attention_mask)
    assert isinstance(output_1, CausalLMOutputWithPast)
    assert isinstance(output_2, CausalLMOutputWithPast)
    # Alter outside the attention mask should not change the output
    torch.testing.assert_close(output_1.logits, output_2.logits)

    output_3 = model(input_ids=input_ids_1)
    assert isinstance(output_3, CausalLMOutputWithPast)
    # Applying the attention mask should change the output
    assert not torch.allclose(output_1.logits, output_3.logits)


def test_gia_model_accelerate_compat():
    dataset = Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 256, (10, 32), dtype=torch.long).tolist(),
            "patches": torch.randint(0, 256, (10, 32, 4, 16, 16), dtype=torch.uint8).tolist(),
            "patch_positions": random_patch_positions((10, 32)).tolist(),
            "input_types": torch.randint(0, 2, (10, 32), dtype=torch.long).tolist(),
            "local_positions": torch.randint(0, 256, (10, 32), dtype=torch.long).tolist(),
            "attention_mask": torch.randint(0, 2, (10, 32), dtype=torch.bool).tolist(),
            "loss_mask": torch.randint(0, 2, (10, 32), dtype=torch.bool).tolist(),
        }
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=default_data_collator)
    model = GiaModel(GiaConfig())
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    for batch in dataloader:
        output = model(**batch)
        assert output.loss.item() > 0.0
        assert output.logits.shape == (2, 32, 30_000 + 1024 + 1)


def test_gia_model_trainer_compat():
    dataset = Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 256, (10, 32), dtype=torch.long).tolist(),
            "patches": torch.randint(0, 256, (10, 32, 4, 16, 16), dtype=torch.uint8).tolist(),
            "patch_positions": random_patch_positions((10, 32)).tolist(),
            "input_types": torch.randint(0, 2, (10, 32), dtype=torch.long).tolist(),
            "local_positions": torch.randint(0, 256, (10, 32), dtype=torch.long).tolist(),
            "attention_mask": torch.randint(0, 2, (10, 32), dtype=torch.bool).tolist(),
            "loss_mask": torch.randint(0, 2, (10, 32), dtype=torch.bool).tolist(),
        }
    )
    args = TrainingArguments(output_dir="./", report_to="none")
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=default_data_collator)
    trainer.train()
