import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from jat.processing_jat import JatProcessor


@pytest.fixture
def processor():
    image_size = 224
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_input_names=["input_ids", "attention_mask"])
    processor = JatProcessor(tokenizer=tokenizer, image_processor=image_processor)
    return processor


def test_unbatched_text_encoding(processor):
    text = "The quick brown fox jumps over the lazy dog"
    encoding = processor(text=text, return_tensors="pt")
    assert "input_ids" in encoding
    assert encoding["input_ids"].shape == torch.Size([1, 9])


def test_unbatched_text_encoding_pad(processor):
    text = "The quick brown fox jumps over the lazy dog"
    encoding = processor(text=text, return_tensors="pt", padding="max_length", max_length=16)
    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert encoding["input_ids"].shape == torch.Size([1, 16])
    assert encoding["attention_mask"].shape == torch.Size([1, 16])
    assert torch.all(encoding["attention_mask"][:, :9] == 1)
    assert torch.all(encoding["attention_mask"][:, 9:] == 0)


def test_unbatched_text_encoding_truncate(processor):
    text = "The quick brown fox jumps over the lazy dog"
    encoding = processor(text=text, return_tensors="pt", truncation=True, max_length=8)
    assert "input_ids" in encoding
    assert encoding["input_ids"].shape == torch.Size([1, 8])


def test_unbatched_text_encoding_truncate_preserve(processor):
    text = "The quick brown fox jumps over the lazy dog"
    encoding = processor(text=text, return_tensors="pt", truncation="preserve", max_length=6, padding=True)
    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert encoding["input_ids"].shape == torch.Size([2, 6])
    assert encoding["attention_mask"].shape == torch.Size([2, 6])
    assert torch.all(encoding["attention_mask"] == torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]))


def test_image_encoding(processor):
    image = Image.new("RGB", (224, 224))
    encoding = processor(images=image, return_tensors="pt")
    assert "pixel_values" in encoding
    assert encoding["pixel_values"].shape == torch.Size([1, 3, 224, 224])


def test_image_encoding_batched(processor):
    images = [Image.new("RGB", (224, 224))]
    encoding = processor(images=images, return_tensors="pt")
    assert "pixel_values" in encoding
    assert encoding["pixel_values"].shape == torch.Size([1, 3, 224, 224])


def test_text_and_image_encoding(processor):
    text = "The quick brown fox jumps over the lazy dog"
    image = Image.new("RGB", (224, 224))
    encoding = processor(text=text, images=image, return_tensors="pt")
    assert "input_ids" in encoding
    assert "pixel_values" in encoding
    assert encoding["input_ids"].shape == torch.Size([1, 9])
    assert encoding["pixel_values"].shape == torch.Size([1, 3, 224, 224])


def test_batch_decode(processor):
    texts = ["The quick brown fox", "jumps over the lazy dog"]
    encoding = processor(text=texts)
    decoded_texts = processor.batch_decode(encoding["input_ids"])
    assert isinstance(decoded_texts, list)
    assert len(decoded_texts) == 2
    assert decoded_texts[0] == "The quick brown fox"
    assert decoded_texts[1] == "jumps over the lazy dog"


def test_decode(processor):
    text = "The quick brown fox jumps over the lazy dog"
    encoding = processor.tokenizer(text)
    decoded_text = processor.decode(encoding["input_ids"])
    assert decoded_text == text


def test_continuous_observations_encoding(processor):
    continuous_observations = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]  # 1 episode, 4 steps, 2 features
    encoding = processor(continuous_observations=continuous_observations, return_tensors="pt")
    assert "continuous_observations" in encoding
    assert encoding["continuous_observations"].shape == torch.Size([1, 4, 2])


def test_discrete_observations_encoding(processor):
    discrete_observations = [[[1, 2], [3, 4], [5, 6], [7, 8]]]  # 1 episode, 4 steps, 2 features
    encoding = processor(discrete_observations=discrete_observations, return_tensors="pt")
    assert "discrete_observations" in encoding
    assert encoding["discrete_observations"].shape == torch.Size([1, 4, 2])


def test_image_observations_encoding(processor):
    image_observations = [[Image.new("RGBA", (84, 84))]]
    encoding = processor(image_observations=image_observations, return_tensors="pt")
    assert "image_observations" in encoding
    assert encoding["image_observations"].shape == torch.Size([1, 1, 4, 84, 84])


def test_continuous_actions_encoding(processor):
    continuous_actions = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]  # 1 episode, 4 steps, 2 features
    encoding = processor(continuous_actions=continuous_actions, return_tensors="pt")
    assert "continuous_actions" in encoding
    assert encoding["continuous_actions"].shape == torch.Size([1, 4, 2])


def test_discrete_actions_encoding(processor):
    discrete_actions = [[1, 2, 3, 4]]  # 1 episode, 4 steps
    encoding = processor(discrete_actions=discrete_actions, return_tensors="pt")
    assert "discrete_actions" in encoding
    assert encoding["discrete_actions"].shape == torch.Size([1, 4])
