from transformers import AutoTokenizer, CLIPImageProcessor

from gia2.configuration_gia2 import Gia2Config
from gia2.processor import GIAProcessor


# Small model
config = Gia2Config(
    attention_types=[[["global", "local"], 6]],
    hidden_size=768,
    num_heads=12,
    num_layers=12,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_input_names=["input_ids", "attention_mask"])
processor = GIAProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("gia-project/gia2-small")
processor.push_to_hub("gia-project/gia2-small")

# Medium model
config = Gia2Config(
    attention_types=[[["global", "local"], 12]],
    hidden_size=2048,
    num_heads=16,
    num_layers=24,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_input_names=["input_ids", "attention_mask"])
processor = GIAProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("gia-project/gia2-medium")
processor.push_to_hub("gia-project/gia2-medium")

# Large model
config = Gia2Config(
    attention_types=[[["global", "local"], 16]],
    hidden_size=2560,
    num_heads=20,
    num_layers=32,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_input_names=["input_ids", "attention_mask"])
processor = GIAProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("gia-project/gia2-large")
processor.push_to_hub("gia-project/gia2-large")
