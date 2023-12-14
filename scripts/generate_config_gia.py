from transformers import AutoTokenizer, CLIPImageProcessor

from jat.configuration_jat import JatConfig
from jat.processing_jat import JatProcessor


# Small model
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_input_names=["input_ids", "attention_mask"])
config = JatConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,
    hidden_size=768,
    num_layers=12,
    attention_types=[[["global", "local"], 6]],
    num_heads=12,
    max_discrete_value=148 + 64,  # 148 (discrete obs from BabyAI) + 64 (max size of BabyAI's text observation)
    tokenizer_class=tokenizer.__class__.__name__,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer.model_max_length = config.max_position_embeddings
tokenizer.pad_token = tokenizer.eos_token
processor = JatProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("jat-project/jat-small")
processor.push_to_hub("jat-project/jat-small")

# Medium model
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_input_names=["input_ids", "attention_mask"])
config = JatConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=1024,
    hidden_size=2048,
    num_layers=24,
    attention_types=[[["global", "local"], 12]],
    num_heads=16,
    max_discrete_value=148 + 64,  # 148 (discrete obs from BabyAI) + 64 (max size of BabyAI's text observation)
    tokenizer_class=tokenizer.__class__.__name__,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer.model_max_length = config.max_position_embeddings
tokenizer.pad_token = tokenizer.eos_token
processor = JatProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("jat-project/jat-medium")
processor.push_to_hub("jat-project/jat-medium")

# Large model
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_input_names=["input_ids", "attention_mask"])
config = JatConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=2048,
    hidden_size=2560,
    num_layers=32,
    attention_types=[[["global", "local"], 16]],
    num_heads=20,
    max_discrete_value=148 + 64,  # 148 (discrete obs from BabyAI) + 64 (max size of BabyAI's text observation)
    tokenizer_class=tokenizer.__class__.__name__,
)
image_processor = CLIPImageProcessor(
    size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
)
tokenizer.model_max_length = config.max_position_embeddings
tokenizer.pad_token = tokenizer.eos_token
processor = JatProcessor(tokenizer=tokenizer, image_processor=image_processor)
config.push_to_hub("jat-project/jat-large")
processor.push_to_hub("jat-project/jat-large")
