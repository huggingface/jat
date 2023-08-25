#!/usr/bin/env python3
"""Generate config for GIA models and processors, and push them to the hub."""

from gia import GiaConfig, GiaModel
from gia.processing import GiaProcessor


# gia-80m
config = GiaConfig(seq_len=1024, patch_size=16, num_layers=8, num_heads=24, hidden_size=768)
processor = GiaProcessor(patch_size=16, seq_len=1024)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-80m")
processor.push_to_hub("gia-project/gia-80m")

# gia-387m
config = GiaConfig(seq_len=1024, patch_size=16, num_layers=12, num_heads=12, hidden_size=1536)
processor = GiaProcessor(patch_size=16, seq_len=1024)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-387m")
processor.push_to_hub("gia-project/gia-387m")

# gia-1.27b
config = GiaConfig(seq_len=1024, patch_size=16, num_layers=24, num_heads=16, hidden_size=2048)
processor = GiaProcessor(patch_size=16, seq_len=1024)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-1.27b")
processor.push_to_hub("gia-project/gia-1.27b")
