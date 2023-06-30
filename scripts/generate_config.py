#!/usr/bin/env python3
"""Generate config for GIA models"""

from gia import GiaConfig, GiaModel


# gia-80m
config = GiaConfig(num_layers=8, num_heads=24, hidden_size=768)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-80m")

# gia-387m
config = GiaConfig(num_layers=12, num_heads=12, hidden_size=1536)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-387m")

# gia-1.27b
config = GiaConfig(num_layers=24, num_heads=16, hidden_size=2048)
model = GiaModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gia-1.27b")
