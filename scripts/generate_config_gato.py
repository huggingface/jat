#!/usr/bin/env python3
"""Generate config for GATO models and processors, and push them to the hub."""

from gato import GatoConfig, GatoModel
from gato.processing import GatoProcessor


# gato-80m
config = GatoConfig(seq_len=1024, patch_size=16, num_layers=8, num_heads=24, hidden_size=768)
processor = GatoProcessor(patch_size=16, seq_len=1024)
model = GatoModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gato-80m", private=True)
processor.push_to_hub("gia-project/gato-80m", private=True)

# gato-387m
config = GatoConfig(seq_len=1024, patch_size=16, num_layers=12, num_heads=12, hidden_size=1536)
processor = GatoProcessor(patch_size=16, seq_len=1024)
model = GatoModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gato-387m", private=True)
processor.push_to_hub("gia-project/gato-387m", private=True)

# gato-1.27b
config = GatoConfig(seq_len=1024, patch_size=16, num_layers=24, num_heads=16, hidden_size=2048)
processor = GatoProcessor(patch_size=16, seq_len=1024)
model = GatoModel(config)
num_params = sum(p.numel() for p in model.causal_lm_model.parameters() if p.requires_grad) // 1_000_000
print(f"Number of parameters: {num_params}m")
config.push_to_hub("gia-project/gato-1.27b", private=True)
processor.push_to_hub("gia-project/gato-1.27b", private=True)
