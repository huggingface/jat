import numpy as np
import torch
from typing import List, Any, Dict


class GIADataCollator:
    def __init__(self) -> None:
        self.pad_values = {
            "input_ids": 0,
            "patches": np.zeros((4, 16, 16), dtype=np.uint8).tolist(),
            "patch_positions": np.zeros((2, 2), dtype=np.float32).tolist(),
            "input_types": -1,  # Shouldn't be used, so put an wrong value
            "local_positions": -1,
            "loss_mask": True,
            "attention_mask": True,
        }
        self.dtypes = {
            "input_ids": torch.int64,
            "patches": torch.uint8,
            "patch_positions": torch.float32,
            "input_types": torch.int64,
            "local_positions": torch.int64,
            "loss_mask": torch.bool,
            "attention_mask": torch.bool,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Any:
        keys = features[0].keys()  # they should all have the same keys
        output = {}
        for key in keys:
            feature_batch = [f[key] for f in features]
            if any(feature_sequence is not None for feature_sequence in feature_batch):
                # Get the length of the first non-None element
                max_len = max(len(x) for x in feature_batch if x is not None)
                for batch_idx in range(len(feature_batch)):
                    if feature_batch[batch_idx] is None:
                        feature_batch[batch_idx] = [None] * max_len
                    for feature_idx, feature in enumerate(feature_batch[batch_idx]):
                        if feature is None:
                            pad_value = self.pad_values[key]
                            feature_batch[batch_idx][feature_idx] = pad_value
                output[key] = torch.tensor(feature_batch, dtype=self.dtypes[key])
        return output
