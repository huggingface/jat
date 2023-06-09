from typing import Any, Dict, List

import numpy as np
import torch


class GiaDataCollator:
    """
    A callable data collator used for GIA model.

    - ensure all sequences within a batch have the same length
    - pad None sequences with a sequence of pad values for each feature
    - pad None features with the corresponding pad value for each feature
    - convert all features to PyTorch tensors

    Class Attributes:
        pad_values (dict): A mapping from feature keys to their associated pad values.
        dtypes (dict): A mapping from feature keys to their associated PyTorch dtype.

    Example:
        >>> collator = GiaDataCollator()
        >>> features = [
        ...     {
        ...         "input_ids": [1, 2, None],
        ...         "local_positions": [7, 8, 9],
        ...     },
        ...     {
        ...         "input_ids": [4, 5, 6],
        ...         "local_positions": [None, 11, 12],
        ...     },
        ... ]
        >>> collator(features)
        {
            "input_ids": tensor([[1, 2, 0], [4, 5, 6]]),
            "local_positions": tensor([[7, 8, 9], [-1, 11, 12]]),
        }
    """

    pad_values = {
        "input_ids": 0,
        "patches": np.zeros((4, 16, 16), dtype=np.uint8).tolist(),
        "patch_positions": np.zeros((2, 2), dtype=np.float32).tolist(),
        "input_types": -1,  # Shouldn't be used, so put an wrong value
        "local_positions": -1,
        "loss_mask": True,
        "attention_mask": True,
    }
    dtypes = {
        "input_ids": torch.int64,
        "patches": torch.uint8,
        "patch_positions": torch.float32,
        "input_types": torch.int64,
        "local_positions": torch.int64,
        "loss_mask": torch.bool,
        "attention_mask": torch.bool,
    }

    def _to_tensor(self, features: List[List[Any]], dtype: torch.dtype) -> torch.Tensor:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # When the input is a list of list of numpy.ndarrays, first convert it to a single numpy.ndarray, then
        # convert it to a tensor.
        if isinstance(features[0][0], np.ndarray):
            features = np.array(features)
        return torch.tensor(features, dtype=dtype)

    def __call__(self, features: List[Dict[str, List[Any]]]) -> Dict[str, torch.Tensor]:
        keys = features[0].keys()  # they should all have the same keys
        if any(key not in self.pad_values for key in keys):
            raise KeyError(f"Found unexpected keys: {set(keys) - set(self.pad_values.keys())}")
        output = {}
        length = None
        for key in keys:
            feature_batch = [f[key] for f in features]  # list of lists of features
            # If a feature is absent in all sequences in the batch, we just ignore it.
            if all(feature_sequence is None for feature_sequence in feature_batch):
                continue
            # At this point, we have a list of lists (sequence) of features.
            # E.g. [[1, 2, 3], None, [4, 5, 6]]
            # If a feature is absent in a sequence, it is None. When it's the case, we replace the whole sequence
            # with a list of the pad value for this feature.
            # We check that all sequences have the same length.
            lengths = [len(x) for x in feature_batch if x is not None]
            if len(set(lengths)) != 1:
                raise ValueError(f"Sequences for feature {key} should all have the same length, got lengths {lengths}")
            if length is None:
                length = lengths[0]
            else:
                if length != lengths[0]:
                    raise ValueError(
                        f"Sequences length should be the same for all features, got {length} and {lengths[0]}"
                    )
            for batch_idx in range(len(feature_batch)):
                # Replace sequence==None by [None, None, ...]
                if feature_batch[batch_idx] is None:
                    feature_batch[batch_idx] = [None] * length
                for feature_idx, feature in enumerate(feature_batch[batch_idx]):
                    # Replace None by pad_value
                    if feature is None:
                        feature_batch[batch_idx][feature_idx] = self.pad_values[key]
            output[key] = self._to_tensor(feature_batch, dtype=self.dtypes[key])
        return output
