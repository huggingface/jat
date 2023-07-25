from typing import Any, Dict, List, Sized


class LocalPositionsAdder:
    r"""
    Add local positional information to a set of features.

    It takes a list of key groups, where each group is a list of keys. Each key corresponds to a set of features in a
    nested dictionary structure.

    Args:
        key_groups (`List[List[str]]`):
            List of key groups. Each key group is a list of keys to which positional information should be added.
            The key groups should be disjoint.

    Raises:
        `ValueError`: If the key groups are not disjoint or the size of batches does not match.

    Example:
        >>> from gia.processing import LocalPositionsAdder
        >>> adder = LocalPositionsAdder([["a"]])
        >>> ep_1 = [[0], [1, 2, 3]]
        >>> ep_2 = [[4, 5], [6], [7, 8, 9]]
        >>> input_dict = {"a": {"aa": [ep_1, ep_2]}}
        >>> output_dict = adder(input_dict)
        >>> output_dict["a"]["local_positions"]
        [[[0], [0, 1, 2]], [[0, 1], [0], [0, 1, 2]]]
    """

    def __init__(self, key_groups: List[List[str]]) -> None:
        # Check that the groups are disjoint
        for i in range(len(key_groups)):
            for j in range(i + 1, len(key_groups)):
                if set(key_groups[i]).intersection(set(key_groups[j])):
                    raise ValueError("The intersection between groups must be empty.")
        self.key_groups = key_groups

    def _get_size(self, data: Dict[str, Dict[str, List[List]]]) -> int:
        size = None
        for outer_key in data.keys():
            for inner_key, batch in data[outer_key].items():
                if size is None:  # check size is consistent
                    size = len(batch)
                elif len(batch) != size:
                    raise ValueError(f"Size for key '{outer_key}/{inner_key}' does not match the others.")
        return size

    def __call__(self, features: Dict[str, Dict[str, Sized]]) -> Any:
        for key_group in self.key_groups:
            group_features = {key: features[key] for key in key_group if key in features}  # order is preserved
            if group_features:
                self._add_local_positions_to_group(group_features)
        return features

    def _add_local_positions_to_group(self, features: Dict[str, Dict[str, Sized]]) -> None:
        batch_size = self._get_size(features)
        local_positions = {key: [] for key in features}
        for batch_idx in range(batch_size):
            data = {
                outer_key: {inner_key: features[outer_key][inner_key][batch_idx] for inner_key in features[outer_key]}
                for outer_key in features
            }
            ep_local_positions = self._calculate_positions_for_batch(data)
            for key in features:
                local_positions[key].append(ep_local_positions[key])

        for key in features:
            features[key]["local_positions"] = local_positions[key]

    def _calculate_positions_for_batch(self, features: Dict[str, Dict[str, Sized]]) -> Dict[str, List[List[int]]]:
        seq_len = self._get_size(features)
        local_positions = {key: [] for key in features}
        for timestep in range(seq_len):
            features_t = {
                outer_key: {inner_key: features[outer_key][inner_key][timestep] for inner_key in features[outer_key]}
                for outer_key in features
            }
            local_positions_t = self._calculate_positions_for_timestep(features_t)
            for key in features:
                local_positions[key].append(local_positions_t[key])
        return local_positions

    def _calculate_positions_for_timestep(self, features: Dict[str, Dict[str, Sized]]) -> Dict[str, List[int]]:
        local_positions = {key: [] for key in features}
        current_position = 0
        for key in features:
            features_mod = features[key]
            first_feature = next(iter(features_mod.values()))
            length = len(first_feature) if isinstance(first_feature, list) else 1
            local_positions[key].extend(list(range(current_position, current_position + length)))
            current_position += length
        return local_positions
