from typing import Any, Dict, List, Optional

import numpy as np


def _is_episode(sample_data: Dict[str, Dict[str, Any]]) -> bool:
    """
    Determines if the keys of the sample_data dictionary follow the episode format. Keys can be either in
    ["image", "text"] or in ["image_observations", "text_observations", "discrete_observations",
    "continuous_observations", "discrete_actions", "continuous_actions"]. They can't be in both.

    Args:
        sample_data (Dict[str, Dict[str, Any]]): A dictionary containing sample data with specific keys.

    Returns:
        bool: True if the keys follow the episode format, False otherwise.

    Raises:
        ValueError: If the keys are mixed and do not follow the expected format.
    """
    key_set = set(sample_data.keys())
    epsiode_keys = set(
        [
            "image_observations",
            "text_observations",
            "discrete_observations",
            "continuous_observations",
            "discrete_actions",
            "continuous_actions",
        ]
    )
    standalone_keys = set(["image", "text"])

    if key_set.issubset(epsiode_keys):
        return True
    elif key_set.issubset(standalone_keys):
        return False
    else:
        raise ValueError("Keys are mixed and do not follow the expected format.")


def _extract_idx_element(
    nested_dict: Dict[str, Dict[str, List[Optional[Any]]]], index: int
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the index-th element from each sequence in the nested dictionary.

    Args:
        nested_dict (Dict[str, Dict[str, List[Optional[Any]]]]): A nested dictionary where the innermost
            values are lists of elements, which can be None or any other type.
        index (int): The index of the element to be extracted from each list in the nested dictionary.

    Returns:
        Dict[str, Dict[str, Any]]: A nested dictionary with the same structure as the input dictionary,
            containing only the extracted index-th elements, excluding None values.
    """
    output: Dict[str, Dict[str, Any]] = {}
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, inner_list in inner_dict.items():
            element = inner_list[index]
            if element is not None:
                output.setdefault(outer_key, {})[inner_key] = element
    return output


def _append(batch_data: Dict[str, Dict[str, Any]], processed_data: Dict[str, List[Any]]) -> None:
    # get the number of elements
    for key in batch_data:
        if not isinstance(batch_data[key], list):
            batch_data[key] = [batch_data[key]]
    num_elements = len(next(iter(batch_data.values())))
    input_ids = batch_data.get("input_ids", [0] * num_elements)
    patches = batch_data.get("patches", np.zeros((num_elements, 3, 16, 16)).tolist())
    positions = batch_data.get("positions", [[[0, 0], [0, 0]]] * num_elements)
    if "input_ids" in batch_data:
        input_type = [0] * num_elements
    elif "patches" in batch_data and "positions" in batch_data:
        input_type = [1] * num_elements
    else:
        raise ValueError("Batch data must contain either input_ids or patches and positions.")

    processed_data["input_ids"].extend(input_ids)
    processed_data["patches"].extend(patches)
    processed_data["positions"].extend(positions)
    processed_data["input_type"].extend(input_type)


def _interleave_episode(episode_data: Dict[str, Dict[str, Any]]) -> dict:
    """
    Unrolls a single episode of data into a dictionary of lists of data.

    Args:
        episode_data (Dict[str, Any]): The episode of data to unroll. First level keys must be "input_ids",
            "patches", "positions". Second level keys must be "text_observations", "image_observations",
            "discrete_observations", "continuous_observations", "discrete_actions", and "continuous_actions".

    Returns:
        dict: The interleaved episode episode of data.
            keys are "input_ids", "patches", "positions", "input_type".

    Example:
        >>> episode_data = {"discrete_observations": {"input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        ...                 "image_observations": {"patches": [[P1, P2], [P3, P4], [P5, P6]],
        ...                                        "positions": [[LEFT, RIGHT], [LEFT, RIGHT], [LEFT, RIGHT]]}}
        >>> _interleave_episode(episode_data)
        {'input_ids': [0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9],
        'patches': [P1, P2, P0, P0, P0, P3, P4, P0, P0, P0, P5, P6],
        'positions': [LEFT, RIGHT, POS0, POS0, POS0, LEFT, RIGHT, POS0, POS0, POS0, LEFT, RIGHT]}
        'input_type': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]}

    Where P0 = zeros(shape=(3, 16, 16)) and  POS0 = [[0, 0], [0, 0]] are placeholders for respectively patches
    and positions. (Similar to 0 for input_ids)
    """
    output = {"input_ids": [], "patches": [], "positions": [], "input_type": []}

    # Get the length of each sequence in the d and make sure they are all equal
    sequence_lengths = [len(in_val) for out_val in episode_data.values() for in_val in out_val.values()]
    common_length = sequence_lengths[0]
    assert all(common_length == length for length in sequence_lengths)
    for t in range(common_length):
        # Extract the current element from each sequence in the d
        timestep_data = _extract_idx_element(episode_data, t)
        for data_key in [
            "text_observations",
            "image_observations",
            "discrete_observations",
            "continuous_observations",
            "discrete_actions",
            "continuous_actions",
        ]:
            if data_key in timestep_data:
                _append(timestep_data[data_key], output)

    return output


def _interleave_standalone(data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Optional[Any]]]:
    output = {"input_ids": [], "patches": [], "positions": [], "input_type": []}
    if "image" in data:
        _append(data["image"], output)
    if "text" in data:
        _append(data["text"], output)
    return output


def _get_batch_size(batch_data: Dict[str, Dict[str, Any]]) -> int:
    first_mod = next(iter(batch_data.values()))
    first_val = next(iter(first_mod.values()))
    return len(first_val)


def interleave_batch(batch_data: Dict[str, Any]) -> Dict[str, List[Any]]:
    batch_size = _get_batch_size(batch_data)
    output = {"input_ids": [], "patches": [], "positions": [], "input_type": []}

    for batch_idx in range(batch_size):
        sample_data = _extract_idx_element(batch_data, batch_idx)
        if _is_episode(sample_data):
            x = _interleave_episode(sample_data)
        else:
            x = _interleave_standalone(sample_data)

        for key in x:
            output[key].append(x[key])

    return output
