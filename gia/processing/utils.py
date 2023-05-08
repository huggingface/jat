from typing import Any, Dict, List, Optional


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
    num_elements = len(next(iter(batch_data.values())))
    for k in range(num_elements):
        for key in processed_data:
            if key in batch_data:
                processed_data[key].append(batch_data[key][k])
            else:
                processed_data[key].append(None)


def _interleave_episode(episode_data: Dict[str, Dict[str, Any]]) -> dict:
    """
    Unrolls a single episode of data into a dictionary of lists of data.

    Args:
        d (Dict[str, Any]): The episode of data to unroll. First level keys must be "input_ids", "patches", and
            "positions". Second level keys must be "text_observations", "image_observations", "discrete_observations",
            "continuous_observations", "discrete_actions", and "continuous_actions".

    Returns:
        dict: The unrolled episode of data.

    Example:
    >>> episode_data = {"discrete_observations": {"input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    ...                 "image_observations": {"patches": [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4], [PATCH_5, PATCH_6]],
    ...                                        "positions": [[LEFT, RIGHT], [LEFT, RIGHT], [LEFT, RIGHT]]}}
    >>> _unroll_single_episode(episode_data)
    {'input_ids': [None, None, 1, 2, 3, None, None, 4, 5, 6, None, None, 7, 8, 9],
     'patches': [PATCH_1, PATCH_2, None, None, None, PATCH_3, PATCH_4, None, None, None, PATCH_5, PATCH_6],
     'positions': [LEFT, RIGHT, None, None, None, LEFT, RIGHT, None, None, None, LEFT, RIGHT]}
    """
    processed_data = {key: [] for key in ["input_ids", "patches", "positions"]}

    # Get the length of each sequence in the d and make sure they are all equal
    sequence_lengths = [len(in_val) for out_val in episode_data.values() for in_val in out_val.values()]

    if len(sequence_lengths) == 0:
        return None

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
                _append(timestep_data[data_key], processed_data)

    return processed_data


def _interleave_standalone(data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Optional[Any]]]:
    output_data = {key: [] for key in ["input_ids", "patches", "positions"]}
    if "image" in data:
        _append(data["image"], output_data)
    if "text" in data:
        _append(data["text"], output_data)
    return output_data


def _get_batch_size(batch_data: Dict[str, Dict[str, Any]]) -> int:
    first_mod = next(iter(batch_data.values()))
    first_val = next(iter(first_mod.values()))
    return len(first_val)


def interleave_batch(batch_data: Dict[str, Any]) -> Dict[str, List[Any]]:
    batch_size = _get_batch_size(batch_data)
    output = {"input_ids": [], "patches": [], "positions": []}

    for batch_idx in range(batch_size):
        sample_data = _extract_idx_element(batch_data, batch_idx)
        if _is_episode(sample_data):
            x = _interleave_episode(sample_data)
        else:
            x = _interleave_standalone(sample_data)

        for key in x:
            output[key].append(x[key])

    return output
