from typing import Any, Dict, List, Optional


def indexing_from_nested(nested_dict: Dict, index: int) -> Dict:
    r"""
    Extract the index-th element from each sequence in the nested dictionary.

    Args:
        nested_dict (`Dict`):
            A nested dictionary where the innermost values are lists of elements, which can be `None` or any other
            type.
        index (`int`):
            The index of the element to be extracted from each list in the nested dictionary.

    Returns:
        nested_dict (`Dict`):
            A nested dictionary with the same structure as the input dictionary, containing only the extracted
            index-th elements, excluding `None` values.

    Example:
        >>> nested_dict = {"outer_key1": {"inner_key": [1, 2, 3]},
        ...                "outer_key2": {"inner_key": [4, 5, 6]}}
        >>> extract_idx_element(nested_dict, 1)
        {"outer_key1": {"inner_key": 2}, "outer_key2": {"inner_key": 5}}
    """
    if isinstance(nested_dict, list):
        if index < len(nested_dict):
            return nested_dict[index]
        else:
            return None
    elif nested_dict is None:
        return None
    elif isinstance(nested_dict, dict):
        result = {}
        for k, v in nested_dict.items():
            inner = indexing_from_nested(v, index)
            if inner is not None:
                result[k] = inner
        return result if result else None
    else:
        raise TypeError(f"Unsupported type: {type(nested_dict)}")


def extend_dol(dol: Dict[str, List[Any]], other_dol: Dict[str, List[Any]]) -> None:
    r"""
    Extend the lists in a dictionary (dict of list, abbreviated dol) with corresponding lists from another dictionary
    (other_dol).

    This function modifies the input `dol` in-place by extending its lists with the corresponding lists from
    `other_dol`. If a key is missing in `other_dol`, the corresponding list in `dol` is extended with `None` values.

    Args:
        dol (`Dict[str, List[Any]]`):
            A dictionary where the values are lists. All lists within the dol should have the same length.
        other_dol (`Dict[str, List[Any]]`):
            Another dictionary with the same structure, where the values are lists to extend the corresponding lists
            in `dol`.

    Example:
        >>> dol = {"key1": [1, 2],
        ...        "key2": [5, 6]}
        >>> other_dol = {"key1": [3, 4, 5]}
        >>> extend_dol(dol, other_dol)
        >>> print(dol)
        {"key1": [1, 2, 3, 4, 5],
         "key2": [5, 6, None, None, None]}
    """

    def get_length(dol: Dict) -> int:
        if dol:
            return len(next(iter(dol.values())))
        return 0

    length = get_length(dol)
    other_length = get_length(other_dol)

    for key in set(dol.keys()).union(set(other_dol.keys())):
        if key not in dol:
            dol[key] = [None] * length
        if key not in other_dol:
            other_dol[key] = [None] * other_length
        dol[key].extend(other_dol[key])


class Interleaver:
    r"""
    Class that interleaves a batch of data into a unified dictionary of data lists.

    Example:
        In this example, the batch is composed of one standalone sample and one episode sample.
        The standalone sample is composed of two tokens of text.
        The episode sample is composed of two timesteps. The observations at each timestep are composed of
        two image patches (with the associated positions) and one discrete observation composed of 2 integers.

        >>> interleaver = Interleaver()
        >>> batch_data = {
        ...     "text":                  {"input_ids":       [[1, 2], None]},
        ...     "image_observations":    {"patches":         [None,   [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4]]],
        ...                               "patch_positions": [None,   [[LEFT, RIGHT], [LEFT, RIGHT]]]},
        ...     "discrete_observations": {"input_ids":       [None,   [3, 4]]},
        ...     "continuous_actions":    {"input_ids":       [None,   [[5, 6], [7, 8]]]}}
        >>> interleaver(batch_data)
        {
            "input_ids": [
                [1, 2],
                [0, 0, 3, 5, 6, 0, 0, 4, 7, 8],
            ],
            "patches": [
                [PATCH_PAD, PATCH_PAD],
                [PATCH_1, PATCH_2, PATCH_PAD, PATCH_PAD, PATCH_PAD, PATCH_3, PATCH_4, PATCH_PAD, PATCH_PAD, PATCH_PAD],
            ],
            "patch_positions": [
                [POS_PAD, POS_PAD],
                [LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD],
            ],
            "input_types": [
                [0, 0],
                [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            "loss_mask": [
                [1, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            ],
        }
    """

    OBSERVATION_KEYS = [
        "text_observations",
        "image_observations",
        "discrete_observations",
        "continuous_observations",
    ]
    ACTION_KEYS = [
        "discrete_actions",
        "continuous_actions",
    ]

    EPISODE_KEYS = set(OBSERVATION_KEYS + ACTION_KEYS)
    STANDALONE_KEYS = {"images", "text"}

    def __init__(self, separator: Optional[Dict[str, Any]] = None) -> None:
        self.separator = separator

    @classmethod
    def _is_episode(cls, sample_data: Dict[str, Dict[str, Any]]) -> bool:
        r"""
        Determine if the keys of the sample_data dictionary follow the episode format. Keys can be either in
        `["images", "text"]` or in `["image_observations", "text_observations", "discrete_observations",
        "continuous_observations", "discrete_actions", "continuous_actions"]`. They can't be in both.

        Args:
            sample_data (`Dict[str, Dict[str, Any]]`):
                A dictionary containing sample data with specific keys.

        Returns:
            is_episode (`bool`):
                `True` if the keys follow the episode format, `False` otherwise.

        Raises:
            `ValueError`: If the keys are mixed and do not follow the expected format.
        """
        key_set = set(sample_data.keys())
        if key_set.issubset(cls.EPISODE_KEYS):
            return True
        elif key_set.issubset(cls.STANDALONE_KEYS):
            return False
        else:
            raise ValueError("Keys are mixed and do not follow the expected format.")

    def _interleave_episode(self, episode_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        r"""
        Interleave an episode's data into a unified dictionary of data lists, thereby converting data stored
        in different categories into a time-sequenced format.

        Args:
            episode_data (`Dict[str, Dict[str, Any]]`):
                A dictionary containing a single episode's data, categorized by type. At the first level, keys must be
                in `"text_observations"`, `"image_observations"`, `"discrete_observations"`,
                `"continuous_observations"`, `"discrete_actions"`, and `"continuous_actions"`.

        Returns:
            interleaved (`Dict[str, List[Any]]`):
                A dictionary containing the interleaved episode data. Each key corresponds to a list, where each
                list item represents data from a specific time step.

        Example:
            In practive, we use `"input_ids"`, `"patches"` as second-level keys.
            >>> episode_data = {"image_observations": {"a": [0, 1], "b": [2, 3]},
            ...                 "discrete_actions": {"a": [[1, 2], [3, 4]], "c": [[5, 6], [7, 8]]}}
            >>> _interleave_episode(episode_data)
            {'a': [0, 1, 2, 1, 3, 4],
             'b': [2, None, None, 3, None, None],
             'c': [None, 5, 6, None, 7, 8]}

        Note:
            The function assumes that all data sequences in the provided episode have the same length.
            It will raise an `AssertionError` error if this is not the case.
        """
        output = {}

        # Get the length of each sequence in the d and make sure they are all equal
        obs_total_timesteps = None
        for key in self.OBSERVATION_KEYS:
            if key in episode_data:
                for ep in episode_data[key].values():
                    if obs_total_timesteps is None:
                        obs_total_timesteps = len(ep)
                    assert obs_total_timesteps == len(ep)

        action_total_timesteps = None
        for key in self.ACTION_KEYS:
            if key in episode_data:
                for ep in episode_data[key].values():
                    if action_total_timesteps is None:
                        action_total_timesteps = len(ep)
                    assert action_total_timesteps == len(ep)

        # It is possible that the number of observations exceeds the number of actions by 1 (when the model has
        # to predict the next action)
        assert obs_total_timesteps is not None, "No observations found in the episode data."

        # Interleave the data
        # Order: observation (text, image, discrete, continuous), then action (discrete, continuous)
        for t in range(obs_total_timesteps):
            # Extract the current element from each sequence in the d
            data_t = indexing_from_nested(episode_data, t)

            ordered_obs_keys = [key for key in self.OBSERVATION_KEYS if key in data_t]
            ordered_action_keys = [key for key in self.ACTION_KEYS if key in data_t]

            for mod_key in ordered_obs_keys:
                func_dict = data_t[mod_key]
                to_append = {key: val if isinstance(val, list) else [val] for key, val in func_dict.items()}
                extend_dol(output, to_append)

            if self.separator is not None:
                extend_dol(output, self.separator)

            for mod_key in ordered_action_keys:
                func_dict = data_t[mod_key]
                to_append = {key: val if isinstance(val, list) else [val] for key, val in func_dict.items()}
                extend_dol(output, to_append)
        return output

    def _interleave_standalone(self, standalone_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        r"""
        Interleaves standalone data, such as images or text, into a unified dictionary of data lists.

        Args:
            data (`Dict[str, Dict[str, Any]]`):
                A dictionary containing data categorized by type. The first level keys must be in  `"images"` and
                `"text"`.

        Returns:
            interleaved (`Dict`):
                A dictionary containing the interleaved data. Each key corresponds to a list of data points.

        Example:
            >>> standalone_data = {"images": {"patches": [P1, P2], "patch_positions": [LEFT, RIGHT]},
                                   "text": {"input_ids": [1, 2]}
            >>> _interleave_standalone(standalone_data)
            {'input_ids': [0, 0, 1, 2],
             'patches': [P1, P2, None, None],
             'loss_mask': [0, 0, 1, 1]}
        """
        output = {}
        for key in ["images", "text"]:
            if key in standalone_data:
                extend_dol(output, standalone_data[key].copy())
        return output

    def __call__(self, batch_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        r"""
        Interleave data.

        Args:
            batch_data (`Dict[str, Dict[str, Any]]`):
                Batch data as dict of dict.

        Returns:
            interleaved_batched_data (`Dict[str, List[Any]]`):
                A dict of list, containing the interleaved data.
        """
        output = {}

        # Get the batch size
        first_mod = next(iter(batch_data.values()))
        first_val = next(iter(first_mod.values()))
        batch_size = len(first_val)

        for batch_idx in range(batch_size):
            data = indexing_from_nested(batch_data, batch_idx)
            if self._is_episode(data):
                x = self._interleave_episode(data)
            else:
                x = self._interleave_standalone(data)

            x = {key: [val] for key, val in x.items()}  # Convert all values to lists of size 1
            extend_dol(output, x)

        return output
