from typing import Any, Dict, List, Optional

import numpy as np


class Interleaver:
    """
    Interleaver is a class that interleaves a batch of data into a unified dictionary of data lists.

    Important:
        For memory efficiency, we use the same pad object for all patches and positions.

    Example:
        In this example, the batch is composed of one standalone sample and one episode sample.
        The standalone sample is composed of two tokens of text.
        The episode sample is composed of two timesteps. The observations at each timestep are composed of
        two image patches (with the associated positions) and one discrete observation composed of 2 ints.
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
            "local_positions": [
                [0, 0],
                [0, 1, 2, 0, 0, 0, 1, 2, 0, 0],
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

    TOKEN_TYPE_ID = 0
    PATCH_TYPE_ID = 1

    def __init__(
        self,
        separator_token: Optional[int] = 31024,
        token_pad_value: int = 0,
        local_position_pad_value: int = -1,  # The one used for actions
        patch_pad_value: Optional[np.ndarray] = None,
        patch_position_pad_value: Optional[List[List[int]]] = None,
    ) -> None:
        self.token_pad_value = token_pad_value
        self.patch_pad_value = np.zeros((1, 1, 1), dtype=np.int64) if patch_pad_value is None else patch_pad_value
        self.local_position_pad_value = local_position_pad_value
        self.patch_position_pad_value = (
            [[0.0, 0.0], [0.0, 0.0]] if patch_position_pad_value is None else patch_position_pad_value
        )
        if separator_token is not None:
            self.separator = {
                "input_ids": [separator_token],
                "patches": [self.patch_pad_value],
                "patch_positions": [self.patch_position_pad_value],
                "input_types": [self.TOKEN_TYPE_ID],
            }
        else:
            self.separator = None

    @staticmethod
    def _is_episode(sample_data: Dict[str, Dict[str, Any]]) -> bool:
        """
        Determines if the keys of the sample_data dictionary follow the episode format. Keys can be either in
        ["images", "text"] or in ["image_observations", "text_observations", "discrete_observations",
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
        standalone_keys = set(["images", "text"])

        if key_set.issubset(epsiode_keys):
            return True
        elif key_set.issubset(standalone_keys):
            return False
        else:
            raise ValueError("Keys are mixed and do not follow the expected format.")

    @staticmethod
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

        Example:
            >>> nested_dict = {"outer_key": {"inner_key": [1, 2, 3]},
            ...                "outer_key2": {"inner_key": [4, 5, 6]}}
            >>> _extract_idx_element(nested_dict, 1)
            {"outer_key": {"inner_key": 2}, "outer_key2": {"inner_key": 5}}
        """
        output: Dict[str, Dict[str, Any]] = {}
        for outer_key, inner_dict in nested_dict.items():
            for inner_key, inner_list in inner_dict.items():
                if len(inner_list) <= index:
                    continue
                element = inner_list[index]
                if element is not None:
                    output.setdefault(outer_key, {})[inner_key] = element
        return output

    def _dict_append(
        self,
        batch_data: Dict[str, Dict[str, Any]],
        processed_data: Dict[str, List[Any]],
        local_positions: List[int],
        loss_mask_value: int,
    ) -> None:
        """
        Appends the data from a single episode to the processed data dictionary.

        Args:
            batch_data (Dict[str, Dict[str, Any]]): A dictionary containing the data from a single episode.
            processed_data (Dict[str, List[Any]]): A dictionary containing the processed data from all episodes.
            loss_mask_value (int): The value to use for the loss mask.

        Raises:
            ValueError: If the batch data does not contain either "input_ids" or "patches" and "patch_positions".

        Example:
            >>> batch_data = {"input_ids": [43]}
            >>> processed_data = {"input_ids": [42], "patches": [PATCH_PAD], "patch_positions": [POS_PAD],
            ...                   "input_types": [0], "loss_mask": [1]}
            >>> _dict_append(batch_data, processed_data, loss_mak_value=0)
            >>> processed_data
            {"input_ids": [42, 43], "patches": [PATCH_PAD, PATCH_PAD], "patch_positions": [POS_PAD, POS_PAD],
                "input_types": [0, 0], "loss_mask": [1, 0]}
            >>> batch_data = {"patches": [np.ones((4, 16, 16))], "patch_positions": [[[0.0, 0.0], [0.2, 0.5]]]}
            >>> _dict_append(batch_data, processed_data, loss_mak_value=0)
            >>> processed_data
            {"input_ids": [42, 43, 0], "patches": [PATCH_PAD, PATCH_PAD, np.ones((4, 16, 16))],
                "patch_positions": [POS_PAD, POS_PAD, [[0.0, 0.0], [0.2, 0.5]]], "input_types": [0, 0, 1],
                "loss_mask": [1, 0, 1]}
        """
        # If the batch data is not a list, convert it to a list of size 1
        for key in batch_data:
            if not isinstance(batch_data[key], list):
                batch_data[key] = [batch_data[key]]

        # Get the number of elements in the batch data
        num_elements = len(next(iter(batch_data.values())))

        # Get the input ids, patches, positions, input type and loss mask
        input_ids = batch_data.get("input_ids", [self.token_pad_value] * num_elements)
        patches = batch_data.get("patches", [self.patch_pad_value] * num_elements)
        patch_positions = batch_data.get("patch_positions", [self.patch_position_pad_value] * num_elements)
        if "input_ids" in batch_data:
            input_types = [self.TOKEN_TYPE_ID] * num_elements
        elif "patches" in batch_data and "patch_positions" in batch_data:
            input_types = [self.PATCH_TYPE_ID] * num_elements
        else:
            raise ValueError("Batch data must contain either input_ids or patches and patch_positions.")
        loss_maskes = [loss_mask_value] * num_elements

        # Append the data to the processed data dictionary
        processed_data["input_ids"].extend(input_ids)
        processed_data["local_positions"].extend(local_positions)
        processed_data["patches"].extend(patches)
        processed_data["patch_positions"].extend(patch_positions)
        processed_data["input_types"].extend(input_types)
        processed_data["loss_mask"].extend(loss_maskes)

    def _interleave_episode(self, episode_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Interleaves an episode's data into a unified dictionary of data lists, thereby converting data stored
        in different categories into a time-sequenced format.

        Args:
            episode_data (Dict[str, Dict[str, Any]]): A dictionary containing a single episode's data, categorized
            by type. At the first level, keys must be "input_ids", "patches", "patch_positions". At the second level,
            keys must be "text_observations", "image_observations", "discrete_observations", "continuous_observations",
            "discrete_actions", and "continuous_actions".

        Returns:
            Dict[str, List[Any]]: A dictionary containing the interleaved episode data. The keys are "input_ids",
            "patches",  "patch_positions", "input_types", and "loss_mask". Each key corresponds to a list, where each
            list item represents data from a specific time step.

        Example:
            >>> episode_data = {"image_observations": {"patches": [[P1, P2], [P3, P4]],
            ...                                        "patch_positions": [[LEFT, RIGHT], [LEFT, RIGHT]]},
            ...                 "discrete_actions": {"input_ids": [[1, 2], [3, 4]]}
            >>> _interleave_episode(episode_data)
            {'input_ids': [0, 0, 1, 2, 0, 0, 3, 4],
             'patches': [P1, P2, PATCH_PAD, PATCH_PAD, P3, P4, PATCH_PAD, PATCH_PAD],
             'patch_positions': [LEFT, RIGHT, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD],
             'input_types': [0, 0, 1, 1, 0, 0, 1, 1],
             'loss_mask': [0, 0, 1, 1, 0, 0, 1, 1]}

        Note:
            The function assumes that all data sequences in the provided episode have the same length.
            It will raise an assertion error if this is not the case.
        """
        output = {
            "input_ids": [],
            "local_positions": [],
            "patches": [],
            "patch_positions": [],
            "input_types": [],
            "loss_mask": [],
        }

        observation_keys = [
            "text_observations",
            "image_observations",
            "discrete_observations",
            "continuous_observations",
        ]
        action_keys = [
            "discrete_actions",
            "continuous_actions",
        ]

        # Get the length of each sequence in the d and make sure they are all equal
        obs_total_timesteps = None
        for key in observation_keys:
            if key in episode_data:
                for ep in episode_data[key].values():
                    if obs_total_timesteps is None:
                        obs_total_timesteps = len(ep)
                    assert obs_total_timesteps == len(ep)

        action_total_timesteps = None
        for key in action_keys:
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
            local_position = 0
            # Extract the current element from each sequence in the d
            data_t = self._extract_idx_element(episode_data, t)

            ordered_obs_keys = [key for key in observation_keys if key in data_t]
            ordered_action_keys = [key for key in action_keys if key in data_t]
            for mod_key in ordered_obs_keys:
                for func_key in data_t[mod_key]:
                    # If the data is not a list, convert it to a list of size 1
                    if not isinstance(data_t[mod_key][func_key], list):
                        data_t[mod_key][func_key] = [data_t[mod_key][func_key]]
                    # Get the number of elements in the data (should be the same for all func keys)
                    num_elements = len(data_t[mod_key][func_key])

                # Compute the local positions of the data
                if mod_key in observation_keys:
                    local_positions = list(range(local_position, local_position + num_elements))
                    local_position += num_elements
                else:  # mod_key in action_keys
                    local_positions = [self.local_position_pad_value] * num_elements

                # Compute the loss mask value
                if mod_key in action_keys + ["text_observations"]:
                    loss_mask_value = 1
                else:  # mod_key in ["image_observations", "discrete_observations", "continuous_observations"]
                    loss_mask_value = 0

                self._dict_append(data_t[mod_key], output, local_positions, loss_mask_value)

            if self.separator is not None:
                self._dict_append(
                    self.separator, output, local_positions=[self.local_position_pad_value], loss_mask_value=1
                )

            for mod_key in ordered_action_keys:
                for func_key in data_t[mod_key]:
                    # If the data is not a list, convert it to a list of size 1
                    if not isinstance(data_t[mod_key][func_key], list):
                        data_t[mod_key][func_key] = [data_t[mod_key][func_key]]
                    # Get the number of elements in the data (should be the same for all func keys)
                    num_elements = len(data_t[mod_key][func_key])

                # Compute the local positions of the data
                if mod_key in observation_keys:
                    local_positions = list(range(local_position, local_position + num_elements))
                    local_position += num_elements
                else:  # mod_key in action_keys
                    local_positions = [self.local_position_pad_value] * num_elements

                # Compute the loss mask value
                if mod_key in action_keys + ["text_observations"]:
                    loss_mask_value = 1
                else:  # mod_key in ["image_observations", "discrete_observations", "continuous_observations"]
                    loss_mask_value = 0

                self._dict_append(data_t[mod_key], output, local_positions, loss_mask_value)
        return output

    def _interleave_standalone(self, standalone_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Interleaves standalone data, such as images or text, into a unified dictionary of data lists.

        Args:
            data (Dict[str, Dict[str, Any]]): A dictionary containing data categorized by type. The first level keys
                can be "images" or "text". The second level keys can be "input_ids", "patches", "patch_positions", etc.

        Returns:
            dict: A dictionary containing the interleaved data. The keys are "input_ids", "patches", "patch_positions",
                "input_types", and "loss_mask". Each key corresponds to a list of data points.

        Example:
            >>> standalone_data = {"images": {"patches": [P1, P2], "patch_positions": [LEFT, RIGHT]},
                        "text": {"input_ids": [1, 2]}
            >>> _interleave_standalone(standalone_data)
            {'input_ids': [0, 0, 1, 2],
             'patches': [P1, P2, PATCH_PAD, PATCH_PAD],
             'patch_positions': [LEFT, RIGHT, POS_PAD, POS_PAD],
             'input_types': [0, 0, 1, 1],
             'loss_mask': [0, 0, 1, 1]}
        """
        output = {
            "input_ids": [],
            "local_positions": [],
            "patches": [],
            "patch_positions": [],
            "input_types": [],
            "loss_mask": [],
        }

        if "images" in standalone_data:
            local_positions = [self.local_position_pad_value] * len(standalone_data["images"]["patches"])
            self._dict_append(standalone_data["images"], output, local_positions, loss_mask_value=0)
        if "text" in standalone_data:
            local_positions = [self.local_position_pad_value] * len(standalone_data["text"]["input_ids"])
            self._dict_append(standalone_data["text"], output, local_positions, loss_mask_value=1)
        return output

    def __call__(self, batch_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        output = {
            "input_ids": [],
            "local_positions": [],
            "patches": [],
            "patch_positions": [],
            "input_types": [],
            "loss_mask": [],
        }

        # Get the batch size
        first_mod = next(iter(batch_data.values()))
        first_val = next(iter(first_mod.values()))
        batch_size = len(first_val)

        for batch_idx in range(batch_size):
            data = self._extract_idx_element(batch_data, batch_idx)
            if self._is_episode(data):
                x = self._interleave_episode(data)
            else:
                x = self._interleave_standalone(data)

            for key in x:
                output[key].append(x[key])

        return output
