from typing import Any, Dict, List, Optional

import numpy as np


class Interleaver:
    """
    Interleaver is a class that interleaves a batch of data into a unified dictionary of data lists.

    Example:
        # In this example, the batch is composed of one standalone sample and one episode sample.
        # The standalone sample is composed of two tokens of text.
        # The episode sample is composed of two timesteps. The observations at each timestep are composed of
        # two image patches (with the associated positions) and one discrete observation composed of 2 ints.
        >>> interleaver = Interleaver()
        >>> batch_data = {"text": {"input_ids": [[1, 2], None]},
        ...               "image_observations": {"patches": [None, [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4]]],
        ...                                      "positions": [None, [[LEFT, RIGHT], [LEFT, RIGHT]]]},
        ...               "continuous_actions": {"input_ids": [None, [[11, 12], [13, 14]]]}}
        >>> interleaver(batch_data)
        {
            "input_ids": [
                [1, 2],
                [0, 0, 11, 12, 0, 0, 13, 14],
            ],
            "patches": [
                [PATCH_PAD, PATCH_PAD],
                [PATCH_1, PATCH_2, PATCH_PAD, PATCH_PAD, PATCH_3, PATCH_4, PATCH_PAD, PATCH_PAD],
            ],
            "positions": [
                [POS_PAD, POS_PAD],
                [LEFT, RIGHT, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD],
            ],
            "input_type": [
                [0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0],
            ],
            "loss_mask": [
                [1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
            ],
        }
    """

    # Here, we define the pads for the patches and positions.
    # For memory efficiency, we use the same pad for all patches and positions.
    PATCH_PAD = np.zeros((3, 16, 16), dtype=np.int64)
    POSITION_PAD = [[0.0, 0.0], [0.0, 0.0]]
    TOKEN_PAD = 0

    TOKEN_TYPE_ID = 0
    PATCH_TYPE_ID = 1

    @staticmethod
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
                element = inner_list[index]
                if element is not None:
                    output.setdefault(outer_key, {})[inner_key] = element
        return output

    @classmethod
    def _dict_append(
        cls, batch_data: Dict[str, Dict[str, Any]], processed_data: Dict[str, List[Any]], loss_mask_value: int
    ) -> None:
        """
        Appends the data from a single episode to the processed data dictionary.

        Args:
            batch_data (Dict[str, Dict[str, Any]]): A dictionary containing the data from a single episode.
            processed_data (Dict[str, List[Any]]): A dictionary containing the processed data from all episodes.
            loss_mask_value (int): The value to use for the loss mask.

        Raises:
            ValueError: If the batch data does not contain either "input_ids" or "patches" and "positions".

        Example:
            >>> batch_data = {"input_ids": [43]}
            >>> processed_data = {"input_ids": [42], "patches": [PATCH_PAD], "positions": [POS_PAD], "input_type": [0],
            ...                   "loss_mask": [1]}
            >>> _dict_append(batch_data, processed_data, loss_mak_value=0)
            >>> processed_data
            {"input_ids": [42, 43], "patches": [PATCH_PAD, PATCH_PAD], "positions": [POS_PAD, POS_PAD],
                "input_type": [0, 0], "loss_mask": [1, 0]}
            >>> batch_data = {"patches": [np.ones((3, 16, 16))], "positions": [[[0.0, 0.0], [0.2, 0.5]]]}
            >>> _dict_append(batch_data, processed_data, loss_mak_value=0)
            >>> processed_data
            {"input_ids": [42, 43, 0], "patches": [PATCH_PAD, PATCH_PAD, np.ones((3, 16, 16))],
                "positions": [POS_PAD, POS_PAD, [[0.0, 0.0], [0.2, 0.5]]], "input_type": [0, 0, 1],
                "loss_mask": [1, 0, 1]}
        """
        # If the batch data is not a list, convert it to a list of size 1
        for key in batch_data:
            if not isinstance(batch_data[key], list):
                batch_data[key] = [batch_data[key]]

        # Get the number of elements in the batch data
        num_elements = len(next(iter(batch_data.values())))

        # Get the input ids, patches, positions, input type and loss mask
        input_ids = batch_data.get("input_ids", [cls.TOKEN_PAD] * num_elements)
        patches = batch_data.get("patches", [cls.PATCH_PAD] * num_elements)
        positions = batch_data.get("positions", [cls.POSITION_PAD] * num_elements)
        if "input_ids" in batch_data:
            input_type = [cls.TOKEN_TYPE_ID] * num_elements
        elif "patches" in batch_data and "positions" in batch_data:
            input_type = [cls.PATCH_TYPE_ID] * num_elements
        else:
            raise ValueError("Batch data must contain either input_ids or patches and positions.")
        loss_maskes = [loss_mask_value] * num_elements

        # Append the data to the processed data dictionary
        processed_data["input_ids"].extend(input_ids)
        processed_data["patches"].extend(patches)
        processed_data["positions"].extend(positions)
        processed_data["input_type"].extend(input_type)
        processed_data["loss_mask"].extend(loss_maskes)

    def _interleave_episode(self, episode_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Interleaves an episode's data into a unified dictionary of data lists, thereby converting data stored
        in different categories into a time-sequenced format.

        Args:
            episode_data (Dict[str, Dict[str, Any]]): A dictionary containing a single episode's data, categorized
            by type. At the first level, keys must be "input_ids", "patches", "positions". At the second level,
            keys must be "text_observations", "image_observations", "discrete_observations", "continuous_observations",
            "discrete_actions", and "continuous_actions".

        Returns:
            Dict[str, List[Any]]: A dictionary containing the interleaved episode data. The keys are "input_ids",
            "patches",  "positions", "input_type", and "loss_mask". Each key corresponds to a list, where each list
            item represents data from a specific time step.

        Example:
            >>> episode_data = {"image_observations": {"patches": [[P1, P2], [P3, P4]],
            ...                                        "positions": [[LEFT, RIGHT], [LEFT, RIGHT]]},
            ...                 "discrete_actions": {"input_ids": [[1, 2], [3, 4]]}
            >>> _interleave_episode(episode_data)
            {'input_ids': [0, 0, 1, 2, 0, 0, 3, 4],
             'patches': [P1, P2, PATCH_PAD, PATCH_PAD, P3, P4, PATCH_PAD, PATCH_PAD],
             'positions': [LEFT, RIGHT, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD],
             'input_type': [0, 0, 1, 1, 0, 0, 1, 1],
             'loss_mask': [0, 0, 1, 1, 0, 0, 1, 1]}

        Note:
            The function assumes that all data sequences in the provided episode have the same length.
            It will raise an assertion error if this is not the case.
        """
        output = {"input_ids": [], "patches": [], "positions": [], "input_type": [], "loss_mask": []}

        # Get the length of each sequence in the d and make sure they are all equal
        sequence_lengths = [len(episode) for episodes in episode_data.values() for episode in episodes.values()]
        common_length = sequence_lengths[0]
        assert all(common_length == length for length in sequence_lengths)

        # Interleave the data
        # Order: observation (text, image, discrete, continuous), then action (discrete, continuous)
        for t in range(common_length):
            # Extract the current element from each sequence in the d
            timestep_data = self._extract_idx_element(episode_data, t)
            if "text_observations" in timestep_data:
                self._dict_append(timestep_data["text_observations"], output, loss_mask_value=1)
            if "image_observations" in timestep_data:
                self._dict_append(timestep_data["image_observations"], output, loss_mask_value=0)
            if "discrete_observations" in timestep_data:
                self._dict_append(timestep_data["discrete_observations"], output, loss_mask_value=0)
            if "continuous_observations" in timestep_data:
                self._dict_append(timestep_data["continuous_observations"], output, loss_mask_value=0)
            if "discrete_actions" in timestep_data:
                self._dict_append(timestep_data["discrete_actions"], output, loss_mask_value=1)
            if "continuous_actions" in timestep_data:
                self._dict_append(timestep_data["continuous_actions"], output, loss_mask_value=1)
        return output

    def _interleave_standalone(self, standalone_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Interleaves standalone data, such as images or text, into a unified dictionary of data lists.

        Args:
            data (Dict[str, Dict[str, Any]]): A dictionary containing data categorized by type.
            The first level keys can be "image" or "text". The second level keys can be "input_ids",
            "patches", "positions", etc.

        Returns:
            dict: A dictionary containing the interleaved data. The keys are "input_ids", "patches",
            "positions", "input_type", and "loss_mask". Each key corresponds to a list of data points.

        Example:
            >>> standalone_data = {"image": {"patches": [P1, P2], "positions": [LEFT, RIGHT]},
                        "text": {"input_ids": [1, 2]}
            >>> _interleave_standalone(standalone_data)
            {'input_ids': [0, 0, 1, 2],
             'patches': [P1, P2, PATCH_PAD, PATCH_PAD],
             'positions': [LEFT, RIGHT, POS_PAD, POS_PAD],
             'input_type': [0, 0, 1, 1],
             'loss_mask': [0, 0, 1, 1]}
        """
        output = {"input_ids": [], "patches": [], "positions": [], "input_type": [], "loss_mask": []}
        if "image" in standalone_data:
            self._dict_append(standalone_data["image"], output, loss_mask_value=0)
        if "text" in standalone_data:
            self._dict_append(standalone_data["text"], output, loss_mask_value=1)
        return output

    def __call__(self, batch_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        output = {"input_ids": [], "patches": [], "positions": [], "input_type": [], "loss_mask": []}

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
