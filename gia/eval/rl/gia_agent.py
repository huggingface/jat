from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from gymnasium import spaces
from torch import Tensor

from gia.datasets import GiaDataCollator, Prompter
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor

from .envs.core import get_task_names, make


class GiaAgent:
    r"""
    An RL agent that uses Gia to generate actions.

    Warning:
        The agent caches past key values from the model. This means that when you call `get_action`  multiple times in
        succession, the agent will generate actions based on all the previous actions passed to `get_action`. If you
        want to reset the agent to generate actions based on the initial prompt, you need to call the `reset` method.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            Can be either:

                - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                    huggingface.co.
                - A path to a *directory* containing a configuration file saved using the
                    [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                    e.g., `./my_model_directory/`.
                - A path or url to a saved configuration JSON *file*, e.g.,
                    `./my_model_directory/configuration.json`.
        task_name (`str`):
            The environment id. Check the available task names with `GiaAgent.get_available_task_names()`.
        collator (`GiaDataCollator`):
            The GiaDataCollator to use for collating processed observations.
        observation_space (`spaces.Space`):
            The observation space.
        action_space (`spaces.Space`):
            The action space.
        prompter (`Prompter`, *optional*, defaults to None):
            The Prompter to use for generating prompts. When None, the agent will not use prompts. Defaults to None.
        deterministic (`bool`, *optional*, defaults to False):
            Whether to use deterministic action generation. Defaults to False.
    """

    def __init__(
        self,
        model: GiaModel,
        processor: GiaProcessor,
        task_name: str,
        num_envs: int = 1,
        use_prompt: bool = True,
        p_prompt: float = 0.25,
        p_end: float = 0.1,
        min_prompt_len: int = 1,
        max_prompt_len: int = 1024,
        deterministic: bool = False,
    ) -> None:
        self.processor = processor
        self.model = model

        if use_prompt:
            dataset = load_dataset("gia-project/gia-dataset", task_name, split="test", writer_batch_size=1)
            self.prompter = Prompter(dataset, p_prompt, p_end, min_prompt_len, max_prompt_len)
        else:
            self.prompter = None

        self.collator = GiaDataCollator()

        self.num_envs = num_envs

        # Get observation and action space
        env = make(task_name)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.deterministic = deterministic
        self.device = next(self.model.parameters()).device
        self._max_length = self.model.config.max_position_embeddings - 100  # TODO: check this

        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("Unsupported observation space")

        if isinstance(self.action_space, spaces.Box):
            self._num_act_tokens = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            self._num_act_tokens = 1
        else:
            raise TypeError("Unsupported action space")

    @staticmethod
    def get_available_task_names() -> List[str]:
        """
        Returns the available task names.

        Returns:
            List[str]: The available task names.
        """
        return get_task_names()

    def _truncate_past_key_values(
        self, past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        return tuple((k[:, :, -self._max_length :], v[:, :, -self._max_length :]) for (k, v) in past_key_values)

    def reset(self) -> None:
        if self.prompter is not None:
            prompts = self.prompter.generate_prompts(self.num_envs)
            processed_prompt = self.processor(
                **prompts,
                padding=False,
                truncation="max_length",
                truncation_side="left",
                max_length=self._max_length,
            )
            processed_prompt = self.collator(
                [
                    {key: processed_prompt[key][ep_idx] for key in processed_prompt.keys()}
                    for ep_idx in range(self.num_envs)
                ]
            )
            for key in processed_prompt.keys():
                processed_prompt[key] = processed_prompt[key].to(self.device)
            output = self.model(**processed_prompt, use_cache=True)
            self._past_key_values = self._truncate_past_key_values(output.past_key_values)
        else:
            self._past_key_values = None

    def get_action(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the next action given the current observation.

        Args:
            observations (np.ndarray): The current observation

        Returns:
            np.ndarray: The next action
        """
        # Turn into episode
        dict_observations = {}
        for key, values in observations.items():
            if isinstance(values[0], np.ndarray):
                dict_observations[key] = np.expand_dims(np.stack(values), axis=1)
            elif isinstance(values[0], str):
                dict_observations[key] = [[value] for value in values]
            else:
                raise TypeError(f"Unsupported type for {key}")

        # Process observations
        processed = self.processor(
            **dict_observations,
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=self._max_length,  # ensure not to not overflow
        )

        # Process and move to device
        num_envs = len(processed["input_types"])
        processed = self.collator(
            [{key: processed[key][ep_idx] for key in processed.keys()} for ep_idx in range(num_envs)]
        )
        for key in processed.keys():
            processed[key] = processed[key].to(self.device)

        # Generate action tokens
        action_tokens = []
        for _ in range(self._num_act_tokens):
            output = self.model(**processed, use_cache=True, past_key_values=self._past_key_values)
            self._past_key_values = self._truncate_past_key_values(output.past_key_values)
            action_logits = output.logits[:, -1]

            if self.deterministic:
                action_token = torch.argmax(action_logits, dim=-1)
            else:
                action_token = torch.multinomial(torch.softmax(action_logits, dim=-1), num_samples=1).squeeze(-1)
            action_tokens.append(action_token.tolist())

            processed = {
                "input_ids": action_token[:, None],
                "loss_mask": torch.ones(num_envs, 1, dtype=torch.bool, device=self.device),
                "input_types": torch.zeros(num_envs, 1, dtype=torch.int64, device=self.device),
                "local_positions": -torch.ones(num_envs, 1, dtype=torch.int64, device=self.device),
            }

        # To ensure the KV cache includes the last action token
        output = self.model(**processed, use_cache=True, past_key_values=self._past_key_values)
        self._past_key_values = self._truncate_past_key_values(output.past_key_values)

        # Transpose action_tokens to be (num_envs, num_act_tokens)
        action_tokens = np.array(action_tokens, dtype=self.action_space.dtype).T.tolist()

        if isinstance(self.action_space, spaces.Box):
            # Decode the action tokens
            actions = np.array(self.processor.decode_continuous(action_tokens), dtype=self.action_space.dtype)

            # Clip the action if necessary
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        elif isinstance(self.action_space, spaces.Discrete):
            # Decode the action tokens
            actions = np.array(self.processor.decode_discrete(action_tokens), dtype=self.action_space.dtype)
            actions = actions.squeeze(axis=1)

            # Clip the action if necessary (decoded actions are between 0 and num_bins)
            actions = np.clip(actions, a_min=0, a_max=self.action_space.n - 1)
        return actions
