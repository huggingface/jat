from typing import Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor

from gia.datasets import GiaDataCollator, Prompter
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor


class GiaAgent:
    r"""
    An RL agent that uses Gia to generate actions.

    Warning:
        The agent caches past key values from the model. This means that when you call `get_action`  multiple times in
        succession, the agent will generate actions based on all the previous actions passed to `get_action`. If you
        want to reset the agent to generate actions based on the initial prompt, you need to call the `reset` method.

    Args:
        model (`GiaModel`):
            The GiaModel to use for action generation.
        processor (`GiaProcessor`):
            The GiaProcessor to use for processing observations.
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
        collator: GiaDataCollator,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prompter: Optional[Prompter] = None,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.prompter = prompter
        self.processor = processor
        self.collator = collator
        self.observation_space = observation_space
        self.action_space = action_space
        self.deterministic = deterministic
        self.device = next(model.parameters()).device
        self._max_length = self.model.config.max_position_embeddings - 10

        if isinstance(observation_space, spaces.Box):
            self._observation_key = "continuous_observations"
        elif isinstance(observation_space, spaces.MultiDiscrete):
            self._observation_key = "discrete_observations"
        else:
            raise TypeError("Unsupported observation space")

        if isinstance(action_space, spaces.Box):
            self._num_act_tokens = action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            self._num_act_tokens = 1
        else:
            raise TypeError("Unsupported action space")

    def _truncate_past_key_values(
        self, past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        return tuple((k[:, :, -self._max_length :], v[:, :, -self._max_length :]) for (k, v) in past_key_values)

    def reset(self, num_envs: int = 1) -> None:
        if self.prompter is not None:
            prompts = self.prompter.generate_prompts(num_envs)
            processed_prompt = self.processor(
                **prompts,
                padding=False,
                truncation="max_length",
                truncation_side="left",
                max_length=self._max_length,
            )

            processed_prompt = self.collator(
                [{key: processed_prompt[key][ep_idx] for key in processed_prompt.keys()} for ep_idx in range(num_envs)]
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
        num_envs = observations.shape[0]
        observations = np.expand_dims(observations, axis=1).tolist()

        # Process observations
        inputs = {self._observation_key: observations}
        processed = self.processor(
            **inputs,
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=self._max_length,  # ensure not to not overflow
        )

        # Process and move to device
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

            processed["input_ids"] = action_token[:, None]
            processed["loss_mask"] = torch.ones(num_envs, 1, dtype=torch.bool, device=self.device)
            processed["input_types"] = torch.zeros(num_envs, 1, dtype=torch.int64, device=self.device)
            processed["local_positions"] = -torch.ones(num_envs, 1, dtype=torch.int64, device=self.device)

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
            actions = np.clip(actions, a_min=0, a_max=self.action_space.n)
        return actions
