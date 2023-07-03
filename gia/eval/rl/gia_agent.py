import numpy as np
import torch
from typing import Optional
from gia.datasets import GiaDataCollator, Prompter
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor
from gymnasium import spaces


class GiaAgent:
    def __init__(
        self,
        model: GiaModel,
        processor: GiaProcessor,
        collator: GiaDataCollator,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prompter: Optional[Prompter] = None,
        deterministic: bool = False,
    ):
        self.model = model
        self.prompter = prompter
        self.processor = processor
        self.collator = collator
        self.deterministic = deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo: same as model
        self.max_length = 20

        if isinstance(observation_space, spaces.Box):
            self.observation_key = "continuous_observations"
        elif isinstance(observation_space, spaces.Discrete):
            self.observation_key = "discrete_observations"
        else:
            raise TypeError("Unsupported observation space")

        if isinstance(action_space, spaces.Box):
            self.action_key = "continuous_actions"
            self._num_act_tokens = action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            self.action_key = "discrete_actions"
            self._num_act_tokens = action_space.n
        else:
            raise TypeError("Unsupported action space")

        # self._num_obs_tokens = observation_space.shape[1]
        # self._num_act_tokens = action_space.shape[1]
        # self._tokens_per_step = self._num_obs_tokens + self._num_act_tokens + int(processor.use_separator)
        # self._int_per_seq = (self.model.config.seq_len // self._tokens_per_step) - 1
        # self._max_kv_size = self._int_per_seq * self._tokens_per_step

    def reset(self, num_envs: int = 1) -> None:
        if self.prompter is not None:
            prompts = self.prompter.generate_prompts(num_envs)
            processed_prompt = self.processor(
                **prompts,
                padding=False,
                truncation="max_length",
                truncation_side="left",
                max_length=self.max_length,
            )

            processed_prompt = self.collator(
                [{key: processed_prompt[key][ep_idx] for key in processed_prompt.keys()} for ep_idx in range(num_envs)]
            )
            for key in processed_prompt.keys():
                processed_prompt[key] = processed_prompt[key].to(self.device)
            output = self.model(**processed_prompt, use_cache=True)
            self._past_key_values = output.past_key_values
        else:
            self._past_key_values = None

    def get_action(self, observations: np.ndarray) -> np.ndarray:
        # Turn into episode
        num_envs = observations.shape[0]
        observations = np.expand_dims(observations, axis=1).tolist()
        processed = self.processor(
            continuous_observations=observations,
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=self.max_length,  # ensure not to not overflow
        )
        processed = self.collator(
            [{key: processed[key][ep_idx] for key in processed.keys()} for ep_idx in range(num_envs)]
        )
        for key in processed.keys():
            processed[key] = processed[key].to(self.device)
        action_tokens = []

        for _ in range(self._num_act_tokens):
            output = self.model(**processed, use_cache=True, past_key_values=self._past_key_values)
            self._past_key_values = output.past_key_values
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

        # to ensure the KV cache includes the last action token
        output = self.model(**processed, use_cache=True, past_key_values=self._past_key_values)
        self._past_key_values = output.past_key_values
        if self._past_key_values[0][0].shape[2] > self._max_kv_size:
            # remove one step of tokens, to ensure context < 1024
            self._past_key_values = [
                (k[:, :, self._tokens_per_step :], v[:, :, self._tokens_per_step :])
                for (k, v) in self._past_key_values
            ]
        action_tokens = torch.stack(action_tokens, dim=-1)

        # Decode the action tokens
        action = np.array(self.processor.tokenizer.decode_continuous(action_tokens.cpu().numpy()))
        # TODO: Clamp action to be in domain of action space?
        return action
