import numpy as np
import torch
from datasets import load_dataset

from gia.datasets import GiaDataCollator, Prompter
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor


class GiaAgent:
    def __init__(
        self,
        dataset_name: str,
        model: GiaModel,
        obs_space,
        action_space,
        use_separator: bool = True,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_obs_tokens = obs_space.shape[1]
        self._num_act_tokens = action_space.shape[1]
        self._tokens_per_step = self._num_obs_tokens + self._num_act_tokens + int(use_separator)
        self._int_per_seq = (self.model.config.seq_len // self._tokens_per_step) - 1

        dataset = load_dataset("gia-project/gia-dataset", dataset_name, split="test")
        self.prompter = Prompter(
            dataset,
            min_prompt_len=self._int_per_seq,
            max_prompt_len=self._int_per_seq,
        )

        self._max_kv_size = self._int_per_seq * self._tokens_per_step

        self.processor = GiaProcessor()
        self.collator = GiaDataCollator()
        self.token_shift = self.processor.tokenizer.token_shift

    def reset(self) -> None:
        prompt = self.prompter.generate_prompts(1)
        prompt_observations = np.array([prompt["continuous_observations"][0]])
        prompt_actions = np.array([prompt["continuous_actions"][0]])

        processed_prompt = self.processor(
            continuous_observations=prompt_observations,
            continuous_actions=prompt_actions,
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=self.model.config.seq_len - self._num_act_tokens,
        )

        processed_prompt = self.collator([{key: processed_prompt[key][0] for key in processed_prompt.keys()}])
        for key in processed_prompt.keys():
            processed_prompt[key] = processed_prompt[key].to(self.device)
        output = self.model(**processed_prompt, use_cache=True)
        self._past_key_values = output.past_key_values

    def get_action(self, obs) -> np.ndarray:
        processed = self.processor(
            continuous_observations=[obs],
            continuous_actions=[],
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=self.model.config.seq_len - self._num_act_tokens,  # ensure not to not overflow
        )
        processed = self.collator([{key: processed[key][0] for key in processed.keys()}])
        for key in processed.keys():
            processed[key] = processed[key].to(self.device)
        action_tokens = []

        for i in range(self._num_act_tokens):
            output = self.model(**processed, use_cache=True, past_key_values=self._past_key_values)
            self._past_key_values = output.past_key_values
            action_logits = output.logits[:, -1, self.token_shift :]

            action_token = torch.argmax(action_logits, -1) + self.token_shift
            action_tokens.append(action_token)

            processed["input_ids"] = action_token[None, :]
            if i == 0:  # only needs to be done once
                processed["loss_mask"] = torch.ones(1, 1, dtype=torch.bool, device=self.device)
                processed["input_types"] = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
                processed["local_positions"] = -torch.ones(1, 1, dtype=torch.int64, device=self.device)

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
