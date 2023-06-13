import gym
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from gia.config.arguments import Arguments
from gia.datasets import GiaDataCollator, Prompter
from gia.eval.evaluator import Evaluator
from gia.eval.mappings import DATASET_FILE_MAPPING, TASK_TO_ENV_MAPPING
from gia import GiaConfig, GiaModel
from gia.processing import GiaProcessor


def make_mujoco_env(env_name, render_mode=None):
    return gym.make(env_name, render_mode=render_mode)


class MujocoEvaluator(Evaluator):
    def __init__(self, args: Arguments):
        self.task = "mujoco"
        self.env_names = TASK_TO_ENV_MAPPING[self.task]
        self.data_filepaths = DATASET_FILE_MAPPING[self.task]
        self.args = args

    def evaluate(self, model: GiaModel):
        stats = {}
        for env_name, dataset_name in zip(self.env_names, self.data_filepaths):
            stats[env_name] = self._evaluate_env(env_name, dataset_name, model)
        return stats

    @torch.no_grad()
    def _evaluate_env(self, env_name: str, dataset_name: str, model: GiaModel):
        num_envs = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = gym.vector.make(env_name, num_envs)
        num_obs_tokens = env.observation_space.shape[1]
        num_act_tokens = env.action_space.shape[1]
        tokens_per_step = num_obs_tokens + num_act_tokens + int(self.args.use_separator)
        int_per_seq = (self.args.seq_len // tokens_per_step) - 1
        max_kv_size = int_per_seq * tokens_per_step

        dataset = load_dataset("gia-project/gia-dataset", dataset_name, split="train")
        prompter = Prompter(
            dataset,
            min_prompt_len=int_per_seq,
            max_prompt_len=int_per_seq,
        )

        prompts = prompter.generate_prompts(self.args.n_episodes)

        processor = GiaProcessor()
        collator = GiaDataCollator()
        token_shift = processor.tokenizer.token_shift

        returns = []
        # due to how to KV cache is used, we only can evaluate one env instance at a time
        for ep in tqdm(range(self.args.n_episodes)):
            prompt_observations = np.array([prompts["continuous_observations"][ep]])
            prompt_actions = np.array([prompts["continuous_actions"][ep]])

            processed_prompt = processor(
                continuous_observations=prompt_observations,
                continuous_actions=prompt_actions,
                padding=False,
                truncation="max_length",
                truncation_side="left",
                max_length=args.seq_len - num_act_tokens,  # ensure not to overflow when the actions are added
            )

            # TODO:
            # - confirm attention masks are not needed in this setting

            processed_prompt = collator([{key: processed_prompt[key][0] for key in processed_prompt.keys()}])
            for key in processed_prompt.keys():
                processed_prompt[key] = processed_prompt[key].to(device)
            output = model(**processed_prompt, use_cache=True)
            past_key_values = output.past_key_values

            accum_rewards = []
            done = False
            obs, info = env.reset()

            while not done:
                # Compute the output of the model
                processed = processor(
                    continuous_observations=[obs],
                    continuous_actions=[],
                    padding=False,
                    truncation="max_length",
                    truncation_side="left",
                    max_length=args.seq_len - num_act_tokens,  # ensure not to overflow when the actions are added
                )
                processed = collator([{key: processed[key][0] for key in processed.keys()}])
                for key in processed.keys():
                    processed[key] = processed[key].to(device)
                action_tokens = []

                for i in range(num_act_tokens):
                    output = model(**processed, use_cache=True, past_key_values=past_key_values)
                    past_key_values = output.past_key_values
                    action_logits = output.logits[:, -1, token_shift:]

                    action_token = torch.argmax(action_logits, -1) + token_shift
                    action_tokens.append(action_token)

                    processed["input_ids"] = action_token[None, :]
                    if i == 0:  # only needs to be done once
                        processed["loss_mask"] = torch.ones(1, 1, dtype=torch.bool, device=device)
                        processed["input_types"] = torch.zeros(1, 1, dtype=torch.int64, device=device)
                        processed["local_positions"] = -torch.ones(1, 1, dtype=torch.int64, device=device)

                # to ensure the KV cache includes the last action token
                output = model(**processed, use_cache=True, past_key_values=past_key_values)
                past_key_values = output.past_key_values
                if past_key_values[0][0].shape[2] > max_kv_size:
                    # remove one step of tokens, to ensure context < 1024
                    past_key_values = [
                        (k[:, :, tokens_per_step:], v[:, :, tokens_per_step:]) for (k, v) in past_key_values
                    ]
                action_tokens = torch.stack(action_tokens, dim=-1)

                # Decode the action tokens
                action = processor.tokenizer.decode_continuous(action_tokens.cpu().numpy())
                # TODO: Clamp action to be in domain of action space?
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                accum_rewards.append(reward[0])

            returns.append(sum(accum_rewards))
        env.close()

        return returns


if __name__ == "__main__":

    config = GiaConfig()

    args = Arguments(output_dir="tmp", n_episodes=2)
    model = GiaModel(config).to("cuda")

    evaluator = MujocoEvaluator(args)
    evaluator.evaluate(model)
