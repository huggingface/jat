import gym
import numpy as np
import torch
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoConfig


from gia.config.arguments import Arguments
from gia.datasets.core import load_prompt_dataset
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor

from gia.eval.evaluator import Evaluator
from gia.eval.mappings import DATASET_FILE_MAPPING, TASK_TO_ENV_MAPPING


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

    def _create_buffer(self, num_envs, int_per_seq, num_obs_tokens, num_act_tokens, device):
        buffer = {
            "continuous_observations": torch.zeros(
                (num_envs, int_per_seq, num_obs_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_observations_attention_mask": torch.zeros(
                (num_envs, int_per_seq, num_obs_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_actions": torch.zeros(
                (num_envs, int_per_seq, num_act_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_actions_attention_mask": torch.zeros(
                (num_envs, int_per_seq, num_act_tokens),
                dtype=torch.long,
                device=device,
            ),
        }
        return buffer

    @torch.no_grad()
    def _evaluate_env(self, env_name, dataset_name, model):
        num_envs = 1
        # number of interactions per sequence. Hard-coded for now
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = GiaModel(Arguments()).to(device)
        env = gym.vector.make(env_name, num_envs)
        num_obs_tokens = env.observation_space.shape[1]
        num_act_tokens = env.action_space.shape[1]
        int_per_seq = (self.args.seq_len // (num_obs_tokens + num_act_tokens)) - 1



        prompt_dataset = load_prompt_dataset(dataset_name, args) #  load_prompt_dataset(dataset_name)
        sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset), size=num_envs)

        # Fill (right side) the prompt_buffer with the prompts. Truncate if necessary.

        processor = GiaProcessor(args)
        returns = []
        
        returns = []
        # due to how to KV cache is used, we only can evaluate one env instance at a time
        for ep in tqdm(range(self.args.n_episodes)):
            accum_rewards = []
            obs, info = env.reset()
            done = False
            past_key_values = None
            use_cache = False
            sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset))
            prompt_buffer = self._create_buffer(num_envs,int_per_seq, num_obs_tokens, num_act_tokens, device)
            step_buffer = self._create_buffer(num_envs,1, num_obs_tokens, num_act_tokens, device)
            for key in prompt_buffer.keys():
                length = min(prompt_buffer[key].shape[1], prompt_dataset[key][sampled_prompts_idxs].shape[1])
                prompt_buffer[key][:, -length:] = torch.from_numpy(prompt_dataset[key][sampled_prompts_idxs, -length:]).to(device)

            # update the kv cache from the prompt
            output = model(prompt_buffer, eval=True, use_cache=use_cache, past_key_values=past_key_values)
            past_key_values = output.past_key_values

            while not done:
                obs_tokens = processor({"continuous_observations": obs})["continuous_observations"]
                step_buffer["continuous_observations"] = torch.from_numpy(obs_tokens).unsqueeze(1).to(device)
                step_buffer["continuous_actions_attention_mask"] *= 0     
                
                # still TODO attn masks for obs and actins, buffer resetting
                action = np.zeros((num_envs, num_act_tokens))
                token_shift = 32_000 # TODO: remove?
                for i in range(num_act_tokens):
                    output = model(step_buffer, eval=True, use_cache=use_cache, past_key_values=past_key_values)
                    action_logits = output.logits[:, -num_act_tokens + i, token_shift:]
                    action_tokens = torch.argmax(action_logits, -1) + token_shift
                    step_buffer["continuous_actions"][:, -1, i] = action_tokens
                    action[:, i] = processor.inverse_tokenize_continuous(action_tokens.cpu()).numpy()
                    step_buffer["continuous_actions_attention_mask"][:, -1, i] = 1

                # to ensure the KV cache includes the last action token
                output = model(step_buffer, eval=True, use_cache=use_cache, past_key_values=past_key_values)
                past_key_values = output.past_key_values
                
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                accum_rewards.append(reward)
            
            print(len(accum_rewards))
            returns.append(sum(accum_rewards))
        env.close()

        return returns


if __name__ == "__main__":
    args = Arguments(output_dir="tmp")
    model = GiaModel(args).to("cuda")
    
    evaluator = MujocoEvaluator(args)
    evaluator.evaluate(model)