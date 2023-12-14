import argparse
import signal

import gymnasium as gym
import numpy as np
from bot_agent import BabyAIBot as Bot
from datasets import Dataset
from datasets.features import Features, Sequence, Value


TASK_NAME_TO_ENV_ID = {
    "babyai-action-obj-door": "BabyAI-ActionObjDoor-v0",
    "babyai-blocked-unlock-pickup": "BabyAI-BlockedUnlockPickup-v0",
    "babyai-boss-level-no-unlock": "BabyAI-BossLevelNoUnlock-v0",
    "babyai-boss-level": "BabyAI-BossLevel-v0",
    "babyai-find-obj-s5": "BabyAI-FindObjS5-v0",
    "babyai-go-to-door": "BabyAI-GoToDoor-v0",
    "babyai-go-to-imp-unlock": "BabyAI-GoToImpUnlock-v0",
    "babyai-go-to-local": "BabyAI-GoToLocal-v0",
    "babyai-go-to-obj-door": "BabyAI-GoToObjDoor-v0",
    "babyai-go-to-obj": "BabyAI-GoToObj-v0",
    "babyai-go-to-red-ball-grey": "BabyAI-GoToRedBallGrey-v0",
    "babyai-go-to-red-ball-no-dists": "BabyAI-GoToRedBallNoDists-v0",
    "babyai-go-to-red-ball": "BabyAI-GoToRedBall-v0",
    "babyai-go-to-red-blue-ball": "BabyAI-GoToRedBlueBall-v0",
    "babyai-go-to-seq": "BabyAI-GoToSeq-v0",
    "babyai-go-to": "BabyAI-GoTo-v0",
    "babyai-key-corridor": "BabyAI-KeyCorridor-v0",
    "babyai-mini-boss-level": "BabyAI-MiniBossLevel-v0",
    "babyai-move-two-across-s8n9": "BabyAI-MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "BabyAI-OneRoomS8-v0",
    "babyai-open-door": "BabyAI-OpenDoor-v0",
    "babyai-open-doors-order-n4": "BabyAI-OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "BabyAI-OpenRedDoor-v0",
    "babyai-open-two-doors": "BabyAI-OpenTwoDoors-v0",
    "babyai-open": "BabyAI-Open-v0",
    "babyai-pickup-above": "BabyAI-PickupAbove-v0",
    "babyai-pickup-dist": "BabyAI-PickupDist-v0",
    "babyai-pickup-loc": "BabyAI-PickupLoc-v0",
    "babyai-pickup": "BabyAI-Pickup-v0",
    "babyai-put-next-local": "BabyAI-PutNextLocal-v0",
    "babyai-put-next": "BabyAI-PutNextS7N4-v0",
    "babyai-synth-loc": "BabyAI-SynthLoc-v0",
    "babyai-synth-seq": "BabyAI-SynthSeq-v0",
    "babyai-synth": "BabyAI-Synth-v0",
    "babyai-unblock-pickup": "BabyAI-UnblockPickup-v0",
    "babyai-unlock-local": "BabyAI-UnlockLocal-v0",
    "babyai-unlock-pickup": "BabyAI-UnlockPickup-v0",
    "babyai-unlock-to-unlock": "BabyAI-UnlockToUnlock-v0",
    "babyai-unlock": "BabyAI-Unlock-v0",
}


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def call_with_timeout(func, args=[], kwargs={}, timeout_duration=1.0):
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # Set the interval timer
    signal.setitimer(signal.ITIMER_REAL, timeout_duration, 0)

    try:
        result = func(*args, **kwargs)
    except TimeoutError as e:
        raise e
    finally:
        # Disable the interval timer
        signal.setitimer(signal.ITIMER_REAL, 0)
    return result


def reset_env_and_policy(env):
    obs, _ = env.reset()
    return obs, Bot(env.env)


def generate_episode(env):
    episode = {"text_observations": [], "discrete_observations": [], "discrete_actions": [], "rewards": []}
    observation, policy = reset_env_and_policy(env)
    t = 0
    while True:
        episode["text_observations"].append(observation["mission"])
        flattened_symbolic_obs = observation["image"].flatten()
        concatenated_discrete_obs = np.append(observation["direction"], flattened_symbolic_obs)
        episode["discrete_observations"].append(concatenated_discrete_obs)
        action = call_with_timeout(policy.replan, timeout_duration=0.02)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode["discrete_actions"].append(int(action))
        episode["rewards"].append(reward)

        if terminated or truncated:
            break

        t += 1
        if t > 1000:
            raise Exception("Episode too long")
    return episode


def create_babyai_dataset(task_name, max_num_episodes):
    env_id = TASK_NAME_TO_ENV_ID[task_name]
    env = gym.make(env_id)
    data = {"text_observations": [], "discrete_observations": [], "discrete_actions": [], "rewards": []}

    print("Starting trajectories generation")
    while len(data["rewards"]) < max_num_episodes:
        print(f"Episode {len(data['rewards']) + 1}/{max_num_episodes}")

        try:
            episode = generate_episode(env)
        except Exception as e:
            print(e)
            continue

        for k, v in episode.items():
            data[k].append(v)

    print(f"Finished generation. Generated {len(data['rewards'])} transitions.")

    features = Features(
        {
            "text_observations": Sequence(Value("string")),
            "discrete_observations": Sequence(Sequence(Value("int64"))),
            "discrete_actions": Sequence(Value("int64")),
            "rewards": Sequence(Value("float32")),
        }
    )
    dataset = Dataset.from_dict(data, features)
    print("Saving dataset...")
    dataset.save_to_disk(task_name)
    print("Saved dataset!")

    print("Pushing dataset to hub...")
    dataset = dataset.train_test_split(test_size=0.02)
    dataset.push_to_hub("jat-project/jat-dataset", task_name, branch="additional_babyai_tasks")
    print("Pushed dataset to hub!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--max_num_episodes", default=100_000, type=int)
    args = parser.parse_args()

    create_babyai_dataset(task_name=args.task_name, max_num_episodes=args.max_num_episodes)
