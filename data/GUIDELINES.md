# Dataset Contribution Guidelines

## Reinforcement Learning (RL) Tasks

### Data Generation

The dataset should be organized episodically. This involves storing observations, actions, and rewards. Please note, the final observation of each episode is not stored.

Here's an example script for data generation:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

discrete_actions = []
continuous_observations = []
rewards = []

for ep_idx in range(100):
    ep_discrete_actions = []
    ep_continuous_observations = []
    ep_rewards = []

    obs, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store s_t, a_t, r_{t+1}
        ep_continuous_observations.append(obs)
        ep_discrete_actions.append(action)
        ep_rewards.append(reward)

        obs = next_obs

    discrete_actions.append(ep_discrete_actions)
    continuous_observations.append(ep_continuous_observations)
    rewards.append(ep_rewards)
```

### Data Storage

The data should be stored as npz files.

```python
import numpy as np

# Convert discrete values to int64 and continuous values to float32
discrete_actions = [np.array(ep, dtype=np.int64) for ep in discrete_actions]
continuous_observations = [np.array(ep, dtype=np.float32) for ep in continuous_observations]
rewards = [np.array(ep, dtype=np.float32) for ep in rewards]

# Create the dataset
dataset = {
    "discrete_actions": np.array(discrete_actions, dtype=object),  # object dtype for variable length arrays
    "continuous_observations": np.array(continuous_observations, dtype=object),
    "rewards": np.array(rewards, dtype=object),
}

# Save the dataset
np.savez_compressed("train.npz", **dataset)

# To load:
# dataset = np.load("train.npz", allow_pickle=True)
```

It's essential to use only the following keys with the designated dtypes:

| Column name               | dtype        | Description                       |
| ------------------------- | ------------ | --------------------------------- |
| `text_observations`       | `str`        | Text observations                 |
| `image_observations`      | `np.uint8`   | Visual observations, channel last |
| `discrete_observations`   | `np.int64`   | Discrete observations             |
| `continuous_observations` | `np.float32` | Continuous observations           |
| `discrete_actions`        | `np.int64`   | Discrete actions                  |
| `continuous_actions`      | `np.float32` | Continuous actions                |
| `rewards`                 | `np.float32` | Rewards                           |


### Data Splits

We employ two splits: "train" and "test", with roughly 90% and 10% of the data respectively.

### Data Storage Locations

In RL, a task is a subset of a domain (for instance, `"ant"` is a task within the `"mujoco"` domain). The task data should be stored in the folder `"my_domain/my_task"`. Adhere to the naming convention of lowercase, with a hyphen separating domain and task, such as `"mujoco-ant"`. Exclude the version number; for example, `"Ant-v2"` becomes `"mujoco-ant"`.

The repository structure should be as follows:

```
gia-dataset
├── data
│   └── domain
│       ├── task1
│       │   ├── train.npz
│       │   └── test.npz
│       └── task2
│           ├── train.npz
│           └── test.npz
├── README.md
└── gia-dataset.py
```

### Builder Configuration

Add a builder configuration for each task in `gia-dataset.py`. You may use existing configurations as a reference.

### Testing

Assume you've pushed the dataset to the branch `"my_branch"`.

1. Update the `_BASE_URL` in `gia-dataset.py` with your branch (remember to revert this change before merging to main).

```python
_BASE_URL = "https://huggingface.co/datasets/gia-project/gia-dataset/resolve/my_branch/data"
```

2. Run the test:

```python
from datasets import load_dataset

load_dataset("gia-project/gia-dataset", "my_task", revision="my_branch")
```


## Non-RL Tasks

Non-RL tasks are usually not associated with a domain (for example, `"conceptual-captions"`). In this case, the task data is stored in the `"data/my_task"` folder.

Data storage and builder config depends on the task, so please refer to the `datasets` documentation.

Use the following column names and feature type:

| Column name | Feature                    | Description |
| ----------- | -------------------------- | ----------- |
| `text`      | `datasets.Value("string")` | Text        |
| `images`    | `datasets.Image()`         | Visual      |

If the data comes from an existing dataset, use the same split as the original dataset.