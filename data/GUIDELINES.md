# Dataset Contribution Guidelines

## Reinforcement Learning (RL) Tasks

### Data Generation

The dataset should be organized episodically. This involves storing observations, actions, and rewards. Please note, the final observation of each episode is not stored.

### Data Storage

It's important to use only the following keys with the designated dtypes:

| Column name               | dtype   | Description                       |
| ------------------------- | ------- | --------------------------------- |
| `text_observations`       | TODO    | Text observations                 |
| `image_observations`      | TODO    | Visual observations, channel last |
| `discrete_observations`   | TODO    | Discrete observations             |
| `continuous_observations` | TODO    | Continuous observations           |
| `discrete_actions`        | TODO    | Discrete actions                  |
| `continuous_actions`      | TODO    | Continuous actions                |
| `rewards`                 | TODO    | Rewards                           |

### Data Splits

We employ two splits: "train" and "test", with roughly 90% and 10% of the data respectively.

## Non-RL Tasks

Non-RL tasks are usually not associated with a domain (for example, `"conceptual-captions"`).

Use the following column names and feature type:

| Column name | Feature                    | Description |
| ----------- | -------------------------- | ----------- |
| `text`      | `datasets.Value("string")` | Text        |
| `images`    | `datasets.Image()`         | Visual      |

If the data comes from an existing dataset, use the same split as the original dataset.