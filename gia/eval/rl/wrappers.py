from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0.

    Adapted from Stable-Baselines3.

    Args:
        env (`gym.Env`):
            The environment to wrap.
        noop_max (`int`):
            The maximum number of no-ops to perform.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[self.noop_action] == "NOOP"

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            observation, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated | truncated:
                observation, info = self.env.reset(**kwargs)
        return observation, info


class FireResetEnv(gym.Wrapper):
    """
    Take FIRE action on reset for environments that are fixed until firing.

    Adapted from Stable-Baselines3.

    Args:
        env (`gym.Env`):
            The environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.env.reset(**kwargs)
        observation, reward, terminated, truncated, info = self.env.step(1)
        if terminated | truncated:
            self.env.reset(**kwargs)
        observation, reward, terminated, truncated, info = self.env.step(2)
        if terminated | truncated:
            observation, info = self.env.reset(**kwargs)
        return observation, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over (lives exhausted).
    Done by DeepMind for the DQN and co. since it helps value estimation.
    This way all states are still reachable even though lives are episodic,
    and the learner need not know about any of this behind-the-scenes.

    Adapted from Stable-Baselines3.

    Args:
        env (`gym.Env`):
            The environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.inner_game_over = True

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.inner_game_over = terminated | truncated
        # Check current lives, make loss of life terminal, then update lives to handle bonus lives.
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # For Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if self.inner_game_over:
            observation, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            observation, reward, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return observation, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping).

    Adapted from Stable-Baselines3.

    Args:
        env (`gym.Env`):
            The environment to wrap.
        skip (`int`):
            The number of frames to skip.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        info = {}
        terminated = truncated = False
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            if i == self.skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated | truncated:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    Adapted from Stable-Baselines3.

    Args:
        env (`gym.Env`):
            The environment to wrap.
    """

    def reward(self, reward: float) -> float:
        return np.sign(reward)


class NumpyObsWrapper(gym.ObservationWrapper):
    """
    RL algorithm generally expects numpy arrays or Tensors as observations. Atari envs for example return
    LazyFrames which need to be converted to numpy arrays before we actually use them.
    """

    def observation(self, observation: Any) -> np.ndarray:
        return np.array(observation)


class RenderMission(gym.Wrapper):
    """
    Wrapper to add mission in the RGB rendering for BabyAI.
    """

    @staticmethod
    def add_text_to_image(image, text, position=(10, 5), font_size=20, text_color=(255, 255, 255)):
        """
        Add text to an RGB image represented as a NumPy array and return the modified image as a NumPy array.

        Args:
            image (numpy.ndarray): The input RGB image as a NumPy array.
            text (str): The text to be added to the image.
            position (tuple): The (x, y) coordinates of the top-left corner of the text.
            font_size (int): The font size for the text.
            text_color (tuple): The RGB color code for the text color.

        Returns:
            numpy.ndarray: The modified RGB image as a NumPy array.
        """
        # Convert the input NumPy array to a PIL Image
        image = Image.fromarray(np.uint8(image))

        # Create a drawing context on the image
        draw = ImageDraw.Draw(image)

        # Use the default font
        font = ImageFont.load_default().font_variant(size=font_size)

        # Add the text to the image
        draw.text(position, text, fill=text_color, font=font)

        # Convert the modified image back to a NumPy array
        modified_image_np = np.array(image)

        return modified_image_np

    def render(self):
        img = super().render()
        if img is not None:
            img = self.add_text_to_image(img, self.mission)
        return img
