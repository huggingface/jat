from typing import Dict

import numpy as np
from gymnasium import Env, ObservationWrapper, spaces


BABYAI_ENV_NAMES = [
    "BabyAI-GoTo-v0",
    "BabyAI-GoToDoor-v0",
    "BabyAI-GoToImpUnlock-v0",
    "BabyAI-GoToLocal-v0",
    "BabyAI-GoToObj-v0",
    "BabyAI-GoToObjDoor-v0",
    "BabyAI-GoToRedBall-v0",
    "BabyAI-GoToRedBallGrey-v0",
    "BabyAI-GoToRedBallNoDists-v0",
    "BabyAI-GoToRedBlueBall-v0",
    "BabyAI-GoToSeq-v0",
    "BabyAI-Open-v0",
    "BabyAI-OpenDoor-v0",
    "BabyAI-OpenDoorsOrderN4-v0",
    "BabyAI-OpenRedDoor-v0",
    "BabyAI-OpenTwoDoors-v0",
    "BabyAI-ActionObjDoor-v0",
    "BabyAI-FindObjS5-v0",
    "BabyAI-KeyCorridor-v0",
    "BabyAI-MoveTwoAcrossS8N9-v0",
    "BabyAI-OneRoomS8-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-PickupAbove-v0",
    "BabyAI-PickupDist-v0",
    "BabyAI-PickupLoc-v0",
    "BabyAI-UnblockPickup-v0",
    "BabyAI-BossLevel-v0",
    "BabyAI-BossLevelNoUnlock-v0",
    "BabyAI-MiniBossLevel-v0",
    "BabyAI-Synth-v0",
    "BabyAI-SynthLoc-v0",
    "BabyAI-SynthSeq-v0",
    "BabyAI-BlockedUnlockPickup-v0",
    "BabyAI-KeyInBox-v0",
    "BabyAI-Unlock-v0",
    "BabyAI-UnlockLocal-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-UnlockToUnlock-v0",
]


class BabyAIWrapper(ObservationWrapper):
    """
    Wrapper for BabyAI environments.

    Flatten the image and direction observations and concatenate them.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        n_image = self.observation_space["image"].high.flatten()
        n_direction = self.observation_space["direction"].n
        self.observation_space = spaces.Dict(
            {
                "text_observations": env.observation_space.spaces["mission"],
                "discrete_observations": spaces.MultiDiscrete([n_direction, *n_image]),
            }
        )

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        discrete_observations = np.append(observation["direction"], observation["image"].flatten())
        return {
            "text_observations": observation["mission"],
            "discrete_observations": discrete_observations,
        }
