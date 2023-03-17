from gymnasium import spaces
from gymnasium.core import ObservationWrapper


class AddRGBImgPartialObsWrapper(ObservationWrapper):
    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict({**self.observation_space.spaces, "rgb_image": new_image_space})

    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True)

        return {**obs, "rgb_image": rgb_img_partial}
