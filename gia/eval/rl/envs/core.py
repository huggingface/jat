import gymnasium as gym
import numpy as np
from gymnasium import Env, ObservationWrapper, spaces
from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
)


TASK_TO_ENV_MAPPING = {
    "atari-alien": "Alien-v4",
    "atari-amidar": "Amidar-v4",
    "atari-assault": "Assault-v4",
    "atari-asterix": "Asterix-v4",
    "atari-asteroids": "Asteroids-v4",
    "atari-atlantis": "Atlantis-v4",
    "atari-bankheist": "BankHeist-v4",
    "atari-battlezone": "BattleZone-v4",
    "atari-beamrider": "BeamRider-v4",
    "atari-berzerk": "Berzerk-v4",
    "atari-bowling": "Bowling-v4",
    "atari-boxing": "Boxing-v4",
    "atari-breakout": "Breakout-v4",
    "atari-centipede": "Centipede-v4",
    "atari-choppercommand": "ChopperCommand-v4",
    "atari-crazyclimber": "CrazyClimber-v4",
    "atari-defender": "Defender-v4",
    "atari-demonattack": "DemonAttack-v4",
    "atari-doubledunk": "DoubleDunk-v4",
    "atari-enduro": "Enduro-v4",
    "atari-fishingderby": "FishingDerby-v4",
    "atari-freeway": "Freeway-v4",
    "atari-frostbite": "Frostbite-v4",
    "atari-gopher": "Gopher-v4",
    "atari-gravitar": "Gravitar-v4",
    "atari-hero": "Hero-v4",
    "atari-icehockey": "IceHockey-v4",
    "atari-jamesbond": "Jamesbond-v4",
    "atari-kangaroo": "Kangaroo-v4",
    "atari-krull": "Krull-v4",
    "atari-kungfumaster": "KungFuMaster-v4",
    "atari-montezumarevenge": "MontezumaRevenge-v4",
    "atari-mspacman": "MsPacman-v4",
    "atari-namethisgame": "NameThisGame-v4",
    "atari-phoenix": "Phoenix-v4",
    "atari-pitfall": "Pitfall-v4",
    "atari-pong": "Pong-v4",
    "atari-privateeye": "PrivateEye-v4",
    "atari-qbert": "Qbert-v4",
    "atari-riverraid": "Riverraid-v4",
    "atari-roadrunner": "RoadRunner-v4",
    "atari-robotank": "Robotank-v4",
    "atari-seaquest": "Seaquest-v4",
    "atari-skiing": "Skiing-v4",
    "atari-solaris": "Solaris-v4",
    "atari-spaceinvaders": "SpaceInvaders-v4",
    "atari-stargunner": "StarGunner-v4",
    "atari-surround": "ALE/Surround-v5",
    "atari-tennis": "Tennis-v4",
    "atari-timepilot": "TimePilot-v4",
    "atari-tutankham": "Tutankham-v4",
    "atari-upndown": "UpNDown-v4",
    "atari-venture": "Venture-v4",
    "atari-videopinball": "VideoPinball-v4",
    "atari-wizardofwor": "WizardOfWor-v4",
    "atari-yarsrevenge": "YarsRevenge-v4",
    "atari-zaxxon": "Zaxxon-v4",
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
    "babyai-key-in-box": "BabyAI-KeyInBox-v0",
    "babyai-mini-boss-level": "BabyAI-MiniBossLevel-v0",
    "babyai-move-two-across": "BabyAI-MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "BabyAI-OneRoomS8-v0",
    "babyai-open-door": "BabyAI-OpenDoor-v0",
    "babyai-open-doors-order": "BabyAI-OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "BabyAI-OpenRedDoor-v0",
    "babyai-open-two-doors": "BabyAI-OpenTwoDoors-v0",
    "babyai-open": "BabyAI-Open-v0",
    "babyai-pickup-above": "BabyAI-PickupAbove-v0",
    "babyai-pickup-dist": "BabyAI-PickupDist-v0",
    "babyai-pickup-loc": "BabyAI-PickupLoc-v0",
    "babyai-pickup": "BabyAI-Pickup-v0",
    "babyai-synth-loc": "BabyAI-SynthLoc-v0",
    "babyai-synth-seq": "BabyAI-SynthSeq-v0",
    "babyai-synth": "BabyAI-Synth-v0",
    "babyai-unblock-pickup": "BabyAI-UnblockPickup-v0",
    "babyai-unlock-local": "BabyAI-UnlockLocal-v0",
    "babyai-unlock-pickup": "BabyAI-UnlockPickup-v0",
    "babyai-unlock-to-unlock": "BabyAI-UnlockToUnlock-v0",
    "babyai-unlock": "BabyAI-Unlock-v0",
    "metaworld-assembly": "assembly-v2",
    "metaworld-basketball": "basketball-v2",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-button-press-topdown-wall": "button-press-topdown-wall-v2",
    "metaworld-button-press-topdown": "button-press-topdown-v2",
    "metaworld-button-press-wall": "button-press-wall-v2",
    "metaworld-button-press": "button-press-v2",
    "metaworld-coffee-button": "coffee-button-v2",
    "metaworld-coffee-pull": "coffee-pull-v2",
    "metaworld-coffee-push": "coffee-push-v2",
    "metaworld-dial-turn": "dial-turn-v2",
    "metaworld-disassemble": "disassemble-v2",
    "metaworld-door-close": "door-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-open": "door-open-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-drawer-close": "drawer-close-v2",
    "metaworld-drawer-open": "drawer-open-v2",
    "metaworld-faucet-close": "faucet-close-v2",
    "metaworld-faucet-open": "faucet-open-v2",
    "metaworld-hammer": "hammer-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "metaworld-handle-press-side": "handle-press-side-v2",
    "metaworld-handle-press": "handle-press-v2",
    "metaworld-handle-pull-side": "handle-pull-side-v2",
    "metaworld-handle-pull": "handle-pull-v2",
    "metaworld-lever-pull": "lever-pull-v2",
    "metaworld-peg-insert-side": "peg-insert-side-v2",
    "metaworld-peg-unplug-side": "peg-unplug-side-v2",
    "metaworld-pick-out-of-hole": "pick-out-of-hole-v2",
    "metaworld-pick-place-wall": "pick-place-wall-v2",
    "metaworld-pick-place": "pick-place-v2",
    "metaworld-plate-slide-back-side": "plate-slide-back-side-v2",
    "metaworld-plate-slide-back": "plate-slide-back-v2",
    "metaworld-plate-slide-side": "plate-slide-side-v2",
    "metaworld-plate-slide": "plate-slide-v2",
    "metaworld-push-back": "push-back-v2",
    "metaworld-push-wall": "push-wall-v2",
    "metaworld-push": "push-v2",
    "metaworld-reach-wall": "reach-wall-v2",
    "metaworld-reach": "reach-v2",
    "metaworld-shelf-place": "shelf-place-v2",
    "metaworld-soccer": "soccer-v2",
    "metaworld-stick-pull": "stick-pull-v2",
    "metaworld-stick-push": "stick-push-v2",
    "metaworld-sweep-into": "sweep-into-v2",
    "metaworld-sweep": "sweep-v2",
    "metaworld-window-close": "window-close-v2",
    "metaworld-window-open": "window-open-v2",
    "mujoco-ant": "Ant-v4",
    "mujoco-doublependulum": "InvertedDoublePendulum-v4",
    "mujoco-halfcheetah": "HalfCheetah-v4",
    "mujoco-hopper": "Hopper-v4",
    "mujoco-humanoid": "Humanoid-v4",
    "mujoco-pendulum": "InvertedPendulum-v4",
    "mujoco-pusher": "Pusher-v4",
    "mujoco-reacher": "Reacher-v4",
    "mujoco-standup": "HumanoidStandup-v4",
    "mujoco-swimmer": "Swimmer-v4",
    "mujoco-walker": "Walker2d-v4",
}


def get_task_names():
    """
    Get all the environment ids.

    Returns:
        list: List of environment ids
    """
    return list(TASK_TO_ENV_MAPPING.keys())


class AtariDictObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {"image_observations": spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)}
        )

    def observation(self, observation):
        observations = np.transpose(observation, (1, 2, 0))  # make channel last
        return {"image_observations": observations}


def make_atari(task_name: str):
    kwargs = {"frameskip": 1, "repeat_action_probability": 0.0}
    if task_name == "atari-montezumarevenge":
        kwargs["max_episode_steps"] = 18_000
    env = gym.make(TASK_TO_ENV_MAPPING[task_name], **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = NumpyObsWrapper(env)
    env = AtariDictObservationWrapper(env)
    return env


class BabyAIDictObservationWrapper(ObservationWrapper):
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

    def observation(self, observation):
        discrete_observations = np.append(observation["direction"], observation["image"].flatten())
        return {
            "text_observations": observation["mission"],
            "discrete_observations": discrete_observations,
        }


def make_babyai(task_name: str):
    env = gym.make(TASK_TO_ENV_MAPPING[task_name])
    env = BabyAIDictObservationWrapper(env)
    return env


class ContinuousObservationDictWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"continuous_observations": env.observation_space})

    def observation(self, observation):
        return {"continuous_observations": observation}


def make_metaworld(task_name: str):
    import metaworld  # noqa

    env = gym.make(TASK_TO_ENV_MAPPING[task_name])
    env = ContinuousObservationDictWrapper(env)
    return env


def make_mujoco(task_name: str):
    env = gym.make(TASK_TO_ENV_MAPPING[task_name])
    env = ContinuousObservationDictWrapper(env)
    return env


def make(task_name: str):
    if task_name.startswith("atari"):
        return make_atari(task_name)

    elif task_name.startswith("babyai"):
        return make_babyai(task_name)

    elif task_name.startswith("metaworld"):
        return make_metaworld(task_name)

    elif task_name.startswith("mujoco"):
        return make_mujoco(task_name)
    else:
        raise ValueError(f"Unknown task name: {task_name}. Available task names: {get_task_names()}")
