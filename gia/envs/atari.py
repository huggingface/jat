import envpool

from gia.config.config import Config
from gia.envs.wrappers import BatchedRecordEpisodeStatistics, EnvPoolResetFixWrapper

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariSpec("atari_alien", "AlienNoFrameskip-v4"),
    AtariSpec("atari_amidar", "AmidarNoFrameskip-v4"),
    AtariSpec("atari_assault", "AssaultNoFrameskip-v4"),
    AtariSpec("atari_asterix", "AsterixNoFrameskip-v4"),
    AtariSpec("atari_asteroid", "AsteroidsNoFrameskip-v4"),
    AtariSpec("atari_atlantis", "AtlantisNoFrameskip-v4"),
    AtariSpec("atari_bankheist", "BankHeistNoFrameskip-v4"),
    AtariSpec("atari_battlezone", "BattleZoneNoFrameskip-v4"),
    AtariSpec("atari_beamrider", "BeamRiderNoFrameskip-v4"),
    AtariSpec("atari_berzerk", "BerzerkNoFrameskip-v4"),
    AtariSpec("atari_bowling", "BowlingNoFrameskip-v4"),
    AtariSpec("atari_boxing", "BoxingNoFrameskip-v4"),
    AtariSpec("atari_breakout", "BreakoutNoFrameskip-v4"),
    AtariSpec("atari_centipede", "CentipedeNoFrameskip-v4"),
    AtariSpec("atari_choppercommand", "ChopperCommandNoFrameskip-v4"),
    AtariSpec("atari_crazyclimber", "CrazyClimberNoFrameskip-v4"),
    AtariSpec("atari_defender", "DefenderNoFrameskip-v4"),
    AtariSpec("atari_demonattack", "DemonAttackNoFrameskip-v4"),
    AtariSpec("atari_doubledunk", "DoubleDunkNoFrameskip-v4"),
    AtariSpec("atari_enduro", "EnduroNoFrameskip-v4"),
    AtariSpec("atari_fishingderby", "FishingDerbyNoFrameskip-v4"),
    AtariSpec("atari_freeway", "FreewayNoFrameskip-v4"),
    AtariSpec("atari_frostbite", "FrostbiteNoFrameskip-v4"),
    AtariSpec("atari_gopher", "GopherNoFrameskip-v4"),
    AtariSpec("atari_gravitar", "GravitarNoFrameskip-v4"),
    AtariSpec("atari_hero", "HeroNoFrameskip-v4"),
    AtariSpec("atari_icehockey", "IceHockeyNoFrameskip-v4"),
    AtariSpec("atari_jamesbond", "JamesbondNoFrameskip-v4"),
    AtariSpec("atari_kangaroo", "KangarooNoFrameskip-v4"),
    AtariSpec("atari_krull", "KrullNoFrameskip-v4"),
    AtariSpec("atari_kongfumaster", "KungFuMasterNoFrameskip-v4"),
    AtariSpec("atari_montezuma", "MontezumaRevengeNoFrameskip-v4", default_timeout=18000),
    AtariSpec("atari_mspacman", "MsPacmanNoFrameskip-v4"),
    AtariSpec("atari_namethisgame", "NameThisGameNoFrameskip-v4"),
    AtariSpec("atari_phoenix", "PhoenixNoFrameskip-v4"),
    AtariSpec("atari_pitfall", "PitfallNoFrameskip-v4"),
    AtariSpec("atari_pong", "PongNoFrameskip-v4"),
    AtariSpec("atari_privateye", "PrivateEyeNoFrameskip-v4"),
    AtariSpec("atari_qbert", "QbertNoFrameskip-v4"),
    AtariSpec("atari_riverraid", "RiverraidNoFrameskip-v4"),
    AtariSpec("atari_roadrunner", "RoadRunnerNoFrameskip-v4"),
    AtariSpec("atari_robotank", "RobotankNoFrameskip-v4"),
    AtariSpec("atari_seaquest", "SeaquestNoFrameskip-v4"),
    AtariSpec("atari_skiing", "SkiingNoFrameskip-v4"),
    AtariSpec("atari_solaris", "SolarisNoFrameskip-v4"),
    AtariSpec("atari_spaceinvaders", "SpaceInvadersNoFrameskip-v4"),
    AtariSpec("atari_stargunner", "StarGunnerNoFrameskip-v4"),
    AtariSpec("atari_surround", "SurroundNoFrameskip-v4"),
    AtariSpec("atari_tennis", "TennisNoFrameskip-v4"),
    AtariSpec("atari_timepilot", "TimePilotNoFrameskip-v4"),
    AtariSpec("atari_tutankham", "TutankhamNoFrameskip-v4"),
    AtariSpec("atari_upndown", "UpNDownNoFrameskip-v4"),
    AtariSpec("atari_venture", "VentureNoFrameskip-v4"),
    AtariSpec("atari_videopinball", "VideoPinballNoFrameskip-v4"),
    AtariSpec("atari_wizardofwor", "WizardOfWorNoFrameskip-v4"),
    AtariSpec("atari_yarsrevenge", "YarsRevengeNoFrameskip-v4"),
    AtariSpec("atari_zaxxon", "ZaxxonNoFrameskip-v4"),
]
ENVPOOL_ATARI_ENVS = [
    AtariSpec(
        spec.name,
        spec.env_id.replace("NoFrameskip-v4", "-v5"),
        default_timeout=spec.default_timeout,
    )
    for spec in ATARI_ENVS
]


def atari_env_by_name(name):
    for atari_spec in ENVPOOL_ATARI_ENVS:
        if atari_spec.name == name:
            return atari_spec
    raise Exception("Unknown Atari env")


def make_atari_env(env_name: str, config: Config, env_config, render_mode=None):
    atari_spec = atari_env_by_name(env_name)
    env_kwargs = dict()

    if atari_spec.default_timeout is not None:
        # envpool max_episode_steps does not take into account frameskip. see https://github.com/sail-sg/envpool/issues/195
        env_kwargs["max_episode_steps"] = atari_spec.default_timeout // 4
    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id

    env = envpool.make(
        atari_spec.env_id,
        env_type="gym",
        num_envs=config.envs.agents_per_env,
        reward_clip=True,
        episodic_life=True,
        frame_skip=config.envs.frameskip,
        **env_kwargs,
    )
    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=config.envs.agents_per_env)
    env.num_envs = config.envs.agents_per_env
    return env
