from gia import GiaModel, GiaConfig
from gia.config.arguments import Arguments
from gia.eval.mujoco_evaluator import MujocoEvaluator


def test_mujoco_evaluator():
    config = GiaConfig()

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")
    model = GiaModel(config)

    evaluator = MujocoEvaluator(args)
    evaluator.env_names = ["InvertedDoublePendulum-v4"]
    evaluator.data_filepaths = ["mujoco-doublependulum"]
    evaluator.evaluate(model)
