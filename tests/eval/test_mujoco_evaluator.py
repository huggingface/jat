from gia import GiaConfig, GiaModel
from gia.config.arguments import Arguments
from gia.eval.mujoco_evaluator import MujocoEvaluator


def test_mujoco_evaluator():
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")

    evaluator = MujocoEvaluator(args)
    evaluator.env_names = ["InvertedDoublePendulum-v4"]
    evaluator.data_filepaths = ["mujoco-doublependulum"]
    evaluator.evaluate(model)
