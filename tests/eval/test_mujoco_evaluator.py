from gia import GiaConfig, GiaModel
from gia.config.arguments import Arguments
from gia.eval.mujoco_evaluator import MujocoEvaluator


def test_mujoco_evaluator():
    config = GiaConfig()

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")
    model = GiaModel(config)

    evaluator = MujocoEvaluator(
        args,
        task_map={"mujoco": ["InvertedDoublePendulum-v4"]},
        dataset_map={"mujoco": ["mujoco-doublependulum"]},
    )
    evaluator.evaluate(model)
