from gia import GiaConfig, GiaModel
from gia.config.arguments import Arguments
from gia.eval.rl.gym_evaluator import GymEvaluator


def test_mujoco_evaluator():
    config = GiaConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GiaModel(config)

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")

    evaluator = GymEvaluator(args, "mujoco-doublependulum")
    evaluator.evaluate(model)
