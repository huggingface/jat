from gato import GatoConfig, GatoModel
from gato.config.arguments import Arguments
from gato.eval.rl.rl_evaluator import RLEvaluator


def test_mujoco_evaluator():
    config = GatoConfig(num_heads=24, num_layers=2, hidden_size=384, intermediate_size=768)
    model = GatoModel(config)

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")

    evaluator = RLEvaluator(args, "mujoco-doublependulum")
    evaluator.evaluate(model)
