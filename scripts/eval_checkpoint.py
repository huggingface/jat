from gato import GatoConfig, GatoModel
from gato.eval.rl import make
from gato.eval.rl.gato_agent import GatoAgent
from gato.processing import GatoProcessor


config = GatoConfig(num_layers=4, num_heads=12, hidden_size=384)
model = GatoModel(config)


task_name = "metaworld-assembly"
env = make(task_name, num_envs=1)
processor = GatoProcessor()
gia_agent = GatoAgent(model, processor, task_name, use_prompt=False)

gia_agent.reset()
obs, info = env.reset()
action = gia_agent.get_action(obs)
print(action)
