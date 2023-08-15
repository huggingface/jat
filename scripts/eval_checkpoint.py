from gia import GiaConfig, GiaModel
from gia.eval.rl import make
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor


config = GiaConfig(num_layers=4, num_heads=12, hidden_size=384)
model = GiaModel(config)


task_name = "metaworld-assembly"
env = make(task_name, num_envs=1)
processor = GiaProcessor()
gia_agent = GiaAgent(model, processor, task_name, use_prompt=False)

gia_agent.reset()
obs, info = env.reset()
action = gia_agent.get_action(obs)
print(action)
