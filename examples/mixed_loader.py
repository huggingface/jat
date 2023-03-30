from gia.datasets import get_dataloader
dataloader = get_dataloader(["babyai-go-to", "mujoco-ant"], shuffle=True, batch_size=2)
iterator = iter(dataloader)

for _ in range(4):
    print(next(iterator).keys())
