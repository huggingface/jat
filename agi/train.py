import torch
from torch import nn

from agi.replay_buffer import ReplayBuffer

class MockDistributedDataLoader():
    def __init__(self):
        self.minibatch_size
        self.buffer_size
        self.obs_space
        self.action_space


    def __iter__(self):
        for i in range(len(self)):
            mini_batch = {
                "obs" : torch.randn(4,7)
            }
            yield mini_batch
        #raise StopIteration()

    def __len__(self):
        return 8


class Model(nn.Module):
    pass

    def get_weights():
        pass


class Opt:
    pass


def loss_function(preds, minibatch):
    return preds.sum()



def train():

    n_updates = 0
    total_updates = 1_000
    num_epochs = 2
    gpu_id = 0  # local rank

    ddl = MockDistributedDataLoader()
    model = nn.Linear(7,1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    while n_updates < total_updates:

        for epoch in range(num_epochs):
            print("epoch")
            for minibatch in ddl:
                #minibatch = minibatch.to(gpu_id)
                print(minibatch["obs"].shape)
                preds = model(minibatch["obs"])
                loss = loss_function(preds, minibatch)
                loss.backward()

                opt.step()
            print("model update")



if __name__ == "__main__":
    train()