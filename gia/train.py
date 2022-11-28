import torch
from torch import nn

from gia.replay_buffer import ReplayBuffer
from gia.utils.mocks.mock_distributed_data_loader import MockDistributedDataLoader

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
        print("starting epoch")
        for minibatch in ddl:
            #minibatch = minibatch.to(gpu_id)
            print(minibatch["obs"]["vector"].shape)
            preds = model(minibatch["obs"]["vector"])
            loss = loss_function(preds, minibatch)
            loss.backward()

            opt.step()
            print("model update")
            n_updates += 1



if __name__ == "__main__":
    train()