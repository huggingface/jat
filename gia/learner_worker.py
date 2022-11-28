from typing import List

import torch
import torch.distributed.rpc as rpc

from gia.distributed_data_loader import DistributedDataLoader

LEARNER_NAME = "LEARNER_{}"


class LearnerWorker:
    def __init__(self, rank: int, world_size: int, worker_ranks: List[int], data_loader=DistributedDataLoader) -> None:
        rpc.init_rpc(LEARNER_NAME.format(rank), rank=rank, world_size=world_size)
        print("starting learner")
        # create the distributed data loader remote workers
        self.data_loader = data_loader(worker_ranks, self)

        # build the learner model,
        # sync weights to env worker
        # add ref to env worker for model weight callback

    def train(self):
        for minibatch in self.data_loader:
            print(minibatch.buffer["rewards"].shape)

    def get_latest_model_weights(self):
        return {"checkpoint": torch.randn(4, 1)}
