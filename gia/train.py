import os

import torch
import torch.distributed.rpc as rpc
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.rpc import RRef

import hydra

from gia.environment_worker import WORKER_NAME
from gia.learner_worker import LearnerWorker
from gia.config.config import Config


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def train(config: Config):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"launching distributed training {rank=} {local_rank=} {world_size=}")
    world_size = 4
    init_process_group(backend="nccl")
    if rank == 0:
        learner = LearnerWorker(config, rank=0, world_size=4, worker_ranks=[1, 2, 3])
        learner.train()
        destroy_process_group()
    else:
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size)
    rpc.shutdown()


if __name__ == "__main__":
    train()
