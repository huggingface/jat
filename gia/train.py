import os
import torch
from torch import nn
import torch.distributed.rpc as rpc
from gia.distributed_data_loader import DistributedDataLoader
from gia.environment_worker import WORKER_NAME
from gia.learner_worker import LearnerWorker
from gia.mocks.mock_distributed_data_loader import MockDistributedDataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.rpc import RRef


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"launching distributed training {rank=} {local_rank=} {world_size=}")
    world_size = 4
    init_process_group(backend="nccl")
    if rank == 0:
        learner = LearnerWorker(rank=0, world_size=4, worker_ranks=[1, 2, 3])
        learner.train()
        destroy_process_group()
    else:
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size)
    rpc.shutdown()
