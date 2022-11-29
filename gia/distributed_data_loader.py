import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote, rpc_async, rpc_sync

from gia.environment_worker import WORKER_NAME, DistributedEnvironmentWorker, EnvInfo, EnvironmentWorker
from gia.utils.utils import _call_remote_method
from gia.config.config import Config


class DistributedDataLoader:
    def __init__(
        self,
        config: Config,
        env_worker_ranks,
        model_server,
    ) -> None:
        assert config.hyp.mini_batch_size <= config.hyp.n_agents
        assert config.hyp.n_agents % config.hyp.mini_batch_size == 0

        self.config = config
        self.iters_per_epoch = config.hyp.n_agents // config.hyp.mini_batch_size

        self.worker_rrefs = []
        self.futures = []

        for worker_rank in env_worker_ranks:
            worker_info = rpc.get_worker_info(WORKER_NAME.format(worker_rank))
            self.worker_rrefs.append(remote(worker_info, DistributedEnvironmentWorker, (RRef(model_server),)))

        self.env_info: EnvInfo = rpc_sync(
            self.worker_rrefs[0].owner(),
            _call_remote_method,
            args=(DistributedEnvironmentWorker.get_env_info, self.worker_rrefs[0]),
        )
        print(f"got env info {self.env_info=}")
        self.launch_remote_sample_trajectory()

    def launch_remote_sample_trajectory(self):
        for worker_rref in self.worker_rrefs:
            future = rpc_async(
                worker_rref.owner(),
                _call_remote_method,
                args=(DistributedEnvironmentWorker.sample_trajectory, worker_rref),
            )

            self.futures.append(future)

    def __iter__(self):
        for epoch in range(self.config.hyp.n_epochs):
            for j in range(self.iters_per_epoch):
                waiting_for_exp = True
                buffer = None
                while waiting_for_exp:
                    for i in range(len(self.futures)):
                        if self.futures[i].done():
                            self.futures[i].wait()
                            buffer = self.futures[i].value()
                            self.futures[i] = rpc_async(
                                self.worker_rrefs[i].owner(),
                                _call_remote_method,
                                args=(DistributedEnvironmentWorker.sample_trajectory, self.worker_rrefs[i]),
                            )
                            waiting_for_exp = False
                            break

                yield buffer

    def __len__(self):
        return 8
