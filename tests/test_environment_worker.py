from gia.environment_worker import EnvironmentWorker


def test_environment_worker():
    env_worker = EnvironmentWorker()
    traj = env_worker.sample_trajectory()
