import torch
from torch import Tensor

from gia.utils.constants import SEP
from gia.utils.utils import discretize, mu_law


def mujoco_tokenization_fn(
    observations: Tensor,
    actions: Tensor,
    mu: float = 100,
    M: float = 256,
    nb_bins: int = 1024,
    token_shift: int = 32000,
):
    """
    Tokenize episodes from MuJoCo environment.

    First, each floating point element of tensors in the observation sequence is mu-law companded as in WaveNet (Oord et al., 2016).
    Then, the tensors are discretized into integers in the range [0, nb_bins-1].
    Finally, the tensors are concatenated and shifted by token_shift. Tensors and actions are separated by SEP.

    Args:
        observations (Tensor): Observations
        actions (Tensor): Actions
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
        token_shift (int, optional): Shift tokens by this value. Defaults to 32000.

    Returns:
        Tensor: Tokenized episodes
    """
    num_timesteps = observations.shape[0]

    # Each floating point element of tensors in the observation sequence is mu-law companded as in WaveNet (Oord et al., 2016):
    normalized_tensor_tokens = mu_law(observations, mu=mu, M=M)

    # No need to normalize actions, since they are already in the range [-1, 1]
    normalized_action_tokens = actions

    # Discretize tensors
    discretized_tensor_tokens = discretize(normalized_tensor_tokens, nb_bins=nb_bins)
    discretized_action_tokens = discretize(normalized_action_tokens, nb_bins=nb_bins)

    # Concatenate
    separators = torch.ones(num_timesteps, 1, dtype=torch.int64) * SEP
    episode_tokens = torch.cat((discretized_tensor_tokens, separators, discretized_action_tokens), dim=1)

    # Shift tokens
    episode_tokens += token_shift
    return episode_tokens
