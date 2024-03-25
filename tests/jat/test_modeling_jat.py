import torch

from jat.modeling_jat import compute_mse_loss


def test_basic():
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, True]])

    loss = compute_mse_loss(predicted, true, mask)
    expected_loss = torch.tensor(0.0)

    assert torch.isclose(loss, expected_loss, atol=1e-8)


def test_masking():
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    true = torch.tensor([[[1.0, 2.0], [10.0, 10.0]]])  # second time step is different
    mask = torch.tensor([[True, False]])  # mask out the second time step

    loss = compute_mse_loss(predicted, true, mask)
    expected_loss = torch.tensor(0.0)  # masked entries should be ignored

    assert torch.isclose(loss, expected_loss, atol=1e-8)


def test_weighted():
    # batch size = 1, time steps = 3, features = 2
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    true = torch.tensor([[[1.0, 2.0], [10.0, 10.0], [5.0, 6.0]]])  # second time step is different
    mask = torch.tensor([[True, True, True]])
    weights = torch.tensor([[1.0, 0.0, 1.0]])  # mask out the second time step

    loss = compute_mse_loss(predicted, true, mask, weights=weights)
    expected_loss = torch.tensor(0.0)  # second time step should be ignored due to zero weight
    print(loss)
    assert torch.isclose(loss, expected_loss, atol=1e-8)
