import pytest
import torch
import torch.nn.functional as F

from gia.model.activation import GEGLU


def test_forward():
    # Create a GEGLU module
    geglu = GEGLU()

    # Create a tensor with shape (2, 4)
    input_tensor = torch.randn(2, 4)

    # Apply the GEGLU module to the input
    output = geglu(input_tensor)

    # Split the input tensor into two chunks
    a, b = input_tensor.chunk(2, dim=-1)

    # Compute the expected output manually
    expected_output = a * F.gelu(b)

    # Assert that the output from GEGLU matches the expected output
    assert torch.allclose(output, expected_output), "GEGLU output does not match expected output"


def test_input_shape():
    # Create a GEGLU module
    geglu = GEGLU()

    # Create a tensor with shape (2, 3) - an odd number of columns
    input_tensor = torch.randn(2, 3)

    # Assert that applying the GEGLU module to this tensor raises a ValueError
    with pytest.raises(ValueError):
        geglu(input_tensor)
