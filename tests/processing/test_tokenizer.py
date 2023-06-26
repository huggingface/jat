import numpy as np
import pytest
from PIL import Image

from gia.processing.tokenizer import GiaTokenizer


def test_decode_continuous():
    # Instantiate GiaProcessor
    tokenizer = GiaTokenizer(nb_bins=1024)

    # Create sample tokens
    x = np.linspace(-5.0, 5.0, 50).reshape(1, 25, -1).tolist()  # Convert to batch of size 1, 25 timesteps, 2 features

    # Call tokenize_continuous method
    tokens = tokenizer(continuous_observations=x)

    # Compute the expected result
    expected_result = tokenizer.decode_continuous(tokens["continuous_observations"]["input_ids"])

    # Assert that the result is close to the expected result
    np.testing.assert_allclose(x, expected_result, atol=1e-1)


@pytest.mark.parametrize("input_type", ["actions", "observations"])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("nb_bins", [8, 1024])
def test_tokenize_continuous_observations(input_type, shape, nb_bins):
    tokenizer = GiaTokenizer(nb_bins=nb_bins)
    key = f"continuous_{input_type}"
    x = {key: np.random.random(shape).tolist()}
    features = tokenizer(**x)
    tokens = np.array(features[key]["input_ids"])
    # Check the shape
    assert tokens.shape == shape
    # Check that the dtype is correct
    assert tokens.dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(tokens < nb_bins + tokenizer.token_shift)

    # Check the input_types
    input_types = np.array(features[key]["input_types"])
    assert input_types.shape == shape
    assert input_types.dtype == np.int64
    assert np.all(input_types == 0)


@pytest.mark.parametrize("input_type", ["actions", "observations"])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_tokenize_discrete_observations(input_type, shape):
    tokenizer = GiaTokenizer()
    key = f"discrete_{input_type}"
    x = {key: np.random.randint(0, tokenizer.nb_bins, shape).tolist()}
    features = tokenizer(**x)
    tokens = np.array(features[key]["input_ids"])
    # Check the shape
    assert tokens.shape == shape
    # Check that the dtype is correct
    assert tokens.dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(tokens <= tokenizer.nb_bins + tokenizer.token_shift)

    # Check the input_types
    input_types = np.array(features[key]["input_types"])
    assert input_types.shape == shape
    assert input_types.dtype == np.int64
    assert np.all(input_types == 0)


@pytest.mark.parametrize("modality", ["text_observations", "text"])  # actually work the same way
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_tokenize_text(modality, shape):
    tokenizer = GiaTokenizer()
    # Generate random text
    dictionary = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
    data = {modality: np.random.choice(dictionary, shape).tolist()}
    features = tokenizer(**data)
    tokens = features[modality]["input_ids"]
    input_types = features[modality]["input_types"]

    # Check the shape and tokens are in the correct range (need recusion because of the nested lists)
    def check_tokens(tokens, shape):
        if len(shape) == 0:
            # don't check the last dim, as it can vary
            assert isinstance(tokens, list)
            assert all(isinstance(token, int) for token in tokens)
            assert all(token >= 0 for token in tokens)
            assert all(token <= tokenizer.token_shift for token in tokens)
        else:
            assert isinstance(tokens, list)
            assert len(tokens) == shape[0]
            for token in tokens:
                check_tokens(token, shape[1:])

    check_tokens(tokens, shape)

    # Check the shape and input_types==0 (need recusion because of the nested lists)
    def check_input_types(input_types, shape):
        if len(shape) == 0:
            # don't check the last dim, as it can vary
            assert isinstance(input_types, list)
            assert all(isinstance(input_type, int) for input_type in input_types)
            assert all(input_type == 0 for input_type in input_types)
        else:
            assert isinstance(input_types, list)
            assert len(input_types) == shape[0]
            for input_type in input_types:
                check_input_types(input_type, shape[1:])

    check_input_types(input_types, shape)


@pytest.mark.parametrize("modality", ["image_observations", "images"])  # actually work the same way
@pytest.mark.parametrize("data_type", ["array", "pillow"])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_tokenize_image(modality, data_type, shape):
    tokenizer = GiaTokenizer()

    def generate_val(shape):
        # Base case: if there's no more dimensions to generate, create the numpy array
        if not shape:
            val = np.random.randint(0, 255, (32, 32, 2), dtype=np.uint8)
            if data_type == "pillow":
                val = Image.fromarray(val)
            return val
        # Recursive case: generate a list with the number of elements specified by the first dimension of the shape
        return [generate_val(shape[1:]) for _ in range(shape[0])]

    data = {modality: generate_val(shape)}
    features = tokenizer(**data)
    patches = np.array(features[modality]["patches"])
    input_types = np.array(features[modality]["input_types"])

    assert patches.shape[-3:] == (4, 16, 16)
    # patches.shape[-4] is the the number of patches per image, which can vary depending on the image size
    assert patches.shape[:-4] == shape
    assert patches.dtype == np.uint8
    assert input_types.shape[:-1] == shape  # input_types.shape[-1] is the number of patches per image
    assert input_types.dtype == np.int64
    assert np.all(input_types == 1)


def test_extract_patches():
    tokenizer = GiaTokenizer(patch_size=3)

    arr = np.arange(9 * 6 * 3, dtype=np.uint8).reshape(9, 6, 3)  # H, W, C
    image = Image.fromarray(arr)
    patches = tokenizer.extract_patches(image)[0]
    arr = arr.transpose(2, 0, 1)  # C, H, W

    np.testing.assert_equal(patches[0][:3], arr[:, 0:3, 0:3])
    np.testing.assert_equal(patches[0][:3], arr[:, 0:3, 0:3])
    np.testing.assert_equal(patches[1][:3], arr[:, 0:3, 3:6])
    np.testing.assert_equal(patches[2][:3], arr[:, 3:6, 0:3])
    np.testing.assert_equal(patches[3][:3], arr[:, 3:6, 3:6])
    np.testing.assert_equal(patches[4][:3], arr[:, 6:9, 0:3])
    np.testing.assert_equal(patches[5][:3], arr[:, 6:9, 3:6])
    for patch in patches:  # pad to have 4 channels
        np.testing.assert_equal(patch[3], np.zeros((3, 3), dtype=np.uint8))
