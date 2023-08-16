import pytest
from transformers import AutoConfig, AutoModel

import gia  # noqa
from gia import GiaConfig, GiaModel
from gia.processing import GiaProcessor


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_config(model_size):
    AutoConfig.from_pretrained(f"gia-project/gia-{model_size}")


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_model(model_size):
    config = AutoConfig.from_pretrained(f"gia-project/gia-{model_size}")
    AutoModel.from_config(config)


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_load_model_and_processor(model_size):
    config = GiaConfig.from_pretrained(f"gia-project/gia-{model_size}")
    GiaProcessor.from_pretrained(f"gia-project/gia-{model_size}")
    GiaModel(config)
