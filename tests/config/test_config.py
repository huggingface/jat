import pytest
from transformers import AutoConfig, AutoModel
import gia  # noqa


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_config(model_size):
    AutoConfig.from_pretrained(f"gia-project/gia-{model_size}")


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_model(model_size):
    config = AutoConfig.from_pretrained(f"gia-project/gia-{model_size}")
    AutoModel.from_config(config)
