import pytest
from transformers import AutoConfig, AutoModel

import gato  # noqa
from gato import GatoConfig, GatoModel
from gato.processing import GatoProcessor


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_config(model_size):
    AutoConfig.from_pretrained(f"gia-project/gato-{model_size}")


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_auto_model(model_size):
    config = AutoConfig.from_pretrained(f"gia-project/gato-{model_size}")
    AutoModel.from_config(config)


@pytest.mark.parametrize("model_size", ["80m", "387m", "1.27b"])
def test_load_model_and_processor(model_size):
    config = GatoConfig.from_pretrained(f"gia-project/gato-{model_size}")
    GatoProcessor.from_pretrained(f"gia-project/gato-{model_size}")
    GatoModel(config)
