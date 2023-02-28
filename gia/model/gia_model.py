from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from gia.config.config import Config


class PatchedFeatureExtractor(nn.Module):
    def __init__(self, config: Config):
        # create the model etc
        pass

    def forward(self, x):
        # TODO: select image patches from batch, with batch[image_patches] ?
        # extracted embeddings add to batch
        return x


class GiaModel(nn.Module):
    def __init__(self, config: Config):
        model_config = AutoConfig.from_pretrained("bert-base-uncased")

        self.decoder = AutoModelForCausalLM.from_config(model_config)
        self.image_patch_model = PatchedFeatureExtractor(config)

    def forward(self, batch):
        batch = self.image_patch_model(batch)
