from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from gia.config import Arguments


class PatchedFeatureExtractor(nn.Module):
    def __init__(self, args):
        # create the model etc
        pass

    def forward(self, x):
        # TODO: select image patches from batch, with batch[image_patches] ?
        # extracted embeddings add to batch
        return x


class GiaModel(nn.Module):
    def __init__(self, args: Arguments):
        model_config = AutoConfig.from_pretrained(args.model_name)

        self.decoder = AutoModelForCausalLM.from_config(model_config)
        self.image_patch_model = PatchedFeatureExtractor(args)

    def forward(self, batch):
        batch = self.image_patch_model(batch)
