from torch import nn

from transformers import AutoModelForCausalLM

from gia.config import Arguments


# class PatchedFeatureExtractor(nn.Module):
#     def __init__(self, args):
#         # create the model etc
#         pass

#     def forward(self, x):
#         # TODO: select image patches from batch, with batch[image_patches] ?
#         # extracted embeddings add to batch
#         return x


class GiaModel(nn.Module):
    def __init__(self, args: Arguments):
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    def forward(self, batch):
        # batch = {
        #   "image_patches" {"patches": ... , "positions": ... }
        #   "tokens" : ...
        #
        #
        # }
        pass
        image_embeddings = self.image_patch_model(batch)


if __name__ == "__main__":
    args = Arguments()
    model = GiaModel(args)
