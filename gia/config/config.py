""" GIA Config model configuration"""

import copy
from typing import Optional

from transformers import AutoConfig, GPTNeoConfig

from gia.config.arguments import Arguments


GIA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gia-project/gia-80m": "https://huggingface.co/gia-project/gia-80m/blob/main/config.json",
    "gia-project/gia-387m": "https://huggingface.co/gia-project/gia-387m/blob/main/config.json",
    "gia-project/gia-1.27b": "https://huggingface.co/gia-project/gia-1.27b/blob/main/config.json",
}


class GiaConfig(GPTNeoConfig):
    r"""
    This is the configuration class to store the configuration of a [`GiaModel`]. It is used to instantiate an
    GIA model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GIA
    [gia-project/gia](https://huggingface.co/gia-project/gia) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30_021):
            Vocabulary size of the GPT Neo model. Defines the number of different tokens that can be represented by
            the [`inputs_ids`] passed when calling [`GiaModel`].
        seq_len (`int`, *optional*, defaults to 2048):
            The length (number of tokens) of a sequence.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to extract from image observations.
        image_vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size for the patch position encodings.
        num_groups (`int`, *optional*, defaults to 32):
            Number of groups for the Residual Attention Block.
        num_res_channels (`int`, *optional*, defaults to 64):
            Number of residual channels for the Residual Attention Block.
        max_local_position (`int`, *optional*, defaults to 512):
            Maximum number of local positions to encode.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to None):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder. If None,
            `intermediate_size = 4 * hidden_size`.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Residual dropout used in the attention pattern.
        embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
    """
    model_type = "gia"

    def __init__(
        self,
        vocab_size: int = 31_025,
        seq_len: int = 2048,
        patch_size: int = 16,
        image_vocab_size: int = 128,
        num_groups: int = 32,
        num_res_channels: int = 64,
        max_local_position: int = 512,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_epsilon: float = 0.00001,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.image_vocab_size = image_vocab_size
        self.num_groups = num_groups
        self.num_res_channels = num_res_channels
        self.max_local_position = max_local_position

        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,  # renamed from max_position_embeddings
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_types=[[["global"], num_layers]],
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            window_size=seq_len,  # we want to attend to all tokens
            activation_function=activation_function,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            attention_dropout=attention_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            use_cache=use_cache,
            **kwargs,
        )

    @property
    def seq_len(self):
        return self.max_position_embeddings

    @seq_len.setter
    def seq_len(self, value):
        self.max_position_embeddings = value

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output.pop("attention_types")
        output.pop("max_position_embeddings")
        output.pop("window_size")
        output["model_type"] = self.__class__.model_type
        return output

    @staticmethod
    def from_args(args: Arguments) -> "GiaConfig":
        config = GiaConfig.from_pretrained("gia-project/gia")
        config.patch_size = args.patch_size
        # TODO add other args
        return config


AutoConfig.register("gia", GiaConfig)
