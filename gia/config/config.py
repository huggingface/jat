""" GIA Config model configuration"""

import copy

from transformers import GPTNeoConfig


GIA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gia-project/gia": "https://huggingface.co/gia-project/gia/resolve/main/config.json",
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
        seq_len (`int`, *optional*, defaults to 2048):
            The length (number of tokens) of a sequence.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to extract from image observations.
        image_vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size for the patch position encodings.
        num_groups (`int`, *optional*, defaults to 32):
            Number of groups for the Residual Attention Block.
        num_res_channels (`int`, *optional*, defaults to 256):
            Number of residual channels for the Residual Attention Block.
        text_vocab_size (`int`, *optional*, defaults to 30_000):
            Vocabulary size for the text tokens.
        nb_bins (`int`, *optional*, defaults to 1024):
            Number of bins for the discretization of continuous values.
        max_local_position (`int`, *optional*, defaults to 512):
            Maximum number of local positions to encode.
        use_separator (`bool`, *optional*, defaults to True):
            Whether to include a separator token between observations and actions.
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
        seq_len: int = 2048,
        patch_size: int = 16,
        image_vocab_size: int = 128,
        num_groups: int = 32,
        num_res_channels: int = 256,
        text_vocab_size: int = 30_000,
        nb_bins: int = 1024,
        max_local_position: int = 512,
        use_separator: bool = True,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size=None,
        activation_function: str = "gelu_new",
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        attention_dropout: float = 0,
        layer_norm_epsilon: float = 0.00001,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.image_vocab_size = image_vocab_size
        self.num_groups = num_groups
        self.num_res_channels = num_res_channels
        self.text_vocab_size = text_vocab_size
        self.nb_bins = nb_bins
        self.max_local_position = max_local_position
        self.use_separator = use_separator

        if self.use_separator:
            vocab_size = self.text_vocab_size + self.nb_bins + 1
        else:
            vocab_size = self.text_vocab_size + self.nb_bins

        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,  # renamed from max_position_embeddings
            hidden_size=hidden_size,
            num_layers=num_layers,
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

    @property
    def vocab_size(self):
        if self.use_separator:
            return self.text_vocab_size + self.nb_bins + 1
        else:
            return self.text_vocab_size + self.nb_bins

    @vocab_size.setter
    def vocab_size(self, value):
        if value != self.vocab_size:
            raise ValueError(
                "vocab_size is a derived attribute that is the sum of text_vocab_size and nb_bins, plus one if using "
                "a separator token. It cannot be set directly. Please set text_vocab_size, nb_bins, and use_separator "
                "instead."
            )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        return output
