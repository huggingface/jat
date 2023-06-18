""" GIA Config model configuration"""

import copy
from typing import Optional

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from gia.config.arguments import Arguments

from ..utils import logger


GIA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gia-project/gia": "https://huggingface.co/gia-project/gia/resolve/main/config.json",
}


class GiaModelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GiaModel`]. It is used to instantiate an
    GIA model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GIA
    [gia-project/gia](https://huggingface.co/gia-project/gia) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        causal_lm_name (`str`, *optional*, defaults to "EleutherAI/gpt-neo-125M"):
            The name of the causal language model to use. This can be a model identifier or the path to a directory
            containing a configuration file saved using the [`~PreTrainedModel.save_pretrained`] method.
        causal_lm_config (`PretrainedConfig`, *optional*):
            The configuration of the causal language model to use. If not provided, the configuration of the model
            specified by `causal_lm_name` will be used. Note that this configuration will be modified to match the
            Gia model architecture.
        embed_dim (`int`, *optional*, defaults to None):
            The embedding dimension of the Gia model. If not provided, the embedding dimension of the causal language
            model will be used.
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
        seq_len (`int`, *optional*, defaults to None):
            The length (number of tokens) of a sequence. If not provided, the maximum sequence length of the causal
            language model will be used.
        use_pretrained (`bool`, *optional*, defaults to False):
            Whether to use the pretrained weights of the causal language model.

    Example:

    ```python
    >>> from gia import GiaModel, GiaConfig
    >>> config = GiaConfig()
    >>> model = GiaModel(config)
    ```
    """
    model_type = "gia"
    is_composition = True

    def __init__(
        self,
        causal_lm_name: str = "EleutherAI/gpt-neo-125M",
        causal_lm_config: PretrainedConfig = None,
        embed_dim: Optional[int] = None,
        patch_size: int = 16,
        image_vocab_size: int = 128,
        num_groups: int = 32,
        num_res_channels: int = 256,
        text_vocab_size: int = 30_000,
        nb_bins: int = 1024,
        max_local_position: int = 512,
        use_separator: bool = True,
        seq_len: Optional[int] = None,
        use_pretrained: bool = False,
        **kwargs,
    ):
        self.causal_lm_name = causal_lm_name
        if causal_lm_config is None:
            logger.info(f"causal_lm_config is None. Initializing {self.causal_lm_name} with default values.")
            self.causal_lm_config = AutoConfig.from_pretrained(causal_lm_name)
        else:
            self.causal_lm_config = AutoConfig.from_pretrained(causal_lm_name, **causal_lm_config)

        self.embed_dim = embed_dim if embed_dim is not None else self.causal_lm_config.hidden_size
        self.patch_size = patch_size
        self.image_vocab_size = image_vocab_size
        self.num_groups = num_groups
        self.num_res_channels = num_res_channels
        self._text_vocab_size = text_vocab_size
        self._nb_bins = nb_bins
        self.max_local_position = max_local_position
        self._use_separator = use_separator
        self.seq_len = seq_len if seq_len is not None else self.causal_lm_config.max_position_embeddings
        self.use_pretrained = use_pretrained

        # update vocab_size
        self.causal_lm_config.vocab_size = self.vocab_size

        super().__init__(**kwargs)

    @property
    def embed_dim(self):
        return self.causal_lm_config.hidden_size

    @embed_dim.setter
    def embed_dim(self, value):
        if self.causal_lm_config.hidden_size != value:
            logger.info("Setting embed_dim also sets causal_lm_config.hidden_size to the same value.")
            self.causal_lm_config.hidden_size = value

    @property
    def seq_len(self):
        return self.causal_lm_config.max_position_embeddings

    @seq_len.setter
    def seq_len(self, value):
        if self.causal_lm_config.max_position_embeddings != value:
            logger.info("Setting seq_len also sets causal_lm_config.max_position_embeddings to the same value.")
            self.causal_lm_config.max_position_embeddings = value

    @property
    def vocab_size(self):
        if self.use_separator:
            return self.text_vocab_size + self.nb_bins + 1
        else:
            return self.text_vocab_size + self.nb_bins

    @vocab_size.setter
    def vocab_size(self, value):
        raise AttributeError(
            "vocab_size is a derived attribute that is the sum of text_vocab_size and nb_bins, plus one if using a "
            "separator token. It cannot be set directly. Please set text_vocab_size, nb_bins, and use_separator "
            "instead."
        )

    @property
    def text_vocab_size(self):
        return self._text_vocab_size

    @text_vocab_size.setter
    def text_vocab_size(self, value):
        if self._text_vocab_size != value:
            logger.info(
                "Updating the text_vocab_size will also update the vocab_size of the causal_lm_config. "
                "The vocab_size is recalculated based on the following formula: "
                "vocab_size = text_vocab_size + nb_bins + 1 if use_separator is True, otherwise 0."
            )
            self._text_vocab_size = value
            self.causal_lm_config.vocab_size = self.vocab_size

    @property
    def nb_bins(self):
        return self._nb_bins

    @nb_bins.setter
    def nb_bins(self, value):
        if self._nb_bins != value:
            logger.info(
                "Updating the nb_bins will also update the vocab_size of the causal_lm_config. "
                "The vocab_size is recalculated based on the following formula: "
                "vocab_size = text_vocab_size + nb_bins + 1 if use_separator is True, otherwise 0."
            )
            self._nb_bins = value
            self.causal_lm_config.vocab_size = self.vocab_size

    @property
    def use_separator(self):
        return self._use_separator

    @use_separator.setter
    def use_separator(self, value):
        if self._use_separator != value:
            logger.info(
                "Updating use_separator will also update the vocab_size of the causal_lm_config. "
                "The vocab_size is recalculated based on the following formula: "
                "vocab_size = text_vocab_size + nb_bins + 1 if use_separator is True, otherwise 0."
            )
            self._use_separator = value
            self.causal_lm_config.vocab_size = self.vocab_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["causal_lm_config"] = self.causal_lm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

    @staticmethod
    def from_args(args: Arguments) -> "GiaModelConfig":
        config = GiaModelConfig.from_pretrained("gia-project/gia")
        config.patch_size = args.patch_size
        config.nb_bins = args.nb_bins
        config.use_separator = args.use_separator
        # TODO add other args
        return config
