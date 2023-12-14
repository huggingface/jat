from transformers import GPTNeoConfig


class JatConfig(GPTNeoConfig):
    r"""
    This is the configuration class to store the configuration of a [`JatModel`]. It is used to instantiate a Jat
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ... (TODO)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT Neo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoModel`]. Vocabulary size of the model. Defines the different
            tokens that can be represented by the *inputs_ids* passed to the forward method of [`GPTNeoModel`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        attention_types (`List`, *optional*, defaults to `[[["global", "local"], 12]]`):
            The type of attention for each layer in a `List` of the following format `[[["attention_type"],
            num_layerss]]` e.g. for a 24 layer model `[[["global"], 24]]` or `[[["global", "local"], 12]]` Choose the
            value of `attention_type` from `["global", "local"]`
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        window_size (`int`, *optional*, defaults to 256):
            The size of the sliding window for local attention.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Residual dropout used in the attention pattern.
        embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model [`GPTNeoForTokenClassification`]. The
            dropout ratio for the hidden layer.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 50256):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            The id of the end of sentence token in the vocabulary.
        max_continuous_size (`int`, *optional*, default to 376):
            The maximum size of the continuous values.
        max_discrete_value (`int`, *optional*, default to 18):
            The maximum value of the discrete values.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        observation_loss_coef (`float`, *optional*, defaults to 0.005):
            The coefficient for the observation loss. When set to 0.0, the observation is not even predicted.
        action_loss_coef (`float`, *optional*, defaults to 0.995):
            The coefficient for the action loss.
    """

    model_type = "jat"

    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        max_continuous_size=377,
        max_discrete_value=18,
        image_size=224,
        num_channels=3,
        patch_size=16,
        observation_loss_coef=0.005,
        action_loss_coef=0.995,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            max_position_embeddings,
            hidden_size,
            num_layers,
            attention_types,
            num_heads,
            intermediate_size,
            window_size,
            activation_function,
            resid_dropout,
            embed_dropout,
            attention_dropout,
            classifier_dropout,
            layer_norm_epsilon,
            initializer_range,
            use_cache,
            bos_token_id,
            eos_token_id,
            **kwargs,
        )
        self.max_continuous_size = max_continuous_size
        self.max_discrete_value = max_discrete_value
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.observation_loss_coef = observation_loss_coef
        self.action_loss_coef = action_loss_coef


JatConfig.register_for_auto_class()
