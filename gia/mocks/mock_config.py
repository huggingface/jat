class MockConfig:
    rollout_length = 16
    n_parallel_agents = 1
    encoder_mlp_layers = [512, 512, 512]
    nonlinearity = "relu"
    encoder_conv_architecture = "convnet_simple"
    encoder_conv_mlp_layers = [512]
