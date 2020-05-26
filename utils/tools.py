import os
import yaml


def get_model_defaults():
    """Load in all of the model yamls and package into a dict."""
    defaults = {}

    # GRU
    with open(os.path.join("cfgs", "gru.yaml")) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)
    defaults["gru"] = hps

    # LSTM
    with open(os.path.join("cfgs", "lstm.yaml")) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)
    defaults["lstm"] = hps

    # Transformer
    with open(os.path.join("cfgs", "transformer.yaml")) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)
    defaults["transformer"] = hps
    return defaults
