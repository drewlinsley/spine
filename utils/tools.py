import os
import yaml
from glob import glob


def get_model_defaults():
    """Load in all of the model yamls and package into a dict."""
    defaults = {}
    yamls = glob(os.path.join("cfgs", "*.yaml"))
    for fl in yamls:
        with open(fl) as fo:
            hps = yaml.load(fo, Loader=yaml.FullLoader)
        defaults[fl.split(os.path.sep)[-1].split('.')[0]] = hps
    return defaults

