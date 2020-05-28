import numpy as np
from models import transformer
from torch import nn
from models import gru
from models import ff
from models import linear
# from models import lstm


def create_model(
        batch_first,
        bptt,
        model_type,
        model_cfg,
        device,
        input_size,
        output_size,
        hidden_size,
        default_model_params):
    """Wrapper for creating models with default hyperparameters."""
    # Consider putting these in a immutable-across-experiments class
    for idx, (k, v) in enumerate(default_model_params.items()):
        v['input_size'] = input_size
        v['output_size'] = output_size
        v['hidden_size'] = hidden_size
        v['batch_first'] = batch_first
        v['bptt'] = bptt
        default_model_params[k] = v
    dct = default_model_params[model_cfg]
    if model_type.lower() == "transformer":
        dct = default_model_params[model_cfg]
        model = transformer.TransformerModel(**dct)
    elif model_type.lower() == 'lstm':
        model = lstm.LSTM(**dct)
    elif model_type.lower() == 'gru':
        model = gru.GRU(**dct)
    elif model_type.lower() == "ff":
        model = ff.FF(**dct)
    elif model_type.lower() == "linear":
        model = linear.linear(**dct)
    else:
        raise NotImplementedError(model_type)
    return model.to(device)

