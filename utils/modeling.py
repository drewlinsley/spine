import numpy as np
from models import transformer
from torch import nn
from models import gru
# from models import lstm


def create_model(
        model_type,
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
        v['input_size'] = input_size
        default_model_params[k] = v
    if model_type.lower() == "transformer":
        dct = default_model_params["transformer"]
        model = transformer.TransformerModel(**dct)
    elif model_type.lower() == 'lstm':
        dct = default_model_params["lstm"]
        model = lstm.LSTM(**dct)
    elif model_type.lower() == 'gru':
        dct = default_model_params["gru"]
        model = gru.GRU(**dct)
    else:
        raise NotImplementedError(model_type)
    return model.to(device)

