from models import transformer
from torch import nn


def create_model(
        model_type,
        ntoken=255,
        ninp=3,
        nhead=2,
        nhid=2,
        nlayers=1,
        dropout=0.5):
    """Wrapper for creating models with default hyperparameters."""
    if model_type.lower() == 'transformer':
        model = transformer.TransformerModel(
            ntoken=ntoken,
            ninp=ninp,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout)
    elif model_type.lower() == 'lstm':
        model = nn.LSTM(ninp, nhid)
    elif model_type.lower() == 'gru':
        model = nn.GRU(ninp, nhid, batch_first=True)
    else:
        raise NotImplementedError(model_type)
    return model
