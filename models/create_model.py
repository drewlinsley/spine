from models import transformer


def create_model(
        model_type,
        ntoken=255,
        ninp=2,
        nhead=2,
        nhid=2,
        nlayers=1,
        dropout=0.5):
    """Wrapper for creating models with default hyperparameters."""
    if model_type == 'transformer':
        model = transformer.TransformerModel(
            ntoken=ntoken,
            ninp=ninp,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout)
    elif model_type == 'lstm':
        model = nn.LSTM(ninp, nhid)
    elif model_type == 'gru':
        model = nn.GRU(ninp, nhid)
    else:
        raise NotImplementedError(model_type)
    return model
