import torch


def get_optimizer(optimizer):
    """Wrapper for optimizers."""
    if optimizer == "AdamW":
        return torch.optim.AdamW
    elif optimizer == "Adam":
        return torch.optim.Adam
    elif optimizer == "SGD":
        return torch.optim.SGD
    else:
        raise NotImplementedError(optimizer)


def get_scheduler(schedule):
    """Wrapper for schedulers."""
    if schedule == "StepLR":
        return lambda x: torch.optim.lr_scheduler.StepLR(x, 1.0, gamma=0.95)
    else:
        return None

