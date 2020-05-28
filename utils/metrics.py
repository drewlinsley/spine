import torch


def pearsonr_old(
        x,
        y,
        REDUCE='mean',
        batch_first=True,
        eps=1e-8):
    r"""Computes Pearson Correlation Coefficient across rows.
    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:
    .. math::
        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}
    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.
    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = 1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std + eps)

    if REDUCE == 'mean':
        corr = 1 - corr.mean()
        return corr
    else:
        raise NotImplementedError(REDUCE)


def pearsonr(x, y, batch_first=False, eps=1e-8, REDUCE="mean"):
    """My implementation of pearson."""
    if batch_first:
        dim = 1
    else:
        dim = 0
    vx = x - x.mean(dim, keepdim=True)
    vy = y - y.mean(dim, keepdim=True)
    numerator = (vx * vy).sum(dim, keepdim=True)
    ssx = torch.sqrt(((vx + eps) ** 2).sum(dim=dim, keepdim=True))
    ssy = torch.sqrt(((vy + eps) ** 2).sum(dim=dim, keepdim=True))
    denominator = ssx * ssy
    loss = 1 - (numerator / denominator)
    if REDUCE == "mean":
        loss = loss.mean()
        return loss
    elif REDUCE == "max":
        return loss.max(-1)[0].mean()
    else:
        raise NotImplementedError(REDUCE)


def bce_pearson(x, y):
    """Pearson after thresholding x."""
    # return pearsonr((torch.sigmoid(x)), y)
    return pearsonr((torch.sigmoid(x) > 0.5).float(), y)


def get_metric(metric, batch_first):
    """Wrapper for returning a function."""
    score = pearsonr
    if metric.lower() == 'pearson':
        metric = pearsonr
    elif metric.lower() == "pearson_max":
        metric = lambda x, y: pearsonr(x, y, REDUCE="max")
    elif metric.lower() == 'l2':
        metric = lambda x, y: torch.norm(x - y, 2)
    elif metric.lower() == 'l2_pearson':
        metric = lambda x, y: (pearsonr(x, y, batch_first) + 1e-4 * torch.norm(x - y, 2))
    elif metric.lower() == 'l1_pearson':
        metric = lambda x, y: (pearsonr(x, y, batch_first) + 1e-2 * torch.norm(x - y, 1))
    elif metric.lower() == 'bce':
        metric = torch.nn.BCEWithLogitsLoss  # (pos_weight=torch.Tensor(10))
        score = lambda x, y: bce_pearson(x, y, batch_first)
    else:
        return NotImplementedError('Metric {} not implemented'.format(metric))
    return score, metric

