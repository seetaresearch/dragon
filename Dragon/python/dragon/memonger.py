# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

def ShareGrads(enabled=True):
    """Enable gradients sharing globally.

    Parameters
    ----------
    enabled : boolean
        Whether to share grads.

    Returns
    -------
    None

    Examples
    --------
    >>> import dragon.memonger as opt
    >>> opt.ShareGrads()

    """
    from dragon.config import option
    option['share_grads'] = enabled


def Drop(op_func, *args, **kwargs):
    """Drop(Share) the inputs for outputs.

    Parameters
    ----------
    op_func : lambda
        The function of any operators.
    args : list
        The args of this operator.
    kwargs : dict
        The kwargs. The kwargs of this operator.

    Returns
    -------
    Tensor or list of Tensor
        As the ``op_func`` returns.

    Examples
    --------
    >>> from dragon.core.tensor import Tensor
    >>> import dragon.ops as ops
    >>> import dragon.memonger as opt
    >>> data = Tensor().Variable()
    >>> conv_1 = ops.Conv2D(data, num_output=8)
    >>> conv_1_bn = opt.Drop(ops.BatchNorm, [conv_1, Tensor().Variable(), Tensor.Variable()])
    >>> conv_1_relu = opt.Drop(ops.Relu, conv_1_bn)

    """
    kwargs['mirror_stage'] = True
    return op_func(*args, **kwargs)