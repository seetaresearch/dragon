# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from . import *

def BatchNorm(inputs, momentum=0.9, eps=1e-3, use_stats=-1, inplace=False, **kwargs):
    """Batch Normalization, introduced by `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var, factor].
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).
    inplace : boolean
        Whether to share input for the output.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |batchnorm_function|

        The moving average of mean/var, calculated as:

        |moving_average_function|

    Notes
    -----
    This operator follows the implementation of `Caffe`_, without scale after normalization.

    The scale procedure is moved to `ops.Scale(*args, **kwargs)`_.

    """
    CheckInputs(inputs, 4)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='BatchNorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def BatchRenorm(inputs, momentum=0.9, eps=1e-3, r_max=3.0, d_max=5.0,
                t_delta=1.0, use_stats=-1, inplace=False, **kwargs):
    """Batch Renormalization, introduced by `[Ioffe, 2017] <https://arxiv.org/abs/1702.03275>`_

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var, factor].
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    r_max : float
        The higher bound of r.
    d_max : float
        The higher bound of d.
    t_delta : float
        The magnitude of incrementing after each iteration.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).
    inplace : boolean
        Whether to share input for the output.

    Returns
    -------
    Tensor
        The output tensor

        |batchrenorm_function|

        The moving average of mean/var, calculated as:

        |moving_average_function|

    Notes
    -----
    This operator follows the implementation of `Caffe`_, without scale after normalization.

    The scale procedure is moved to `ops.Scale(*args, **kwargs)`_.

    """
    CheckInputs(inputs, 4)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='BatchRenorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def BN(inputs, momentum=0.9, eps=1e-3, use_stats=-1, **kwargs):
    """Batch Normalization, with scale procedure after normalization.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var, scale, bias].
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |batchnorm_scale_function|

        The moving average of mean/var, calculated as:

        |moving_average_function|

    """
    CheckInputs(inputs, 5)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='BN', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def InstanceNorm(inputs, eps=1e-3, inplace=False, **kwargs):
    """Instance Normalization, introduced by `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    eps : float
        The eps.
    inplace : boolean
        Whether to share input for the output.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='InstanceNorm', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def L2Norm(inputs, axis=0, num_axes=-1, eps=1e-5, **kwargs):
    """L2 Normalization, introduced by `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The start axis of stats region.
    num_axes : int
        The number of axes of stats region. Default is ``-1`` (Till End).
    eps : float
        The eps.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='L2Norm', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output