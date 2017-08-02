# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def BatchNorm(inputs, momentum=0.9, eps=1e-3,
              use_stats=-1, inplace=True, **kwargs):
    """
    :param inputs:      a list of 4 Tensors contains [input, mean, var, factor]
                        tips:  mean, var, factor should be set to fill 0 before
    :param use_stats:   a int: set 0 or 1 force to not use or use stats
                               specially, set -1 will use(Train) / not use(Test)
                        tips:  set -1 when training with a large batchsize
                               set 0 when without doing batch statistics
                               (p.s statistics will poor if training with a small batchsize)
    :param decay:       a float of moving average factor
    :param eps:         a float of eps in sqrt(x + eps)
    :return:            a Tensor after BatchNorm, which will speed convergence
    """

    if not isinstance(inputs, list) or len(inputs) < 4:
        raise TypeError('BatchNorm Operator accpets a list of 4 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='BatchNorm', **kwargs)


def BatchRenorm(inputs, momentum=0.9, eps=1e-3, r_max=3.0, d_max=5.0,
                t_delta=1.0, use_stats=-1, inplace=True, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 4:
        raise TypeError('BatchRenorm Operator accpets a list of 4 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='BatchRenorm', **kwargs)


def BN(inputs, momentum=0.9, eps=1e-3, use_stats=-1, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 5:
        raise TypeError('BN Operator accpets a list of 5 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='BN', **kwargs)


def InstanceNorm(inputs, eps=1e-3, inplace=True, **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('Instance Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='InstanceNorm', **kwargs)


def L2Norm(inputs, axis=0, num_axes=-1, eps=1e-5, **kwargs):
    """
    """

    if not isinstance(inputs, Tensor):
        raise RuntimeError('L2Norm Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='L2Norm', **kwargs)
