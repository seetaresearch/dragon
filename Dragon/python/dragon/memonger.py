# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""A simple wrapper for memory optimization tricks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import config as _cfg


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
    options = _cfg.GetGlobalOptions()
    options['share_grads'] = enabled


def IsGradsShared():
    """Is grads are shared?

    Returns
    -------
    boolean
        ``True`` if sharing grads else ``False``.

    """
    options = _cfg.GetGlobalOptions()
    return options['share_grads']


def Drop(op_func, *args, **kwargs):
    """Drop(Share) the inputs for outputs.

    Parameters
    ----------
    op_func : lambda
        The function of any operators.

    Returns
    -------
    dragon.Tensor or list[dragon.Tensor]
        The outputs of the given operator.

    Examples
    --------
    >>> import dragon as dg
    >>> import dragon.memonger as opt
    >>> data = dg.Tensor().Variable()
    >>> conv_1 = dg.Conv2d(data, num_output=8)
    >>> conv_1_bn = opt.Drop(dg.BatchNorm, [conv_1, dg.Tensor().Variable(), dg.Tensor.Variable()])
    >>> conv_1_relu = opt.Drop(dg.Relu, conv_1_bn)

    """
    kwargs['mirror_stage'] = True
    return op_func(*args, **kwargs)