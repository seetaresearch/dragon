# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.util import six


def leaky_relu(x, alpha=0.2, name="leaky_relu", **kwargs):
    r"""Apply the leaky rectified linear unit.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tl.layers.Input([10, 200])
    y = tl.layers.Dense(
        n_units=100,
        act=tl.act.lrelu(x, 0.2),
        name='dense',
    )(x)
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=0.2
        The value to :math:`\alpha`.
    name : str, optional
        The operator name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if not (0 <= alpha <= 1):
        raise ValueError("`alpha` value must be in [0, 1]`")
    return activation_ops.leaky_relu(x, alpha=alpha, name=name, **kwargs)


def relu(x, name='relu', **kwargs):
    return leaky_relu(x, 0., name, **kwargs)


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        return globals()[identifier]()
    else:
        raise TypeError(
            'Could not interpret initializer identifier: {}.'
            .format(repr(identifier))
        )


# Alias
lrelu = leaky_relu
