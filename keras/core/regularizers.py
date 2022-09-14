# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Built-in regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import six
from dragon.vm.keras.core.utils import generic_utils


class Regularizer(object):
    """The base regularizer class."""

    def __call__(self, x):
        """Apply the regularizer to input.

        Parameters
        ----------
        dragon.Tensor
            The input tensor.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        if hasattr(self, 'l2'):
            x._weight_decay = self.l2
        return x


class L1(Regularizer):
    r"""The L1 regularizer.

    The **L1** regularizer is defined as:

    .. math:: loss_{reg} = loss + \alpha|w|

    """

    def __init__(self, l1=0.01):
        r"""Create a ``L1`` regularizer.

        Parameters
        ----------
        l1 : float, optional, default=0.01
            The value to :math:`\alpha`.

        """
        self.l1 = l1


class L1L2(Regularizer):
    r"""The L1L2 regularizer.

    The **L1L2** regularizer is defined as:

    .. math:: loss_{reg} = loss + \alpha|w| + \frac{\beta}{2}|w|_{2}

    """

    def __init__(self, l1=0.01, l2=0.01):
        r"""Create a ``L1L2`` regularizer.

        Parameters
        ----------
        l1 : float, optional, default=0.01
            The value to :math:`\alpha`.
        l2 : float, optional, default=0.01
            The value to :math:`\beta`.

        """
        self.l1, self.l2 = l1, l2


class L2(Regularizer):
    r"""The L2 regularizer.

    The **L2** regularizer is defined as:

    .. math:: loss_{reg} = loss + \frac{\beta}{2}|w|_{2}

    """

    def __init__(self, l2=0.01):
        r"""Create a ``L2`` regularizer.

        Parameters
        ----------
        l1 : float, optional, default=0.01
            The value to :math:`\alpha`.

        """
        self.l2 = l2


def l1_l2(l1=0.01, l2=0.01):
    r"""Create a L1L2 regularizer.

    The **L1L2** regularizer is defined as:

    .. math:: loss_{reg} = loss + \alpha|w| + \frac{\beta}{2}|w|_{2}

    Parameters
    ----------
    l1 : float, optional, default=0.01
        The value to :math:`\alpha`.
    l2 : float, optional, default=0.01
        The value to :math:`\beta`.

    Returns
    -------
    dragon.vm.tensorflow.keras.regularizers.Regularizer
        The regularizer.

    """
    return L1L2(l1=l1, l2=l2)


def get(identifier):
    """Return the regularizer callable by identifier.

    Parameters
    ----------
    identifier : Union[callable, str]
        The identifier.

    Returns
    -------
    callable
        The activation callable.

    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        if identifier == 'l1_l2':
            return L1L2(l1=0.01, l2=0.01)
        return generic_utils.deserialize_keras_object(
            identifier, globals(), 'regularizer')
    else:
        raise TypeError(
            'Could not interpret the regularizer identifier: {}.'
            .format(identifier))


# Aliases
l1 = L1
l2 = L2
