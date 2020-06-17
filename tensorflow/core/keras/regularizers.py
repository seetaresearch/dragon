# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/regularizers.py>
#
# ------------------------------------------------------------

"""Built-in regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
        x.__regularizer__ = self
        return x


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
            The value of :math:`\alpha`.
        l2 : float, optional, default=0.01
            The value of :math:`\beta`.

        """
        if l1 <= 0. or l2 <= 0.:
            raise ValueError('<l1> and <l2> should be greater than 0.')
        self.l1, self.l2 = l1, l2


def get(identifier):
    """Return a regularizer from the identifier."""
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            'Could not interpret regularizer identifier:',
            identifier,
        )


# Aliases
def l1(l=0.01):
    r"""Create a L1 regularizer.

    The **L1** regularizer is defined as:

    .. math:: loss_{reg} = loss + \alpha|w|

    Parameters
    ----------
    l : float, optional, default=0.01
        The value of :math:`\alpha`.

    Returns
    -------
    dragon.vm.tensorflow.keras.regularizers.Regularizer
        The regularizer.

    """
    return L1L2(l1=l)


def l1_l2(l1=0.01, l2=0.01):
    r"""Create a L1L2 regularizer.

    The **L1L2** regularizer is defined as:

    .. math:: loss_{reg} = loss + \alpha|w| + \frac{\beta}{2}|w|_{2}

    Parameters
    ----------
    l1 : float, optional, default=0.01
        The value of :math:`\alpha`.
    l2 : float, optional, default=0.01
        The value of :math:`\beta`.

    Returns
    -------
    dragon.vm.tensorflow.keras.regularizers.Regularizer
        The regularizer.

    """
    return L1L2(l1=l1, l2=l2)


def l2(l=0.01):
    r"""Create a L2 regularizer.

    The **L2** regularizer is defined as:

    .. math:: loss_{reg} = loss + \frac{\beta}{2}|w|_{2}

    Parameters
    ----------
    l : float, optional, default=0.01
        The value of :math:`\beta`.

    Returns
    -------
    dragon.vm.tensorflow.keras.regularizers.Regularizer
        The regularizer.

    """
    return L1L2(l2=l)
