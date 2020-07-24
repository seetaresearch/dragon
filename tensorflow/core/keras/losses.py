# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/losses.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.core.ops import loss_ops
from dragon.core.util import six
from dragon.vm.tensorflow.core.keras.utils import losses_utils


class Loss(object):
    """The base class for loss criterion."""

    def __init__(self, reduction=losses_utils.Reduction.MEAN, name=None):
        """Create a ``Loss`` criterion.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean', 'valid'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        losses_utils.Reduction.validate(reduction)
        self.reduction = reduction
        self.name = name

    def call(self, y_true, y_pred):
        """Define the loss computation."""
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        """Compute the defined loss function.

        Parameters
        ----------
        y_true : dragon.Tensor
            The ground-truth tensor.
        y_pred : dragon.Tensor
            The logits tensor.

        Returns
        -------
        dragon.Tensor
            The loss.

        """
        scope_name = 'lambda' if self.name == '<lambda>' else self.name
        with context.name_scope(scope_name or self.__class__.__name__):
            return self.call(y_true, y_pred)


class LossFunctionWrapper(Loss):
    """Wrap loss class over the function."""

    def __init__(
        self,
        fn,
        reduction=losses_utils.Reduction.SUM,
        name=None,
        **kwargs
    ):
        super(LossFunctionWrapper, self).__init__(
            reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred, **self._fn_kwargs)


class BinaryCrossentropy(LossFunctionWrapper):
    r"""A criterion to compute the binary cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    criterion = tf.keras.losses.BinaryCrossentropy()
    y_true = tf.constant([0., 0., 1., 1.])
    y_pred = tf.constant([0.1, 0.2, 0.3, 0.4])
    print(criterion(y_true, y_pred))  # 0.65247655
    ```

    """

    def __init__(
        self,
        reduction=losses_utils.Reduction.MEAN,
        name=None,
    ):
        """Create a ``BinaryCrossentropy`` criterion.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean', 'valid'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        super(BinaryCrossentropy, self).__init__(
            binary_crossentropy,
            reduction=reduction,
            name=name,
        )


class CategoricalCrossentropy(LossFunctionWrapper):
    r"""A criterion to compute the categorical cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    criterion = tf.keras.losses.CategoricalCrossentropy()
    y_true = tf.constant([[0., 1.], [1., 0.]])
    y_pred = tf.constant([[0.5, 0.5], [0.3, 0.7]])
    print(criterion(y_true, y_pred))  # 0.8030813
    ```

    """

    def __init__(
        self,
        axis=-1,
        reduction=losses_utils.Reduction.MEAN,
        name=None,
    ):
        """Create a ``CategoricalCrossentropy`` criterion.

        Parameters
        ----------
        axis : int, optional, default=-1
            The axis to apply softmax.
        reduction : {'none', 'sum', 'mean', 'valid'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        super(CategoricalCrossentropy, self).__init__(
            categorical_crossentropy,
            axis=axis,
            reduction=reduction,
            name=name,
        )


class MeanAbsoluteError(LossFunctionWrapper):
    r"""A criterion to compute the reduced element-wise absolute value difference.

    The **AbsoluteError** function is defined as:

    .. math:: \text{AbsoluteError}(y_{true}, y_{pred}) = |y_{pred} - y_{true}|

    Examples:

    ```python
    criterion = tf.keras.losses.MeanAbsoluteError()
    y_true = tf.constant([1., 2., 3.])
    y_pred = tf.constant([0., 0., 0.])
    print(criterion(y_true, y_pred))  # 2.0
    ```

    """

    def __init__(
        self,
        reduction=losses_utils.Reduction.MEAN,
        name=None,
    ):
        """Create a ``MeanAbsoluteError`` criterion.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        super(MeanAbsoluteError, self).__init__(
            mean_absolute_error,
            reduction=reduction,
            name=name,
        )


class MeanSquaredError(LossFunctionWrapper):
    r"""A criterion to compute the reduced element-wise squared error.

    The **SquaredError** function is defined as:

    .. math:: \text{SquaredError}(y_{true}, y_{pred}) = (y_{pred} - y_{true})^{2}

    Examples:

    ```python
    criterion = tf.keras.losses.MeanSquaredError()
    y_true = tf.constant([1., 2., 3.])
    y_pred = tf.constant([0., 0., 0.])
    print(criterion(y_true, y_pred))  # 4.666666
    ```

    """

    def __init__(
        self,
        reduction=losses_utils.Reduction.MEAN,
        name=None,
    ):
        """Create a ``MeanSquaredError`` criterion.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        super(MeanSquaredError, self).__init__(
            mean_squared_error,
            reduction=reduction,
            name=name,
        )


class SparseCategoricalCrossentropy(LossFunctionWrapper):
    r"""A criterion to compute the categorical cross entropy with sparse labels.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    y_true = tf.constant([1, 0], 'int64')
    y_pred = tf.constant([[0.5, 0.5], [0.3, 0.7]])
    print(criterion(y_true, y_pred))  # 0.8030813
    ```

    """

    def __init__(
        self,
        axis=-1,
        ignore_index=None,
        reduction=losses_utils.Reduction.VALID,
        name=None,
    ):
        """Create a ``SparseCategoricalCrossentropy`` criterion.

        Parameters
        ----------
        axis : int, optional, default=-1
            The reduce axis.
        ignore_index : int, optional
            The label index to ignore.
        reduction : {'none', 'sum', 'mean', 'valid'}, optional
            The reduction method.
        name : str, optional
            A optional name for the operation.

        """
        super(SparseCategoricalCrossentropy, self).__init__(
            sparse_categorical_crossentropy,
            reduction=reduction,
            name=name,
            axis=axis,
            ignore_index=ignore_index,
        )


def binary_crossentropy(
    y_true,
    y_pred,
    reduction=losses_utils.Reduction.VALID,
):
    r"""Compute the binary cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    y_true = tf.constant([0., 0., 1., 1.])
    y_pred = tf.constant([0.1, 0.2, 0.3, 0.4])
    print(tf.keras.losses.binary_crossentropy(y_true, y_pred))  # 0.65247655
    ```

    Parameters
    ----------
    y_true : dragon.Tensor
        The ground truth tensor.
    y_pred : dragon.Tensor
        The logits tensor.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return loss_ops.sigmoid_cross_entropy(
        [y_pred, y_true],
        reduction=reduction.upper(),
    )


def categorical_crossentropy(
    y_true,
    y_pred,
    axis=-1,
    reduction=losses_utils.Reduction.MEAN,
):
    r"""Compute the categorical cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    y_true = tf.constant([[0., 1.], [1., 0.]])
    y_pred = tf.constant([[0.5, 0.5], [0.3, 0.7]])
    print(tf.keras.losses.categorical_crossentropy(y_true, y_pred))  # 0.8030813
    ```

    Parameters
    ----------
    y_true : dragon.Tensor
        The ground truth tensor.
    y_pred : dragon.Tensor
        The logits tensor.
    axis : int, optional, default=-1
        The reduce axis.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The loss.

    """
    return loss_ops.softmax_cross_entropy(
        [y_pred, y_true],
        axis=axis,
        reduction=reduction.upper(),
    )


def mean_absolute_error(
    y_true,
    y_pred,
    reduction=losses_utils.Reduction.MEAN,
):
    r"""Compute the reduced element-wise absolute value difference.

    The **AbsoluteError** function is defined as:

    .. math:: \text{AbsoluteError}(y_{true}, y_{pred}) = |y_{pred} - y_{true}|

    Examples:

    ```python
    y_true = tf.constant([1., 2., 3.])
    y_pred = tf.constant([0., 0., 0.])
    print(tf.keras.losses.mean_absolute_error(y_true, y_pred))  # 2.0
    ```

    Parameters
    ----------
    y_true : dragon.Tensor
        The ground truth tensor.
    y_pred : dragon.Tensor
        The logits tensor.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    """
    return loss_ops.l1_loss(
        [y_pred, y_true],
        reduction=reduction.upper(),
    )


def mean_squared_error(
    y_true,
    y_pred,
    reduction=losses_utils.Reduction.MEAN,
):
    r"""Compute the reduced element-wise squared error.

    The **SquaredError** function is defined as:

    .. math:: \text{SquaredError}(y_{true}, y_{pred}) = (y_{pred} - y_{true})^{2}

    Examples:

    ```python
    y_true = tf.constant([1., 2., 3.])
    y_pred = tf.constant([0., 0., 0.])
    print(tf.keras.losses.mean_squared_error(y_true, y_pred))  # 4.666666
    ```

    Parameters
    ----------
    y_true : dragon.Tensor
        The ground truth tensor.
    y_pred : dragon.Tensor
        The logits tensor.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The loss.

    """
    return loss_ops.l2_loss(
        [y_pred, y_true],
        reduction=reduction.upper(),
    )


def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    axis=-1,
    ignore_index=None,
    reduction=losses_utils.Reduction.VALID,
):
    r"""Compute the categorical cross entropy with sparse labels.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    y_true = tf.constant([1, 0], 'int64')
    y_pred = tf.constant([[0.5, 0.5], [0.3, 0.7]])
    print(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))  # 0.8030813
    ```

    Parameters
    ----------
    y_true : dragon.Tensor
        The ground truth tensor.
    y_pred : dragon.Tensor
        The logits tensor.
    axis : int, optional, default=-1
        The reduce axis.
    ignore_index : int, optional
        The label index to ignore.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The loss.

    """
    return loss_ops.sparse_softmax_cross_entropy(
        [y_pred, y_true],
        axis=axis,
        reduction=reduction,
        ignore_index=ignore_index,
    )


def get(identifier):
    """Return the loss callable by identifier.

    Parameters
    ----------
    identifier : Union[callable, str]
        The identifier.

    Returns
    -------
    callable
        The loss callable.

    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        return globals()[identifier]
    else:
        raise TypeError(
            'Could not interpret loss identifier: {}.'
            .format(repr(identifier)))
