# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'argmax',
    'matmul',
    'add',
    'subtract',
    'multiply',
    'divide',
    'sub',
    'mul',
    'div',
    'log',
    'exp',
    'square',
    'sqrt',
    'reduce_sum',
    'reduce_mean',
    'sigmoid',
    'tanh',
    'add_n'
]

import dragon.ops as ops


def argmax(input, axis=None, name=None, dimension=None):
    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return ops.Argmax(input, axis=axis, name=name)


def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           name=None):

    """
    Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

      The inputs must be matrices (or tensors of rank > 2, representing batches of
      matrices), with matching inner dimensions, possibly after transposition.

      Both matrices must be of the same type. The supported types are:
      `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

      Either matrix can be transposed or adjointed (conjugated and transposed) on
      the fly by setting one of the corresponding flag to `True`. These are `False`
      by default.

      If one or both of the matrices contain a lot of zeros, a more efficient
      multiplication algorithm can be used by setting the corresponding
      `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
      This optimization is only available for plain matrices (rank-2 tensors) with
      datatypes `bfloat16` or `float32`.

      For example:

      ```python
      # 2-D tensor `a`
      a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                            [4. 5. 6.]]
      # 2-D tensor `b`
      b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                               [9. 10.]
                                                               [11. 12.]]
      c = tf.matmul(a, b) => [[58 64]
                              [139 154]]

      Args:
        a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
          `complex128` and rank > 1.
        b: `Tensor` with same type and rank as `a`.
        transpose_a: If `True`, `a` is transposed before multiplication.
        transpose_b: If `True`, `b` is transposed before multiplication.
        name: Name for the operation (optional).

      Returns:
        A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
        the product of the corresponding matrices in `a` and `b`, e.g. if all
        transpose are `False`:

        `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
        for all indices i, j.

      Note: This is matrix product, not element-wise product.

    """

    return ops.Dot([a, b], TransA=transpose_a, TransB=transpose_b, name=name)


def add(x, y, name=None):
    return ops.Add([x, y], name=None)


def subtract(x, y, name=None):
    return ops.Sub([x, y], name=name)


def multiply(x, y, name=None):
    return ops.Mul([x, y], name=name)


def divide(x, y, name=None):
    return ops.Div([x, y], name=name)


def mul(x, y, name=None):
    return multiply(x, y, name)


def sub(x, y, name=None):
    return subtract(x, y, name)


def div(x, y, name=None):
    return divide(x, y, name=name)


def log(x, name=None):

    """
    Computes log of x element-wise.

      I.e., \\(y = log(x)\\).

      Args:
        x: A `Tensor`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `x`.

    """

    return ops.Log(x, name=name)


def exp(x, name=None):

    """
    Computes exp of x element-wise.

      I.e., \\(y = exp(x)\\).

      Args:
        x: A `Tensor`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `x`.

    """

    return ops.Exp(x, name=name)


def square(x, name=None):

    """
    Computes square of x element-wise.

      I.e., \\(y = x * x = x^2\\).

      Args:
        x: A `Tensor`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `x`.

    """

    return ops.Square(x, name=name)


def sqrt(x, name=None):

    """
    Computes square root of x element-wise.

      I.e., \\(y = \sqrt{x} = x^{1/2}\\).

      Args:
        x: A `Tensor`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `x`.

    """

    return ops.Pow(x, power=0.5, name=name)


def pow(x, power, name=None):

    """
    Computes pow of x element-wise.

      I.e., \\(y = \pow{x} = x^{power}\\).

      Args:
        x: A `Tensor`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `x`.

    """

    return ops.Pow(x, power=power, name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    """
    Computes the sum of elements across dimensions of a tensor.

      Reduces `input_tensor` along the dimensions given in `axis`.
      Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
      entry in `axis`. If `keep_dims` is true, the reduced dimensions
      are retained with length 1.

      If `axis` has no entries, all dimensions are reduced, and a
      tensor with a single element is returned.

      For example:

      ```python
      # 'x' is [[1, 1, 1]
      #         [1, 1, 1]]
      tf.reduce_sum(x) ==> 6
      tf.reduce_sum(x, 0) ==> [2, 2, 2]
      tf.reduce_sum(x, 1) ==> [3, 3]
      tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
      tf.reduce_sum(x, [0, 1]) ==> 6
      ```

      Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
          reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).
        reduction_indices: The old (deprecated) name for axis.

      Returns:
        The reduced tensor.
    """

    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    elif axis is None: axis = -1 # reduce all
    if isinstance(axis, list) or isinstance(axis, tuple): # reduce continuously
        if len(axis) < 1:
            raise RuntimeError('reduce axes should at least have one.')
        if len(axis) == 1:
            return ops.Sum(input_tensor, axis=axis[0], keep_dims=keep_dims)
        else:
            ret = ops.Sum(input_tensor, axis=axis[0], keep_dims=True)
            for i in xrange(1, len(axis) - 1):
                ret = ops.Sum(ret, axis=axis[i], keep_dims=True)
            return ops.Sum(ret, axis=axis[len(axis) - 1], keep_dims=keep_dims)
    else:
        return ops.Sum(input_tensor, axis=axis, keep_dims=keep_dims)


def reduce_mean(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):

    """
    Computes the mean of elements across dimensions of a tensor.

      Reduces `input_tensor` along the dimensions given in `axis`.
      Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
      entry in `axis`. If `keep_dims` is true, the reduced dimensions
      are retained with length 1.

      If `axis` has no entries, all dimensions are reduced, and a
      tensor with a single element is returned.

      For example:

      ```python
      # 'x' is [[1., 1.]
      #         [2., 2.]]
      tf.reduce_mean(x) ==> 1.5
      tf.reduce_mean(x, 0) ==> [1.5, 1.5]
      tf.reduce_mean(x, 1) ==> [1.,  2.]
      ```

      Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
          reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).
        reduction_indices: The old (deprecated) name for axis.

      Returns:
        The reduced tensor.

    """

    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    elif axis is None: axis = -1 # reduce all
    if isinstance(axis, list) or isinstance(axis, tuple): # reduce continuously
        if len(axis) < 1:
            raise RuntimeError('reduce axes should at least have one.')
        if len(axis) == 1:
            return ops.Mean(input_tensor, axis=axis[0], keep_dims=keep_dims)
        else:
            ret = ops.Mean(input_tensor, axis=axis[0], keep_dims=True)
            for i in xrange(1, len(axis) - 1):
                ret = ops.Mean(ret, axis=axis[i], keep_dims=True)
            return ops.Mean(ret, axis=axis[len(axis) - 1], keep_dims=keep_dims)
    else:
        return ops.Mean(input_tensor, axis=axis, keep_dims=keep_dims)


def sigmoid(x, name=None):

    """
    Computes sigmoid of `x` element-wise.

     Specifically, `y = 1 / (1 + exp(-x))`.

     Args:
       x: A Tensor.
       name: A name for the operation (optional).

     Returns:
       A Tensor with the same type.

    """

    return ops.Sigmoid(x, name=name)


def tanh(x, name=None):

    """
    Computes hyperbolic tangent of `x` element-wise.

     Args:
       x: A Tensor.
       name: A name for the operation (optional).

     Returns:
       A Tensor with the same type.

    """

    return ops.Tanh(x, name=name)


def add_n(inputs, name=None):
    """
    Adds all input tensors element-wise.

      Args:
        inputs: A list of `Tensor` objects, each with same shape and type.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of same shape and type as the elements of `inputs`.

    """

    return ops.Eltwise(inputs, operation='SUM', name=name)
