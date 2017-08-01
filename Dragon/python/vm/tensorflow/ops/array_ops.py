# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'expand_dims',
    'shape',
    'zeros',
    'ones',
    'concat',
    'transpose',
    'tile',
    'reshape'
]

import dragon.ops as ops
from ..core import dtypes


def expand_dims(input, axis=None, name=None, dim=None):
    """
    Inserts a dimension of 1 into a tensor's shape.

      Given a tensor `input`, this operation inserts a dimension of 1 at the
      dimension index `axis` of `input`'s shape. The dimension index `axis` starts
      at zero; if you specify a negative number for `axis` it is counted backward
      from the end.

      This operation is useful if you want to add a batch dimension to a single
      element. For example, if you have a single image of shape `[height, width,
      channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
      which will make the shape `[1, height, width, channels]`.

      Other examples:

      ```python
      # 't' is a tensor of shape [2]
      shape(expand_dims(t, 0)) ==> [1, 2]
      shape(expand_dims(t, 1)) ==> [2, 1]
      shape(expand_dims(t, -1)) ==> [2, 1]

      # 't2' is a tensor of shape [2, 3, 5]
      shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
      shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
      shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
      ```

      This operation requires that:

      `-1-input.dims() <= dim <= input.dims()`

      This operation is related to `squeeze()`, which removes dimensions of
      size 1.

      Args:
        input: A `Tensor`.
        axis: 0-D (scalar). Specifies the dimension index at which to
          expand the shape of `input`.
        name: The name of the output `Tensor`.
        dim: 0-D (scalar). Equivalent to `axis`, to be deprecated.

      Returns:
        A `Tensor` with the same data as `input`, but its shape has an additional
        dimension of size 1 added.

    """

    if dim is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dim'.")
        axis = dim

    return ops.ExpandDims(input, axis=axis, name=name)


def shape(input, name=None, out_type=dtypes.float32):
    """
    Returns the shape of a tensor.

      This operation returns a 1-D integer tensor representing the shape of `input`.

      For example:

      ```python
      # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
      shape(t) ==> [2, 2, 3]
      ```

      Args:
        input: A `Tensor`.
        name: A name for the operation (optional).
        out_type: (Enforce) The specified output type of the operation.
                            Now only support tf.float32.

      Returns:
        A `Tensor` of type `out_type`.

    """

    return ops.Shape(input, name=None)

def zeros(shape, dtype=dtypes.float32, name=None):
    return ops.Fill(shape, value=0.0, name=name)


def ones(shape, dtype=dtypes.float32, name=None):
    return ops.Fill(shape, value=1.0, name=name)


def concat(values, axis, name=None):

    """
    Concatenates tensors along one dimension.

      Concatenates the list of tensors `values` along dimension `axis`.  If
      `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
      result has shape

          [D0, D1, ... Raxis, ...Dn]

      where

          Raxis = sum(Daxis(i))

      That is, the data from the input tensors is joined along the `axis`
      dimension.

      The number of dimensions of the input tensors must match, and all dimensions
      except `axis` must be equal.

      For example:

      ```python
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
      tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

      # tensor t3 with shape [2, 3]
      # tensor t4 with shape [2, 3]
      tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
      tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
      ```

      Args:
        values: A list of `Tensor` objects or a single `Tensor`.
        axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` resulting from concatenation of the input tensors.

    """

    return ops.Concat(values, axis=axis, name=name)


def transpose(a, perm=None, name=None):
    """
    Transposes `a`. Permutes the dimensions according to `perm`.

      The returned tensor's dimension i will correspond to the input dimension
      `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
      the rank of the input tensor. Hence by default, this operation performs a
      regular matrix transpose on 2-D input Tensors.

      For example:

      ```python
      # 'x' is [[1 2 3]
      #         [4 5 6]]
      tf.transpose(x) ==> [[1 4]
                           [2 5]
                           [3 6]]

      # Equivalently
      tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                        [2 5]
                                        [3 6]]

      # 'perm' is more useful for n-dimensional tensors, for n > 2
      # 'x' is   [[[1  2  3]
      #            [4  5  6]]
      #           [[7  8  9]
      #            [10 11 12]]]
      # Take the transpose of the matrices in dimension-0
      tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                            [2  5]
                                            [3  6]]

                                           [[7 10]
                                            [8 11]
                                            [9 12]]]
      ```

      Args:
        a: A `Tensor`.
        perm: A permutation of the dimensions of `a`.
        name: A name for the operation (optional).

      Returns:
        A transposed `Tensor`.

      """

    return ops.Transpose(a, perm=perm, name=name)


def tile(input, multiples, name=None):
    return ops.Tile(input, multiples=multiples, name=name)


def reshape(tensor, shape, name=None):
    return ops.Reshape(tensor, shape=shape, name=None)
