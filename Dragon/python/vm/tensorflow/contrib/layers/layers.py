# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.ops as ops

__all__ = ['flatten']

def flatten(inputs, name=None):

    """
    Flattens the input while maintaining the batch_size.

      Assumes that the first dimension represents the batch.

      Args:
        inputs: A tensor of size [batch_size, ...].
        scope: Optional name for operation.

      Returns:
        A flattened tensor with shape [batch_size, k].

    """

    return ops.Flatten(inputs, axis=1)

