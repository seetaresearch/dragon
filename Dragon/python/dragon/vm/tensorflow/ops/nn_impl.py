# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.ops as ops

def batch_normalization(x, mean, variance,
                        offset, scale,
                        decay=0.9, variance_epsilon=1e-3, name=None):
    raise NotImplementedError('Deprecated. Use tf.layer.batch_normalization.')



def batch_norm_with_global_normalization(t, m, v,
                                         beta, gamma,
                                         decay=0.9, variance_epsilon=1e-3,
                                         scale_after_normalization=True, name=None):
    raise NotImplementedError('Deprecated. Use tf.layer.batch_normalization.')


def l2_normalize(x, dim, epsilon=1e-12, name=None):
    return ops.L2Norm(inputs=x,
                      axis=dim,
                      num_axes=1,
                      eps=epsilon)
