# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

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
