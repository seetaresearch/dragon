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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/initializers.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import six
from dragon.vm.keras.core.utils import generic_utils
from dragon.vm.tensorflow.core.ops.init_ops import Constant
from dragon.vm.tensorflow.core.ops.init_ops import GlorotNormal
from dragon.vm.tensorflow.core.ops.init_ops import GlorotUniform
from dragon.vm.tensorflow.core.ops.init_ops import Initializer  # noqa
from dragon.vm.tensorflow.core.ops.init_ops import RandomNormal
from dragon.vm.tensorflow.core.ops.init_ops import RandomUniform
from dragon.vm.tensorflow.core.ops.init_ops import TruncatedNormal
from dragon.vm.tensorflow.core.ops.init_ops import Ones
from dragon.vm.tensorflow.core.ops.init_ops import VarianceScaling  # noqa
from dragon.vm.tensorflow.core.ops.init_ops import Zeros

# Aliases
zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform


def get(identifier):
    """Return the initializer callable by identifier.

    Parameters
    ----------
    identifier : Union[callable, str]
        The identifier.

    Returns
    -------
    callable
        The initializer callable.

    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        return generic_utils.deserialize_keras_object(
            identifier, globals(), 'initializer')
    else:
        raise TypeError(
            'Could not interpret the initializer identifier: {}.'
            .format(identifier))
