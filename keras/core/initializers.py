# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Initializer functions."""

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
    elif isinstance(identifier, str):
        return generic_utils.deserialize_keras_object(identifier, globals(), "initializer")
    else:
        raise TypeError("Could not interpret the initializer identifier: {}.".format(identifier))
