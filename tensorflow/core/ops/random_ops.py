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
"""Random ops."""

from dragon.core.ops import random_ops


def random_normal(
    shape,
    mean=0,
    stddev=1,
    dtype="float32",
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    stddev : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, dtype, random_ops.random_normal
    return init_fn(shape, mean, stddev, dtype=dtype, name=name)


def random_uniform(
    shape,
    minval=0,
    maxval=1,
    dtype="float32",
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from the uniform distribution.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    minval : number, optional, default=0
        The value to :math:`\alpha`.
    maxval : number, optional, default=1
        The value to :math:`\beta`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, dtype, random_ops.random_uniform
    return init_fn(shape, minval, maxval, dtype=dtype, name=name)


def truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype="float32",
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from the truncated normal distribution.

    .. math::
        \text{out} \sim \mathcal{TN}(\mu, \sigma^{2}, \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    stddev : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, dtype, random_ops.truncated_normal
    return init_fn(shape, mean, stddev, dtype=dtype, name=name)
