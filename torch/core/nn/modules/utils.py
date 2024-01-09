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
"""Module utilities."""

import collections
import itertools


def _get_loss_reduction(size_average, reduce, reduction):
    """Return the loss reduction."""
    if size_average is not None or reduce is not None:
        reduce = True if reduce is None else reduce
        size_average = True if size_average is None else size_average
        if size_average and reduce:
            reduction = "mean"
        elif reduce:
            reduction = "sum"
        else:
            reduction = "none"
    return reduction


def _get_adaptive_pool_args(input, output_sizes):
    """Return the adaptive pooling arguments."""
    axis = 1 if input.device.type == "mlu" else 2
    input_sizes = input.size()[axis : axis + len(output_sizes)]
    stride, kernel_size = [], []
    for input_size, output_size in zip(input_sizes, output_sizes):
        if output_size == 1:
            stride.append(1)
            kernel_size.append(input_size)
        else:
            stride.append(input_size // output_size)
            kernel_size.append(input_size - (output_size - 1) * stride[-1])
    return {"kernel_size": kernel_size, "stride": stride}


def _ntuple(n):
    """Return the fixed-length tuple creator."""
    def parse(x):
        if isinstance(x, collections.abc.Sequence):
            return x
        return tuple(itertools.repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
