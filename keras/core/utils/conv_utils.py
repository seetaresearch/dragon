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
"""Convolution utilities."""

from dragon.core.util import nest


def convert_data_format(data_format, ndim=4):
    """Return the tf data format."""
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError("Input rank not supported: " + str(ndim))
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError("Input rank not supported: " + str(ndim))
    else:
        raise ValueError("Invalid data_format: " + data_format)


def normalize_data_format(value):
    """Normalize the keras data format."""
    if value is None:
        value = "channels_last"
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "Excepted <data_format> as one of "
            '"channels_first", '
            '"channels_last".'
            " Received: " + str(value)
        )
    return data_format


def normalize_padding(value):
    """Return the padding descriptor."""
    if isinstance(value, str):
        value = value.lower()
        if value not in {"valid", "same"}:
            raise ValueError('Excepted <padding> in "valid", "same".\n' "Received: " + str(value))
    return value


def normalize_tuple(value, rank):
    """Repeat the value according to the rank."""
    value = nest.flatten(value)
    if len(value) > rank:
        return (value[i] for i in range(rank))
    else:
        return tuple(
            [value[i] for i in range(len(value))] + [value[-1] for _ in range(len(value), rank)]
        )


def normalize_paddings(value, rank):
    """Repeat the paddings according to the rank."""
    if isinstance(value, int):
        return ((value, value),) * rank
    elif nest.is_sequence(value):
        value = [normalize_tuple(v, 2) for v in value]
        if len(value) > rank:
            return tuple(value[i] for i in range(rank))
        return tuple(
            [value[i] for i in range(len(value))] + [value[-1] for _ in range(len(value), rank)]
        )
