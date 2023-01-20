# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Constant mappings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Mapping to store the supported device types
DEVICE_STRING_TO_DEVICE_TYPE = {
    'cpu': 'cpu',
    'gpu': 'cuda',
    'cuda': 'cuda',
    'mps': 'mps',
    'mlu': 'mlu',
}
