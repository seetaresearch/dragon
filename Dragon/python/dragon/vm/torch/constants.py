# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon.core.utils as pb_utils


# Enumerate and persistent device options
# Make item frequently will degrade performance
DEVICE_LIMITS = 16
DEVICE_ENGINE = 'CUDNN'
CTX_TO_DEVICE_OPTION = {('CPU', 0): pb_utils.MakeDeviceOption(0, 0)}
for i in range(DEVICE_LIMITS):
    CTX_TO_DEVICE_OPTION['CUDA', i] = \
        pb_utils.MakeDeviceOption(1, i, DEVICE_ENGINE)