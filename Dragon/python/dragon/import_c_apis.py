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

import sys
import logging
import atexit

try:
    from dragon.libdragon import *
except ImportError as e:
    logging.critical(
        'Cannot import dragon. Error: {0}'.format(str(e)))
    sys.exit(1)

atexit.register(OnModuleExitCC)