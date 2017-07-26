# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import logging
import sys

try:
    from dragon.libdragon import *
except ImportError as e:
    logging.critical(
        'cannot load dragon. Error: {0}'.format(str(e)))
    sys.exit(1)

from dragon.core.scope import TensorScope as name_scope
from dragon.core.scope import PhaseScope as phase_scope
from dragon.core.scope import DeviceScope as device_scope

