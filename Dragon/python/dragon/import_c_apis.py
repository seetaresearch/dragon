# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging

try:
    from dragon.libdragon import *
except ImportError as e:
    logging.critical(
        'cannot load dragon. Error: {0}'.format(str(e)))
    sys.exit(1)