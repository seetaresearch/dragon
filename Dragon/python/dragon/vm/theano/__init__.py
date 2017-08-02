# --------------------------------------------------------
# Theano for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np

from .core.swap import shared
from .core.function import function
from .core.scan import scan

floatX = np.float32

