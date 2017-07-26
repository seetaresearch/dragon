# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.scope import TensorScope as variable_scope
from dragon.core.scope import TensorScope as name_scope
from core.session import *
from core.variables import *
from core.collection import *
from core.device import *
import contrib
import ops.nn as nn
from ops import *
from training import train
from utils.gradients import *

