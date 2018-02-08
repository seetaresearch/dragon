# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from .ops import Graph

# Utilities used when building a Graph.
from dragon.vm.tensorflow.framework.ops import device
from dragon.vm.tensorflow.framework.ops import name_scope
from dragon.vm.tensorflow.framework.ops import get_default_graph
from dragon.vm.tensorflow.framework.ops import add_to_collection
from dragon.vm.tensorflow.framework.ops import get_collection
from dragon.vm.tensorflow.framework.ops import convert_to_tensor
from dragon.vm.tensorflow.framework.ops import GraphKeys
from dragon.vm.tensorflow.framework.constant_op import *

from dragon.vm.tensorflow.framework.dtypes import *


from dragon.vm.tensorflow.framework.tensor_shape import Dimension
from dragon.vm.tensorflow.framework.tensor_shape import TensorShape