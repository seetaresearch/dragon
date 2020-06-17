# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/network.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from dragon.core.util import nest
from dragon.vm.tensorflow.core.keras.engine import base_layer


class Network(base_layer.Layer):
    """Compose a group of layers."""

    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(**kwargs)
        self._thread_local = threading.local()
        self._is_compiled = False
        self._updates = []
        self._losses = []
        self._metrics = []
        self.inputs = []
        self.outputs = []
        if not hasattr(self, 'optimizer'):
            self.optimizer = None
        if (len(args) == 2 or
                len(args) == 1 and 'outputs' in kwargs or
                'inputs' in kwargs and 'outputs' in kwargs):
            self._init_graph_network(*args, **kwargs)
        else:
            self._init_subclassed_network(**kwargs)

    def _init_graph_network(self, inputs, outputs, **kwargs):
        self._is_graph_network = True
        if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
            inputs = inputs[0]
        if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
            outputs = outputs[0]
        self._nested_outputs = outputs
        self._nested_inputs = inputs
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.built = True

    def _init_subclassed_network(self, **kwargs):
        self._is_graph_network = False
        self.built = False
