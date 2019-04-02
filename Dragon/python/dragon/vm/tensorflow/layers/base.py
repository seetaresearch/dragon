# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
import weakref

from dragon.vm.tensorflow.framework import ops
from dragon.vm.tensorflow.framework import dtypes
from dragon.vm.tensorflow.ops import var_scope as vs
from dragon.vm.tensorflow.util import nest


class Layer(object):
    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=dtypes.float32,
        **kwargs
    ):
        allowed_kwargs = {'_scope', '_reuse'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        self.trainable = trainable
        self.built = False
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._updates = []
        self._losses = []
        self._reuse = kwargs.get('_reuse')
        self._graph = ops.get_default_graph()
        self._per_input_losses = {}
        self._per_input_updates = {}
        self.dtype = dtypes.as_dtype(dtype)
        self.input_spec = None

        # Determine layer name
        if name is None:
            base_name = _to_snake_case(self.__class__.__name__)
            self.name = _unique_layer_name(base_name)
        else:
            base_name = name
            self.name = name

        self._base_name = base_name

    def build(self, _):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        raise NotImplementedError

    @property
    def updates(self):
        return self._updates

    def __call__(self, inputs, *args, **kwargs):
        with vs.variable_scope(self.name,
            reuse=self.built or self._reuse) as scope:
            if not self.built:
                input_shapes = [x.get_shape() for x in nest.flatten(inputs)]
                if len(input_shapes) == 1: self.build(input_shapes[0])
                else: self.build(input_shapes)
            outputs = self.call(inputs, *args, **kwargs)
            # Update global default collections.
            _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
            return outputs

    def add_variable(
        self,
        name,
        shape,
        dtype=None,
        trainable=True,
        initializer=None,
        regularizer=None,
    ):
        if dtype is None: dtype = self.dtype
        variable = vs.get_variable(
            name,
            shape=shape,
            initializer=initializer,
            regularizer=regularizer,
            dtype=dtypes.as_dtype(dtype),
            trainable=trainable and self.trainable,
        )
        if trainable:
            self._trainable_weights.append(variable)
        else:
            self._non_trainable_weights.append(variable)
        return variable

    def apply(self, inputs, *args, **kwargs):
        return self.__call__(inputs, *args, **kwargs)


class InputSpec(object):
    def __init__(
        self,
        dtype=None,
        shape=None,
        ndim=None,
        max_ndim=None,
        min_ndim=None,
        axes=None,
    ):
        self.dtype = dtype
        self.shape = shape
        if shape is not None: self.ndim = len(shape)
        else: self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}


def _to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    if insecure[0] != '_': return insecure
    return 'private' + insecure


def _unique_layer_name(name):
    global PER_GRAPH_LAYER_NAME_UIDS
    graph = ops.get_default_graph()
    if graph not in PER_GRAPH_LAYER_NAME_UIDS:
        PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
    layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
    layer_name_uids[name] += 1
    return name + '_' + str(layer_name_uids[name])


def _to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _add_elements_to_collection(elements, collection_list):
    elements = _to_list(elements)
    collection_list = _to_list(collection_list)
    for name in collection_list:
        collection = ops.get_collection_ref(name)
        collection_set = set(collection)
        for element in elements:
            if element not in collection_set:
                collection.append(element)


PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()