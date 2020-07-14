# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The base layer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.tensor import TensorRef
from dragon.core.eager import context as eager_context
from dragon.core.framework import context
from dragon.core.util import logging
from dragon.vm.caffe.proto import caffe_pb2


class Layer(object):
    """The abstraction of ``caffe.Layer``."""

    def __init__(self, layer_param):
        """Create a ``Layer``.

        Parameters
        ----------
        layer_param : LayerParameter
            The parameter containing layer arguments.

        """
        self._proto = layer_param
        self._name = layer_param.name
        self._arguments, self.arguments = {'name': 'output'}, {}
        # Store the inputs, outputs and trainable parameters.
        self._bottom, self._top, self._blobs = [], [], []
        for blob in layer_param.bottom:
            self._bottom.append(blob)
        for blob in layer_param.top:
            self._top.append(blob)
        # Store the loss weight to apply gradients.
        self._loss_weight = layer_param.loss_weight \
            if len(layer_param.loss_weight) > 0 else None
        # Optional mirror stage argument for memory optimization.
        if layer_param.HasField('mirror_stage'):
            self._arguments['mirror_stage'] = layer_param.mirror_stage

    @property
    def blobs(self):
        """Return the blobs."""
        return self._blobs

    @property
    def bottom(self):
        """Return the bottom names."""
        return self._bottom

    @property
    def loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    @property
    def name(self):
        """Return the layer name."""
        return self._name

    @property
    def top(self):
        """Return the top names."""
        return self._top

    def add_blob(self, value=None, filler=None, no_grad=False):
        """Add a blob into this layer."""
        # Set the name for reference explicitly.
        data_name = context.get_name_scope() + 'param:{}'.format(len(self._blobs))
        data, diff = TensorRef(data_name), TensorRef(data_name + '_grad')
        if filler is not None:
            data._register_as(**filler)
        else:
            # Register a constant filler by default.
            value = value if value else 0
            data.constant(value=value)
        # Append to the blobs.
        self._blobs.append({'data': data, 'diff': None if no_grad else diff})

    def from_proto(self, proto):
        """Deserialize from the proto.

        Parameters
        ----------
        proto : LayerParameter
            The ``LayerParameter`` protocol buffer.

        """
        for i in range(len(self._blobs)):
            if i < len(proto.blobs):
                blob_proto = proto.blobs[i]
                if len(blob_proto.data) > 0:
                    value = numpy.array(blob_proto.data, dtype='float32')
                elif len(blob_proto.double_data) > 0:
                    value = numpy.array(blob_proto.double_data, dtype='float64')
                else:
                    raise ValueError('Neither <data> or <double_data> in blob proto.')
                if len(blob_proto.shape.dim) > 0:
                    value = value.reshape([dim for dim in blob_proto.shape.dim])
                self._blobs[i]['data'].set_value(value)
                logging.info('Blob({}/param:{}) loaded, shape: {}, size: {}'
                             .format(self._name, i, value.shape, value.size))

    def setup(self, bottom):
        """Setup the layer."""
        self.arguments = dict(self.arguments, **self._arguments)
        bottom = bottom[0] if len(bottom) == 1 else bottom
        with eager_context.graph_mode():
            return self.__call__(bottom)

    def to_proto(self):
        """Serialize to the proto.

        Returns
        -------
        LayerParameter
            The ``LayerParameter`` protocol buffer.

        """
        proto = caffe_pb2.LayerParameter()
        proto.CopyFrom(self._proto)
        for blob in self._blobs:
            value = blob['data'].get_value()
            if str(value.dtype) == 'float32':
                blob_proto = caffe_pb2.BlobProto(
                    data=value.flatten(),
                    shape=caffe_pb2.BlobShape(dim=value.shape))
            elif str(value.dtype) == 'float64':
                blob_proto = caffe_pb2.BlobProto(
                    double_data=value.flatten(),
                    shape=caffe_pb2.BlobShape(dim=value.shape))
            else:
                raise ValueError('Either float32 or float64 blob is required.')
            proto.blobs.extend([blob_proto])
        return proto

    @staticmethod
    def get_filler(proto, filler_name):
        """Return the filler from proto."""
        if proto.HasField(filler_name):
            filler = getattr(proto, filler_name)
            return {
                'type': filler.type.lower(),
                'value': filler.value,
                'low': filler.min,
                'high': filler.max,
                'mean': filler.mean,
                'std': filler.std,
            }
        return None

    def __call__(self, bottom):
        """Define the forward pipeline."""
        raise NotImplementedError
