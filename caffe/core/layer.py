# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The base layer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph import context
from dragon.core.framework.tensor import Tensor
from dragon.vm.caffe.core.proto import caffe_pb2


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
        self._bottom_names = [name for name in layer_param.bottom]
        self._top_names = [name for name in layer_param.top]
        self._blobs = []
        self._call_layer = None

    @property
    def blobs(self):
        """Return the blobs."""
        return self._blobs

    @property
    def bottom(self):
        """Return the bottom names."""
        return self._bottom_names

    @property
    def name(self):
        """Return the layer name."""
        return self._proto.name

    @property
    def top(self):
        """Return the top names."""
        return self._top_names

    def add_blob(self, shape, filler, requires_grad=True):
        """Add a blob into this layer."""
        data = Tensor(shape, name='blob%d' % (len(self._blobs) + 1))
        if filler.type == 'constant':
            data.fill(filler.value)
        elif filler.type == 'gaussian':
            data.normal(filler.mean, filler.std)
        elif filler.type == 'uniform':
            data.uniform(filler.min, filler.max)
        elif filler.type == 'xavier':
            norm_modes = {0: 'fan_in', 1: 'fan_out', 2: 'fan_avg'}
            data.glorot_uniform(norm_modes[filler.variance_norm])
        elif filler.type == 'msra':
            norm_modes = {0: 'fan_in', 1: 'fan_out', 2: 'fan_avg'}
            data.glorot_normal(norm_modes[filler.variance_norm])
        else:
            raise ValueError('Unknown filler type: ' + filler.type)
        data.requires_grad = requires_grad
        self._blobs.append({'data': data, 'diff': None})

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
                self._blobs[i]['data']._impl.FromNumpy(value, False)

    def setup(self, bottom):
        """Setup the layer."""
        bottom = bottom[0] if len(bottom) == 1 else bottom
        with context.graph_mode():
            call_layer = self._call_layer or self
            return call_layer.__call__(bottom)

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
            value = blob['data'].numpy()
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

    def __call__(self, bottom):
        """Define the forward pipeline."""
        raise NotImplementedError
