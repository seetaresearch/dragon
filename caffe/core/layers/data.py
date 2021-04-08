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
"""Data layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import workspace
from dragon.core.io.kpl_record import KPLRecordDataset
from dragon.core.ops import array_ops
from dragon.core.ops import framework_ops
from dragon.utils import vision
from dragon.vm.caffe.core.layer import Layer


class _DataPlugin(object):
    """Embedded plugin for data layer."""

    def setup(self, inputs, outputs):
        kwargs = eval(self.kwargs_str)
        default_ws = workspace.get_workspace()
        self.outputs = [default_ws.get_tensor(output) for output in outputs]
        self.iterator = vision.DataIterator(dataset=KPLRecordDataset, **kwargs)

    def forward(self, inputs, outputs):
        blobs = self.iterator.next()
        for i, blob in enumerate(blobs):
            self.outputs[i].FromNumpy(blob)


class Data(Layer):
    r"""Load batch of data for image classification.

    Examples:

    ```python
    layer {
      type: "Data"
      top: "data"
      top: "label"
      include {
        phase: TRAIN
      }
      data_param {
        source: "/data/train"
        batch_size: 128
        prefetch: 4
      }
      image_data_param {
        shuffle: true
      }
      transform_param {
        mirror: true
        crop_size: 224
        mean_value: 104.00698793
        mean_value: 116.66876762
        mean_value: 122.67891434
      }
    }
    layer {
      type: "Data"
      top: "data"
      top: "label"
      include {
        phase: TEST
      }
      data_param {
        source: "/data/val"
        batch_size: 64
      }
      transform_param {
        crop_size: 224
        mean_value: 104.00698793
        mean_value: 116.66876762
        mean_value: 122.67891434
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Data, self).__init__(layer_param)
        data_param = layer_param.data_param
        image_data_param = layer_param.image_data_param
        transform_param = layer_param.transform_param
        self.data_args = {
            'source': data_param.source,
            'batch_size': data_param.batch_size,
            'prefetch': data_param.prefetch,
            'shuffle': image_data_param.shuffle,
            'phase': {0: 'TRAIN', 1: 'TEST'}[int(layer_param.phase)],
            'crop_size': transform_param.crop_size,
            'mirror': transform_param.mirror,
        }
        self.norm_args = {
            'axis': 1,
            'perm': (0, 3, 1, 2),
            'mean': [e for e in transform_param.mean_value],
            'std': [1. / transform_param.scale] * len(transform_param.mean_value),
            'dtype': 'float32',
        }

    def __call__(self, bottom):
        args = {
            'module_name': __name__,
            'class_name': '_DataPlugin',
            'kwargs_str': str(self.data_args),
            'num_outputs': 2,
        }
        data, label = framework_ops.python_plugin([], **args)
        data._shape = (self.data_args['batch_size'],
                       None, None, len(self.norm_args['mean']))
        label._shape = (self.data_args['batch_size'], None)
        data = array_ops.channel_normalize(data, **self.norm_args)
        return data, label
