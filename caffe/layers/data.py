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

"""The implementation of the data layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import workspace
from dragon.core.io.kpl_record import KPLRecordDataset
from dragon.core.ops import array_ops
from dragon.core.ops import framework_ops
from dragon.utils import vision
from dragon.vm.caffe.layer import Layer


class _DataPlugin(object):
    """Embedded plugin for **Data** layer."""

    def setup(self, inputs, outputs):
        kwargs = eval(self.kwargs_str)
        self.iterator = vision.DataIterator(
            dataset=KPLRecordDataset, **kwargs)

    def forward(self, inputs, outputs):
        blobs = self.iterator.next()
        current_ws = workspace.get_workspace()
        for i, blob in enumerate(blobs):
            current_ws.feed_tensor(outputs[i], blob)


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
        shuffle: true
        num_chunks: 0
        prefetch: 5
      }
      transform_param {
        mirror: true
        random_crop_size: 224
        augment_color: true
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
        resize: 256
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
        param = layer_param.data_param
        transform_param = layer_param.transform_param
        self.data_args = {
            'source': param.source,
            'prefetch': param.prefetch,
            'shuffle': param.shuffle,
            'num_chunks': param.num_chunks,
            'batch_size': param.batch_size,
            'phase': {0: 'TRAIN', 1: 'TEST'}[int(layer_param.phase)],
            'resize': transform_param.resize,
            'padding': transform_param.padding,
            'crop_size': transform_param.crop_size,
            'random_crop_size': transform_param.random_crop_size,
            'augment_color': transform_param.augment_color,
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
        data = array_ops.channel_normalize(data, **self.norm_args)
        return data, label
