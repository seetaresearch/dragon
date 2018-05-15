# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import dragon.ops as ops

from ..layer import Layer


class DataLayer(Layer):
    """
    The implementation of ``DataLayer``.

    Different from ``Caffe``, we force to use `LMDB`_ backend.

    Parameters
    ----------
    source : str
        The path of database. Refer `DataParameter.source`_.
    prefetch: int
        The prefetch count. Refer `DataParameter.prefetch`_.
    batch_size : int
        The size of a mini-batch. Refer `DataParameter.batch_size`_.
    phase : caffe_pb2.Phase
        The phase of layer. Refer `LayerParameter.phase`_.
    scale : float
        The scaling factor. Refer `TransformationParameter.scale`_.
    mirrow : boolean
        Whether to randomly mirror. Refer `TransformationParameter.mirror`_.
    crop_size : int
        The crop size. Refer `TransformationParameter.crop_size`_.
    mean_value : list of float
        The mean of each channel. Refer `TransformationParameter.mean_value`_.
    force_color : boolean
        Force to have 3 channels. Refer `TransformationParameter.force_color`_.
    color_augmentation : boolean
        Whether to distort colors. Extension of `TransformationParameter`_.
    padding : int
        The padding size. Extension of `TransformationParameter`_.
    min_random_scale : float
        The min scale of the images. Extension of `TransformationParameter`_.
    max_random_scale : float
        The max scale of the images. Extension of `TransformationParameter`_.

    """
    def __init__(self, LayerParameter):
        super(DataLayer, self).__init__(LayerParameter)

        param = LayerParameter.data_param
        transform_param = LayerParameter.transform_param
        parallel_param = LayerParameter.parallel_param

        self._param = {'source': param.source,
                       'prefetch': param.prefetch,
                       'batch_size': param.batch_size,
                       'phase': {0: 'TRAIN', 1: 'TEST'}[int(LayerParameter.phase)],
                       'scale': transform_param.scale,
                       'mirror': transform_param.mirror,
                       'crop_size': transform_param.crop_size,
                       'mean_values': [float(element) for element in transform_param.mean_value],
                       'force_color': transform_param.force_color,
                       'color_augmentation': transform_param.color_augmentation,
                       'padding': transform_param.padding,
                       'min_random_scale': transform_param.min_random_scale,
                       'max_random_scale': transform_param.max_random_scale,
                       'shuffle': parallel_param.shuffle,
                       'node_step': parallel_param.node_step,
                       'partition': parallel_param.partition}

    def Setup(self, bottom):
        super(DataLayer, self).Setup(bottom)
        return ops.LMDBData(**self._param)


class MemoryDataLayer(Layer):
    """The implementation of ``MemoryDataLayer``.

    We extend it with ``FP16`` and ``NHWC => NCHW``.

    Parameters
    ----------
    dtype : caffe_pb2.MemoryDataParameter.DataType
        The dest data type. ``FLOAT32`` or ``FLOAT16``.
    mean_value : list of float
        The mean of each channel. Refer `TransformationParameter.mean_value`_.

    """
    def __init__(self, LayerParameter):
        super(MemoryDataLayer, self).__init__(LayerParameter)
        param = LayerParameter.memory_data_param
        transform_param = LayerParameter.transform_param
        self._param = {'dtype': {0: 'FLOAT32', 1: 'FLOAT16'}[param.dtype]}
        if len(transform_param.mean_value) > 0:
            self._param['mean_values'] = \
                [float(element) for element in transform_param.mean_value]

    def Setup(self, bottom):
        super(MemoryDataLayer, self).Setup(bottom)
        return ops.ImageData(bottom[0], **self._param)