# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from layer import Layer
import dragon.ops as ops

class DataLayer(Layer):
    def __init__(self, LayerParameter):
        super(DataLayer, self).__init__(LayerParameter)

        param = LayerParameter.data_param
        transformer_param = LayerParameter.transform_param
        parallel_param = LayerParameter.parallel_param

        self._param = {'mean_value': [float(element) for element in transformer_param.mean_value],
                       'mean_file': transformer_param.mean_file,
                       'source': param.source,
                       'prefetch': param.prefetch,
                       'batch_size': param.batch_size,
                       'shuffle': parallel_param.shuffle,
                       'node_step': parallel_param.node_step,
                       'partition': parallel_param.partition,
                       'force_gray': transformer_param.force_gray,
                       'crop_size': transformer_param.crop_size,
                       'mirror': transformer_param.mirror,
                       'color_augmentation': transformer_param.color_augmentation,
                       'min_random_scale': transformer_param.min_random_scale,
                       'max_random_scale': transformer_param.max_random_scale,
                       'scale': transformer_param.scale,
                       'phase': LayerParameter.phase}

    def Setup(self, bottom):
        super(DataLayer, self).Setup(bottom)
        return ops.LMDBData(**self._param)


class MemoryDataLayer(Layer):
    def __init__(self, LayerParameter):
        super(MemoryDataLayer, self).__init__(LayerParameter)
        param = LayerParameter.memory_data_param
        import numpy as np
        self._param = {'dtype': {0: np.float32, 1: np.float16}[param.dtype]}

    def Setup(self, bottom):
        super(MemoryDataLayer, self).Setup(bottom)
        return ops.MemoryData(bottom[0], **self._param)