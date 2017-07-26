# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.caffe as caffe
import dragon.core.workspace as ws
from minibatch import DataBatch

class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        kwargs = eval(self.param_str)
        self._data_batch = DataBatch(**kwargs)

    def forward(self, bottom, top):
        blobs = self._data_batch.blobs
        for idx, blob in enumerate(blobs):
            ws.FeedTensor(top[idx], blob)