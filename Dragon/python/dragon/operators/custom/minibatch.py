# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.core.workspace as ws
from dragon.io.data_batch import DataBatch

class MiniBatchOp(object):

    def setup(self, inputs, outputs):
        kwargs = eval(self.param_str)
        self._data_batch = DataBatch(**kwargs)

    def run(self, inputs, outputs):
        blobs = self._data_batch.get()
        for idx, blob in enumerate(blobs):
            ws.FeedTensor(outputs[idx], blob)