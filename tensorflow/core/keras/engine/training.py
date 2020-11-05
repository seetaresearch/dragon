# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.keras.engine import network


class Model(network.Network):
    """Compose a group of layers with training and inference features."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
        """
        super(Model, self).__init__(*args, **kwargs)
