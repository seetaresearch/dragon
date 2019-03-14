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
#      <https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/workspace.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
import dragon as dg


class Workspace(object):
    def __init__(self):
        self.name = 'onnx/' + str(uuid.uuid4())

    def __getattr__(self, attr):
        def f(*args, **kwargs):
            with dg.ws_scope(self.name, ):
                return getattr(dg.workspace, attr)(*args, **kwargs)
        return f

    def __del__(self):
        self.ResetWorkspace(self.name)