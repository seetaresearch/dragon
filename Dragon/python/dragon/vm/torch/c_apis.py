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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Size(tuple):
    def __init__(self, sizes):
        super(Size, self).__init__()

    def __setitem__(self, key, value):
        raise TypeError("'torch.Size' object does not support item assignment")

    def __repr__(self):
        return 'torch.Size([{}])'.format(', '.join([str(s) for s in self]))