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

import time


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=False, every_n=-1, name=''):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if every_n > 0 and self.calls % every_n == 0:
            print('[{}]: total = {:.5f}s, average = {:.5f}s'.format(
                name, self.total_time, self.total_time / self.calls * every_n))
        if average:
            return self.average_time
        else:
            return self.diff