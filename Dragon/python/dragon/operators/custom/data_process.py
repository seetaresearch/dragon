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

import numpy as np
import dragon as dg
from multiprocessing import Process, Queue


class Fetcher(Process):
    def __init__(self, queue):
        super(Fetcher, self).__init__()
        self._queue = queue
        self.daemon = True

        def cleanup():
            print('Terminating Fetcher......')
            self.terminate()
            self.join()

        import atexit
        atexit.register(cleanup)

    def run(self):
        while True:
            self._queue.put(np.ones((5, 10)))


class DataProcessOp(object):
    """How to custom a RunOp for data processing.

    Examples
    --------
    >>> import dragon as dg
    >>> y = dg.ops.Run([], module=__name__, op='DataProcessOp', nout=1)
    >>> foo = dg.function(outputs=y)
    >>> foo()
    >>> print(y.get_value())
    >>> [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
         [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]

    """
    def setup(self, inputs, outputs):
        """Setup for params or options.

        Parameters
        ----------
        inputs : list of str
            Indicating the name of input tensors.
        outputs : list of str
            Indicating the name of output tensors.

        Returns
        -------
        None

        """
        self._queue = Queue(100)
        self._fetcher = Fetcher(self._queue)
        self._fetcher.start()


    def run(self, inputs, outputs):
        """Run method, i.e., forward pass.

        Parameters
        ----------
        inputs : list of str
            Indicating the name of input tensors.
        outputs : list of str
            Indicating the name of output tensors.

        Returns
        -------
        None

        """
        dg.workspace.FeedTensor(outputs[0], self._queue.get())


if __name__ == '__main__':
    # Def
    y = dg.ops.Run([], module=__name__, op='DataProcessOp', nout=1)
    foo = dg.function(outputs=y)

    # Run
    foo()

    # Fetch
    print(y.get_value())