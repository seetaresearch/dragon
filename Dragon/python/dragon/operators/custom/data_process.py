# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import print_function
import numpy as np
import dragon.core.workspace as ws
import dragon.ops as ops
import dragon.vm.theano as theano
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
    >>> import dragon.vm.theano as theano
    >>> y = Run([], module=__name__, op='DataProcessOp', nout=1)
    >>> foo = theano.function(outputs=y)
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
        ws.FeedTensor(outputs[0], self._queue.get())


if __name__ == '__main__':

    # def
    y = ops.Run([], module=__name__, op='DataProcessOp', nout=1)
    foo = theano.function(outputs=y)

    # run
    foo()

    # fetch
    print(y.get_value())