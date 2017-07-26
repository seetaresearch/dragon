# --------------------------------------------------------
# DataProcess of RunOp for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import dragon.core.workspace as ws
import dragon.ops as ops
import dragon.vm.theano as theano
from multiprocessing import Process, Queue

""" How to custom a RunOp in Dragon """

class Fetcher(Process):
    def __init__(self, queue):
        """
        Init Process.

            Parameters
            ----------
            queue : multiprocessing.Queue

            Returns
            -------
            None

        """
        super(Fetcher, self).__init__()
        self._queue = queue
        self.daemon = True

        def cleanup():
            print 'Terminating Fetcher......'
            self.terminate()
            self.join()

        import atexit
        atexit.register(cleanup)

    def run(self):
        """
        Run Process.

            Parameters
            ----------
            None

            Returns
            -------
            None

        """
        while True:
            self._queue.put(np.ones((5, 10)))

class DataProcess(object):
    def setup(self, inputs, outputs):
        """
        Setup for params or options.

            Parameters
            ----------
            inputs  : sequence of strs
                Indicating the operator's inputs
            outputs : sequence of strs
                Indicating the operator's outputs

            Returns
            -------
            None

        """
        self._queue = Queue(100)
        self._fetcher = Fetcher(self._queue)
        self._fetcher.start()

    def run(self, inputs, outputs):
        """
        Run implement(i.e. forward-pass).

            Parameters
            ----------
            inputs  : sequence of strs
                Indicating the operator's inputs
            outputs : sequence of strs
                Indicating the operator's outputs

            Returns
            -------
            None

        """
        ws.FeedTensor(outputs[0], self._queue.get())

if __name__ == '__main__':

    # def
    y = ops.Run([], module=__name__, op='DataProcess', nout=1)
    foo = theano.function(outputs=y)

    # run
    foo()

    # fetch
    print 'y \n-------------- \n', y.get_value(), '\n'