# --------------------------------------------------------
# VecMult of TemplateOp for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import dragon.core.workspace as ws
import dragon.ops as ops
from dragon.core.tensor import Tensor
import dragon.vm.theano.tensor as T
import dragon.vm.theano as theano
from dragon.config import logger

""" How to custom a TemplateOp in Dragon """

class VecMult(object):
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
        pass

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
        x1 = ws.FetchTensor(inputs[0])
        x2 = ws.FetchTensor(inputs[1])
        ws.FeedTensor(outputs[0], x1 * x2) # call numpy mult

    def grad(self, inputs, outputs):
        """
        Grad implement(i.e. backward-pass).

            Parameters
            ----------
            inputs  : sequence of strs
                Indicating the operator's inputs + in-grads.
                    The first N strs in sequence is inputs.
                    The N + 1 ... 2N strs in sequence is in-grads.

            outputs : sequence of strs
                Indicating the operator's out-grads

            Returns
            -------
            None

        """
        x1 = ws.FetchTensor(inputs[0])
        x2 = ws.FetchTensor(inputs[1])
        dy = ws.FetchTensor(inputs[-1])
        dx1 = dy * x2
        dx2 = dy * x1
        ws.FeedTensor(outputs[0], dx1)
        ws.FeedTensor(outputs[1], dx2)

if __name__ == '__main__':

    # def
    x1 = Tensor('x1').Variable()
    x2 = Tensor('x2').Variable()
    y = ops.Template([x1, x2], module=__name__, op='VecMult', nout=1)
    dx1 = T.grad(y, x1)
    dx2 = T.grad(y, x2)
    foo = theano.function(outputs=y)

    # feed
    ws.FeedTensor(x1, np.ones((5, 3)))
    ws.FeedTensor(x2, np.ones((5, 3)) * 5.0)

    # run
    foo()

    # fetch
    logger.info('y \n-------------- \n', y.get_value(), '\n')
    logger.info('dx1 \n-------------- \n', dx1.get_value(), '\n')
    logger.info('dx2 \n-------------- \n', dx2.get_value(), '\n')

