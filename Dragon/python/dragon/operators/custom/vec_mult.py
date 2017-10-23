# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import print_function
import numpy as np
import dragon.core.workspace as ws
import dragon.ops as ops
from dragon.core.tensor import Tensor
import dragon.vm.theano.tensor as T
import dragon.vm.theano as theano

class VecMultOp(object):
    """How to custom a TemplateOp for Vector Multiplication.

    Examples
    --------
    >>> import dragon.ops
    >>> import dragon.core.workspace as ws
    >>> import dragon.vm.theano as theano
    >>> x1 = Tensor('x1').Variable()
    >>> x2 = Tensor('x2').Variable()
    >>> y = ops.Template([x1, x2], module=__name__, op='VecMultOp', nout=1)
    >>> dx1 = T.grad(y, x1)
    >>> dx2 = T.grad(y, x2)
    >>> foo = theano.function(outputs=y)
    >>> ws.FeedTensor(x1, np.ones((5, 3), dtype=np.float32))
    >>> ws.FeedTensor(x2, np.ones((5, 3), dtype=np.float32) * 5.0)
    >>> foo()

    >>> print(y.get_value())
    >>> [[ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]]

    >>> print(dx1.get_value())
    >>> [[ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]
         [ 5.  5.  5.]]

    >>> print(dx2.get_value())
    >>>  [[ 1.  1.  1.]
          [ 1.  1.  1.]
          [ 1.  1.  1.]
          [ 1.  1.  1.]
          [ 1.  1.  1.]]

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
        pass


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
        x1 = ws.FetchTensor(inputs[0])
        x2 = ws.FetchTensor(inputs[1])
        ws.FeedTensor(outputs[0], x1 * x2) # call numpy mult

    def grad(self, inputs, outputs):
        """Gradient method, i.e., backward pass.

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
    y = ops.Template([x1, x2], module=__name__, op='VecMultOp', nout=1)
    dx1 = T.grad(y, x1)
    dx2 = T.grad(y, x2)
    foo = theano.function(outputs=y)

    # feed
    ws.FeedTensor(x1, np.ones((5, 3), dtype=np.float32))
    ws.FeedTensor(x2, np.ones((5, 3), dtype=np.float32) * 5.0)

    # run
    foo()

    # fetch
    print('y \n-------------- \n', y.get_value(), '\n')
    print('dx1 \n-------------- \n', dx1.get_value(), '\n')
    print('dx2 \n-------------- \n', dx2.get_value(), '\n')