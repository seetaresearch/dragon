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


class VecMultOp(object):
    """How to custom a TemplateOp for Vector Multiplication.

    Examples
    --------
    >>> import dragon as dg
    >>> x1 = dg.Tensor('x1').Variable()
    >>> x2 = dg.Tensor('x2').Variable()
    >>> y = dg.ops.Template([x1, x2], module=__name__, op='VecMultOp', nout=1)
    >>> dx1 = dg.grad(y, x1)
    >>> dx2 = dg.grad(y, x2)
    >>> foo = dg.function(outputs=y)
    >>> dg.workspace.FeedTensor(x1, np.ones((5, 3), dtype=np.float32))
    >>> dg.workspace.FeedTensor(x2, np.ones((5, 3), dtype=np.float32) * 5.0)
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
        x1 = dg.workspace.FetchTensor(inputs[0])
        x2 = dg.workspace.FetchTensor(inputs[1])
        dg.workspace.FeedTensor(outputs[0], x1 * x2) # call numpy mult

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
        x1 = dg.workspace.FetchTensor(inputs[0])
        x2 = dg.workspace.FetchTensor(inputs[1])
        dy = dg.workspace.FetchTensor(inputs[-1])
        dx1 = dy * x2
        dx2 = dy * x1
        dg.workspace.FeedTensor(outputs[0], dx1)
        dg.workspace.FeedTensor(outputs[1], dx2)


if __name__ == '__main__':

    # def
    x1 = dg.Tensor('x1').Variable()
    x2 = dg.Tensor('x2').Variable()
    y = dg.ops.Template([x1, x2], module=__name__, op='VecMultOp', nout=1)
    dx1 = dg.grad(y, x1)
    dx2 = dg.grad(y, x2)
    foo = dg.function(outputs=y)

    # feed
    dg.workspace.FeedTensor(x1, np.ones((5, 3), dtype=np.float32))
    dg.workspace.FeedTensor(x2, np.ones((5, 3), dtype=np.float32) * 5.0)

    # run
    foo()

    # fetch
    print('y \n-------------- \n', y.get_value(), '\n')
    print('dx1 \n-------------- \n', dx1.get_value(), '\n')
    print('dx2 \n-------------- \n', dx2.get_value(), '\n')