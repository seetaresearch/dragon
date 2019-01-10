=============
:mod:`Tensor`
=============

.. toctree::
   :hidden:


Basic
-----

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`grad`_                             Compute the gradients for variables with respect to the cost.
`disconnected_grad`_                Return the input with truncated gradient flow.
==============================      =======================================================================

Variable
--------

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`scalar`_                           Return a scalar variable.
`iscalar`_                          Return a int32 scalar variable.
==============================      =======================================================================

Initializer
-----------

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`constant`_                         Initialize a tensor with constant value.
`zeros`_                            Initialize a tensor with zeros.
`zeros_like`_                       Initialize a tensor with zeros, refer the shape of another tensor.
`ones`_                             Initialize a tensor with ones.
`ones_like`_                        Initialize a tensor with ones, refer the shape of another tensor.
==============================      =======================================================================

Operator
--------

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`cast`_                             Cast input to the tensor of specific data type.
`dot`_                              Calculate A dot B.
`transpose`_                        Transpose the input according to the given permutations.
`sum`_                              Compute the sum along the given axis.
`mean`_                             Compute the mean along the given axis.
`prod`_                             Compute the product along the given axis.
`argmax`_                           Compute the indices of maximum elements along the given axis.
`argmin`_                           Compute the indices of minimum elements along the given axis.
`square`_                           Calculate the square of input.
`sqrt`_                             Calculate the sqrt of input.
`pow`_                              Calculate the power of input.
`exp`_                              Calculate the exponential of input.
`log`_                              Calculate the logarithm of input.
`clip`_                             Clip the input to be between min and max.
`join`_                             Convenience function to concatenate along the given axis.
`stack`_                            Stack the inputs along the given axis.
`concatenate`_                      Concatenate the inputs along the given axis.
`reshape`_                          Reshape the dimensions of input.
`flatten`_                          Flatten the input by keeping the specific dimensions.
`repeat`_                           Repeat the input along the given axis.
`tile`_                             Tile the input according to the given multiples.
`arange`_                           Return a vector of elements by arange.
`cumsum`_                           Compute the cumulative sum along the given axis.
`cumprod`_                          Compute the cumulative product along the given axis.
`to_one_hot`_                       Generate a matrix where each row corresponds to the one hot encoding.
==============================      =======================================================================

NNet
----

==============================      ===============================================================================
List                                Brief
==============================      ===============================================================================
`batch_normalization`_              Batch Normalization, introduced by `[Ioffe & Szegedy, 2015]`_
`relu`_                             Rectified Linear Unit function, introduces by `[Nair & Hinton, 2010]`_.
`softmax`_                          Softmax function.
`categorical_crossentropy`_         Compute the categorical cross-entropy between input and target distribution.
`sigmoid`_                          Sigmoid function.
`tanh`_                             TanH function.
`binary_crossentropy`_              Compute the binary cross-entropy between input and target distribution.
==============================      ===============================================================================


API Reference
-------------

.. automodule:: dragon.vm.theano.gradient
    :members:

.. automodule:: dragon.vm.theano.tensor.basic
    :members:

.. automodule:: dragon.vm.theano.tensor.extra_ops
    :members:

.. automodule:: dragon.vm.theano.tensor.nnet
    :members:

.. _grad: #dragon.vm.theano.gradient.grad
.. _disconnected_grad: #dragon.vm.theano.gradient.disconnected_grad
.. _scalar: #dragon.vm.theano.tensor.basic.scalar
.. _iscalar: #dragon.vm.theano.tensor.basic.iscalar
.. _constant: #dragon.vm.theano.tensor.basic.constant
.. _zeros: #dragon.vm.theano.tensor.basic.zeros
.. _zeros_like: #dragon.vm.theano.tensor.basic.zeros_like
.. _ones: #dragon.vm.theano.tensor.basic.ones
.. _ones_like: #dragon.vm.theano.tensor.basic.ones_like
.. _cast: #dragon.vm.theano.tensor.basic.cast
.. _dot: #dragon.vm.theano.tensor.basic.dot
.. _transpose: #dragon.vm.theano.tensor.basic.transpose
.. _sum: #dragon.vm.theano.tensor.basic.sum
.. _mean: #dragon.vm.theano.tensor.basic.mean
.. _prod: #dragon.vm.theano.tensor.basic.prod
.. _argmax: #dragon.vm.theano.tensor.basic.argmax
.. _argmin: #dragon.vm.theano.tensor.basic.argmin
.. _square: #dragon.vm.theano.tensor.basic.square
.. _sqrt: #dragon.vm.theano.tensor.basic.sqrt
.. _pow: #dragon.vm.theano.tensor.basic.pow
.. _exp: #dragon.vm.theano.tensor.basic.exp
.. _log: #dragon.vm.theano.tensor.basic.log
.. _clip: #dragon.vm.theano.tensor.basic.clip
.. _join: #dragon.vm.theano.tensor.basic.join
.. _stack: #dragon.vm.theano.tensor.basic.stack
.. _concatenate: #dragon.vm.theano.tensor.basic.concatenate
.. _reshape: #dragon.vm.theano.tensor.basic.reshape
.. _flatten: #dragon.vm.theano.tensor.basic.flatten
.. _repeat: #dragon.vm.theano.tensor.basic.repeat
.. _tile: #dragon.vm.theano.tensor.basic.tile
.. _arange: #dragon.vm.theano.tensor.basic.arange
.. _cumsum: #dragon.vm.theano.tensor.extra_ops.cumsum
.. _cumprod: #dragon.vm.theano.tensor.extra_ops.cumprod
.. _to_one_hot: #dragon.vm.theano.tensor.extra_ops.to_one_hot
.. _batch_normalization: #dragon.vm.theano.tensor.nnet.batch_normalization
.. _relu: #dragon.vm.theano.tensor.nnet.relu
.. _softmax: #dragon.vm.theano.tensor.nnet.softmax
.. _categorical_crossentropy: #dragon.vm.theano.tensor.nnet.categorical_crossentropy
.. _sigmoid: #dragon.vm.theano.tensor.nnet.sigmoid
.. _tanh: #dragon.vm.theano.tensor.nnet.tanh
.. _binary_crossentropy: #dragon.vm.theano.tensor.nnet.binary_crossentropy

.. _[Ioffe & Szegedy, 2015]: https://arxiv.org/abs/1502.03167
.. _[Nair & Hinton, 2010]: http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf
