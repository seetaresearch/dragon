vm.torch
========

.. only:: html

  Classes
  #######

  `class device <torch/device.html>`_
  : Represent the device spec.

  `class dtype <torch/device.html>`_
  : The basic data type.

  `class enable_grad <torch/enable_grad.html>`_
  : Context-manager to enable gradient calculation.

  `class no_grad <torch/no_grad.html>`_
  : Context-manager to disable gradient calculation.

  `class set_grad_enabled <torch/set_grad_enabled.html>`_
  : Context-manager to set gradient calculation on or off.

  `class Size <torch/Size.html>`_
  : Represent the a sequence of dimensions.

  `class Tensor <torch/Tensor_.html>`_
  : A multi-dimensional array containing elements of a single data type.

  Functions
  #########

  `abs(...) <torch/abs.html>`_
  : Compute the absolute value of input.

  `add(...) <torch/add.html>`_
  : Compute the element-wise addition.

  `arange(...) <torch/arange.html>`_
  : Return a tensor of evenly spaced values within a interval.

  `argmax(...) <torch/argmax.html>`_
  : Return the index of maximum elements along the given dimension.

  `argmin(...) <torch/argmin.html>`_
  : Return the index of minimum elements along the given dimension.

  `argsort(...) <torch/argsort.html>`_
  : Return the index of sorted elements along the given dimension.

  `axpby(...) <torch/axpby.html>`_
  : Compute the element-wise addition from input to output.

  `bitwise_not(...) <torch/bitwise_not.html>`_
  : Compute the element-wise NOT bitwise operation.

  `bitwise_xor(...) <torch/bitwise_xor.html>`_
  : Compute the element-wise XOR bitwise operation.

  `cat(...) <torch/cat.html>`_
  : Concatenate the inputs along the given dimension.

  `ceil(...) <torch/ceil.html>`_
  : Compute the smallest integer not less than input.

  `channel_affine(...) <torch/channel_affine.html>`_
  : Apply affine transformation along the channels.

  `channel_normalize(...) <torch/channel_normalize.html>`_
  : Normalize channels with mean and standard deviation.

  `channel_shuffle(...) <torch/channel_shuffle.html>`_
  : Shuffle channels between a given number of groups.
  `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

  `chunk(...) <torch/chunk.html>`_
  : Split input into a specific number of chunks.

  `clamp(...) <torch/clamp.html>`_
  : Compute the clipped input according to the given bounds.

  `cos(...) <torch/cos.html>`_
  : Compute the cos of input.

  `cumsum(...) <torch/cumsum.html>`_
  : Compute the cumulative sum of elements along the given dimension.

  `div(...) <torch/div.html>`_
  : Compute the element-wise division.

  `empty(...) <torch/empty.html>`_
  : Return a tensor filled with uninitialized data.

  `eq(...) <torch/eq.html>`_
  : Compute the element-wise equal comparison.

  `exp(...) <torch/exp.html>`_
  : Compute the exponential of input.

  `eye(...) <torch/eye.html>`_
  : Return a tensor constructed as the identity matrix.

  `flatten(...) <torch/flatten.html>`_
  : Return a tensor with dimensions flattened.

  `floor(...) <torch/floor.html>`_
  : Compute the largest integer not greater than input.

  `from_numpy(...) <torch/from_numpy.html>`_
  : Create a tensor from the given numpy array.

  `full(...) <torch/full.html>`_
  : Return a tensor filled with a scalar.

  `full_like(...) <torch/full_like.html>`_
  : Return a tensor filled with a scalar with size as input.

  `ge(...) <torch/ge.html>`_
  : Compute the element-wise greater-equal comparison.

  `gt(...) <torch/gt.html>`_
  : Compute the element-wise greater comparison.

  `index_select(...) <torch/index_select.html>`_
  : Select the elements along the given dim using index.

  `isinf(...) <torch/isinf.html>`_
  : Check if the elements of input are infinite.

  `isnan(...) <torch/isnan.html>`_
  : Check if the elements of input are NaN.

  `le(...) <torch/le.html>`_
  : Compute the element-wise less-equal comparison.

  `linspace(...) <torch/linspace.html>`_
  : Generate evenly spaced values within intervals along the given axis.

  `log(...) <torch/log.html>`_
  : Compute the natural logarithm of input.

  `logsumexp(...) <torch/logsumexp.html>`_
  : Apply the composite of log, sum, and exp to input.

  `lt(...) <torch/lt.html>`_
  : Compute the element-wise less comparison.

  `masked_select(...) <torch/logsumexp.html>`_
  : Select the input elements where mask is 1.

  `max(...) <torch/max.html>`_
  : Compute the max value of elements along the given dimension.

  `maximum(...) <torch/maximum.html>`_
  : Compute the maximum value of inputs.

  `mean(...) <torch/mean.html>`_
  : Compute the mean value of elements along the given dimension.

  `min(...) <torch/min.html>`_
  : Compute the min value of elements along the given dimension.

  `minimum(...) <torch/minimum.html>`_
  : Compute the minimum value of inputs.

  `mm(...) <torch/mm.html>`_
  : Perform a matrix multiplication.

  `mul(...) <torch/mul.html>`_
  : Compute the element-wise multiplication.

  `multinomial(...) <torch/multinomial.html>`_
  : Return a tensor with index sampled from multinomial distribution.

  `narrow(...) <torch/narrow.html>`_
  : Return a new tensor that is a narrowed version of input tensor.

  `ne(...) <torch/ne.html>`_
  : Compute the element-wise not-equal comparison.

  `neg(...) <torch/neg.html>`_
  : Compute the element-wise negative.

  `nonzero(...) <torch/nonzero.html>`_
  : Return the index of non-zero elements.

  `ones(...) <torch/ones.html>`_
  : Return a tensor filled with ones.

  `ones_like(...) <torch/ones_like.html>`_
  : Return a tensor of ones with shape as the other.

  `one_hot(...) <torch/one_hot.html>`_
  : Return the one-hot representation for input.

  `permute(...) <torch/permute.html>`_
  : Return a new tensor with the specific order of dimensions.

  `pow(...) <torch/pow.html>`_
  : Compute the power of input.

  `rand(...) <torch/rand.html>`_
  : Return a tensor from the uniform distribution of U(0, 1).

  `randn(...) <torch/randn.html>`_
  : Return a tensor from the normal distribution of N(0, 1).

  `randperm(...) <torch/randperm.html>`_
  : Return a tensor with value in the permuted range.

  `reciprocal(...) <torch/reciprocal.html>`_
  : Compute the reciprocal of input.

  `repeat(...) <torch/repeat.html>`_
  : Repeat elements along the specified dimensions.

  `reshape(...) <torch/reshape.html>`_
  : Change the shape of input.

  `round(...) <torch/round.html>`_
  : Compute the nearest integer of input.

  `rsqrt(...) <torch/rsqrt.html>`_
  : Compute the reciprocal square root of input.

  `sign(...) <torch/sign.html>`_
  : Compute the sign indication of input.

  `sin(...) <torch/sin.html>`_
  : Compute the sin of input.

  `sort(...) <torch/sort.html>`_
  : Return the sorted elements along the given dimension.

  `split(...) <torch/split.html>`_
  : Split input into chunks along the given dimension.

  `sqrt(...) <torch/sqrt.html>`_
  : Compute the square root of input.

  `squeeze(...) <torch/squeeze.html>`_
  : Remove the dimensions of input with size 1.

  `stack(...) <torch/stack.html>`_
  : Stack the inputs along the given dimension.

  `sub(...) <torch/sub.html>`_
  : Compute the element-wise subtraction.

  `sum(...) <torch/sum.html>`_
  : Compute the sum value of elements along the given dimension.

  `tensor(...) <torch/tensor.html>`_
  : Create a tensor initializing the content from data.

  `topk(...) <torch/topk.html>`_
  : Return the top-K largest or smallest elements along the given dimension.

  `transpose(...) <torch/transpose.html>`_
  : Return a new tensor with two dimensions swapped.

  `unique(...) <torch/unique.html>`_
  : Return the unique elements of input.

  `unsqueeze(...) <torch/unsqueeze.html>`_
  : Expand the dimensions of input with size 1.

  `where(...) <torch/where.html>`_
  : Select the elements from two branches under the condition.

  `zeros(...) <torch/zeros.html>`_
  : Return a tensor filled with zeros.

  `zeros_like(...) <torch/zeros_like.html>`_
  : Return a tensor of zeros with shape as the other.

.. toctree::
  :hidden:

  torch/Size
  torch/Tensor_
  torch/abs
  torch/add
  torch/arange
  torch/argmax
  torch/argmin
  torch/argsort
  torch/axpby
  torch/bitwise_not
  torch/bitwise_xor
  torch/cat
  torch/ceil
  torch/channel_affine
  torch/channel_normalize
  torch/channel_shuffle
  torch/chunk
  torch/clamp
  torch/cos
  torch/cumsum
  torch/device
  torch/div
  torch/dtype
  torch/empty
  torch/enable_grad
  torch/eq
  torch/exp
  torch/eye
  torch/flatten
  torch/floor
  torch/from_numpy
  torch/full
  torch/full_like
  torch/ge
  torch/gt
  torch/index_select
  torch/isinf
  torch/isnan
  torch/le
  torch/linspace
  torch/log
  torch/logsumexp
  torch/lt
  torch/masked_select
  torch/max
  torch/maximum
  torch/mean
  torch/min
  torch/minimum
  torch/mm
  torch/mul
  torch/multinomial
  torch/narrow
  torch/ne
  torch/neg
  torch/no_grad
  torch/nonzero
  torch/ones
  torch/ones_like
  torch/one_hot
  torch/permute
  torch/pow
  torch/rand
  torch/randn
  torch/randperm
  torch/reciprocal
  torch/repeat
  torch/reshape
  torch/round
  torch/rsqrt
  torch/set_grad_enabled
  torch/sign
  torch/sin
  torch/sort
  torch/split
  torch/sqrt
  torch/squeeze
  torch/stack
  torch/sub
  torch/sum
  torch/tensor
  torch/topk
  torch/transpose
  torch/unique
  torch/unsqueeze
  torch/where
  torch/zeros_like
  torch/zeros

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
