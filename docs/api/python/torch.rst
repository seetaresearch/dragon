vm.torch
========

.. only:: html

  Classes
  -------

  `class device <torch/device.html>`_
  : Represent the device spec.

  `class dtype <torch/device.html>`_
  : The basic data type.

  `class enable_grad <torch/enable_grad.html>`_
  : Context-manager to enable gradient calculation.

  `class inference_mode <torch/inference_mode.html>`_
  : Context-manager to enable or disable inference mode.

  `class no_grad <torch/no_grad.html>`_
  : Context-manager to disable gradient calculation.

  `class set_grad_enabled <torch/set_grad_enabled.html>`_
  : Context-manager to set gradient calculation on or off.

  `class Size <torch/Size.html>`_
  : Represent the a sequence of dimensions.

  `class Tensor <torch/Tensor_.html>`_
  : A multi-dimensional array containing elements of a single data type.

  Functions
  ---------

  `abs(...) <torch/abs.html>`_
  : Compute the absolute value of input.

  `add(...) <torch/add.html>`_
  : Compute element-wise addition.

  `addmm(...) <torch/addmm.html>`_
  : Add input to the result of matrix-matrix multiplication.

  `arange(...) <torch/arange.html>`_
  : Return a tensor of evenly spaced values within a interval.

  `argmax(...) <torch/argmax.html>`_
  : Return the index of maximum elements along the given dimension.

  `argmin(...) <torch/argmin.html>`_
  : Return the index of minimum elements along the given dimension.

  `argsort(...) <torch/argsort.html>`_
  : Return the index of sorted elements along the given dimension.

  `as_tensor(...) <torch/as_tensor.html>`_
  : Create a tensor sharing the given data.

  `atan2(...) <torch/atan2.html>`_
  : Compute element-wise arc-tangent of two arguments.

  `baddbmm(...) <torch/baddbmm.html>`_
  : Add input to the result of batched matrix-matrix multiplication.

  `bitwise_and(...) <torch/bitwise_and.html>`_
  : Compute element-wise AND bitwise operation.

  `bitwise_not(...) <torch/bitwise_not.html>`_
  : Compute element-wise NOT bitwise operation.

  `bitwise_or(...) <torch/bitwise_or.html>`_
  : Compute element-wise OR bitwise operation.

  `bitwise_xor(...) <torch/bitwise_xor.html>`_
  : Compute element-wise XOR bitwise operation.

  `bmm(...) <torch/bmm.html>`_
  : Compute batched matrix-matrix multiplication.

  `cat(...) <torch/cat.html>`_
  : Concatenate the inputs along the given dimension.

  `ceil(...) <torch/ceil.html>`_
  : Compute the smallest integer not less than input.

  `chunk(...) <torch/chunk.html>`_
  : Split input into a specific number of chunks.

  `clamp(...) <torch/clamp.html>`_
  : Clip input according to the given bounds.

  `cos(...) <torch/cos.html>`_
  : Compute the cos of input.

  `cummax(...) <torch/cummax.html>`_
  : Compute the cumulative maximum of elements along the given dimension.

  `cummin(...) <torch/cummin.html>`_
  : Compute the cumulative minimum of elements along the given dimension.

  `cumsum(...) <torch/cumsum.html>`_
  : Compute the cumulative sum of elements along the given dimension.

  `div(...) <torch/div.html>`_
  : Compute element-wise division.

  `empty(...) <torch/empty.html>`_
  : Return a tensor filled with uninitialized data.

  `eq(...) <torch/eq.html>`_
  : Compute element-wise equal comparison.

  `exp(...) <torch/exp.html>`_
  : Compute the exponential of input.

  `eye(...) <torch/eye.html>`_
  : Return a tensor constructed as the identity matrix.

  `flatten(...) <torch/flatten.html>`_
  : Return a tensor with dimensions flattened.

  `flip(...) <torch/flip.html>`_
  : Reverse elements along the given dimension.

  `fliplr(...) <torch/fliplr.html>`_
  : Reverse elements along the second dimension.

  `flipud(...) <torch/flipud.html>`_
  : Reverse elements along the first dimension.

  `floor(...) <torch/floor.html>`_
  : Compute the largest integer not greater than input.

  `from_numpy(...) <torch/from_numpy.html>`_
  : Create a tensor converting from the given numpy array.

  `full(...) <torch/full.html>`_
  : Return a tensor filled with a scalar.

  `full_like(...) <torch/full_like.html>`_
  : Return a tensor filled with a scalar with size as input.

  `gather(...) <torch/gather.html>`_
  : Gather elements along the given dimension of index.

  `ge(...) <torch/ge.html>`_
  : Compute element-wise greater-equal comparison.

  `gt(...) <torch/gt.html>`_
  : Compute element-wise greater comparison.

  `index_select(...) <torch/index_select.html>`_
  : Select elements along the given dimension using index.

  `isfinite(...) <torch/isfinite.html>`_
  : Check if the elements of input are finite.

  `isinf(...) <torch/isinf.html>`_
  : Check if the elements of input are infinite.

  `isnan(...) <torch/isnan.html>`_
  : Check if the elements of input are NaN.

  `le(...) <torch/le.html>`_
  : Compute element-wise less-equal comparison.

  `linspace(...) <torch/linspace.html>`_
  : Generate evenly spaced values within intervals along the given dimension.

  `log(...) <torch/log.html>`_
  : Compute the natural logarithm of input.

  `logical_and(...) <torch/logical_and.html>`_
  : Compute element-wise AND logical operation.

  `logical_not(...) <torch/logical_not.html>`_
  : Compute element-wise NOT logical operation.

  `logical_or(...) <torch/logical_or.html>`_
  : Compute element-wise OR logical operation.

  `logical_xor(...) <torch/logical_xor.html>`_
  : Compute element-wise XOR logical operation.

  `logsumexp(...) <torch/logsumexp.html>`_
  : Apply the composite of log, sum, and exp to input.

  `lt(...) <torch/lt.html>`_
  : Compute element-wise less comparison.

  `manual_seed(...) <torch/manual_seed.html>`_
  : Set the random seed for cpu device.

  `masked_select(...) <torch/logsumexp.html>`_
  : Select the input elements where mask is true.

  `matmul(...) <torch/matmul.html>`_
  : Compute matrix multiplication.

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
  : Compute matrix-matrix multiplication.

  `mul(...) <torch/mul.html>`_
  : Compute element-wise multiplication.

  `multinomial(...) <torch/multinomial.html>`_
  : Return a tensor with index sampled from multinomial distribution.

  `nan_to_num(...) <torch/nan_to_num.html>`_
  : Replace NaN or infinity elements with given value.

  `narrow(...) <torch/narrow.html>`_
  : Return a new tensor that is a narrowed version of input tensor.

  `ne(...) <torch/ne.html>`_
  : Compute lement-wise not-equal comparison.

  `neg(...) <torch/neg.html>`_
  : Compute element-wise negative.

  `nonzero(...) <torch/nonzero.html>`_
  : Return the index of non-zero elements.

  `norm(...) <torch/norm.html>`_
  : Compute the min value of elements along the given dimension.

  `ones(...) <torch/ones.html>`_
  : Return a tensor filled with ones.

  `ones_like(...) <torch/ones_like.html>`_
  : Return a tensor of ones with shape as the other.

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

  `reshape(...) <torch/reshape.html>`_
  : Change the shape of input.

  `roll(...) <torch/roll.html>`_
  : Roll elements along the given dimension.

  `round(...) <torch/round.html>`_
  : Compute the nearest integer of input.

  `rsqrt(...) <torch/rsqrt.html>`_
  : Compute the reciprocal square root of input.

  `scatter(...) <torch/scatter.html>`_
  : Update elements along the given dimension of index.

  `scatter_add(...) <torch/scatter_add.html>`_
  : Add elements along the given dimension of index.

  `sigmoid(...) <torch/sigmoid.html>`_
  : Compute the sigmoid of input.

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

  `square(...) <torch/square.html>`_
  : Compute the square of input.

  `squeeze(...) <torch/squeeze.html>`_
  : Remove the dimensions of input with size 1.

  `stack(...) <torch/stack.html>`_
  : Stack the inputs along the given dimension.

  `sub(...) <torch/sub.html>`_
  : Compute element-wise subtraction.

  `sum(...) <torch/sum.html>`_
  : Compute the sum value of elements along the given dimension.

  `tanh(...) <torch/tanh.html>`_
  : Compute the tanh of input.

  `tensor(...) <torch/tensor.html>`_
  : Create a tensor initializing from the given data.

  `tile(...) <torch/tile.html>`_
  : Repeat elements along each dimension of input.

  `topk(...) <torch/topk.html>`_
  : Return the top k-largest or k-smallest elements along the given dimension.

  `transpose(...) <torch/transpose.html>`_
  : Return a new tensor with two dimensions swapped.

  `tril(...) <torch/tril.html>`_
  : Return the lower triangular part of input.

  `triu(...) <torch/triu.html>`_
  : Return the upper triangular part of input.

  `unbind(...) <torch/unbind.html>`_
  : Unpack input into chunks along the given dimension.

  `unique(...) <torch/unique.html>`_
  : Return the unique elements of input.

  `unsqueeze(...) <torch/unsqueeze.html>`_
  : Expand the dimensions of input with size 1.

  `where(...) <torch/where.html>`_
  : Select the elements from two branches under the condition.

  `var(...) <torch/var.html>`_
  : Compute the variance value of elements along the given dimension.

  `var_mean(...) <torch/var_mean.html>`_
  : Compute the variance and mean of elements along the given dimension.

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
  torch/addmm
  torch/arange
  torch/argmax
  torch/argmin
  torch/argsort
  torch/as_tensor
  torch/atan2
  torch/baddbmm
  torch/bitwise_and
  torch/bitwise_not
  torch/bitwise_or
  torch/bitwise_xor
  torch/bmm
  torch/cat
  torch/ceil
  torch/chunk
  torch/clamp
  torch/cos
  torch/cummax
  torch/cummin
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
  torch/flip
  torch/fliplr
  torch/flipud
  torch/floor
  torch/from_numpy
  torch/full
  torch/full_like
  torch/gather
  torch/ge
  torch/gt
  torch/index_select
  torch/inference_mode
  torch/isfinite
  torch/isinf
  torch/isnan
  torch/le
  torch/linspace
  torch/log
  torch/logical_and
  torch/logical_not
  torch/logical_or
  torch/logical_xor
  torch/logsumexp
  torch/lt
  torch/manual_seed
  torch/masked_select
  torch/matmul
  torch/max
  torch/maximum
  torch/mean
  torch/min
  torch/minimum
  torch/mm
  torch/mul
  torch/multinomial
  torch/nan_to_num
  torch/narrow
  torch/ne
  torch/neg
  torch/no_grad
  torch/nonzero
  torch/norm
  torch/ones
  torch/ones_like
  torch/permute
  torch/pow
  torch/rand
  torch/randn
  torch/randperm
  torch/reciprocal
  torch/reshape
  torch/roll
  torch/round
  torch/rsqrt
  torch/scatter
  torch/scatter_add
  torch/set_grad_enabled
  torch/sigmoid
  torch/sign
  torch/sin
  torch/sort
  torch/split
  torch/sqrt
  torch/square
  torch/squeeze
  torch/stack
  torch/sub
  torch/sum
  torch/tanh
  torch/tensor
  torch/tile
  torch/topk
  torch/transpose
  torch/tril
  torch/triu
  torch/unbind
  torch/unique
  torch/unsqueeze
  torch/where
  torch/var
  torch/var_mean
  torch/zeros_like
  torch/zeros

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
