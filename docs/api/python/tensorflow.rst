vm.tensorflow
=============

.. only:: html

  Classes
  #######

  `class GradientTape <tensorflow/GradientTape.html>`_
  : Record the operations for auto differentiation.

  `class TensorShape <tensorflow/TensorShape.html>`_
  : Represent the a sequence of dimensions.

  `class TensorSpec <tensorflow/TensorSpec.html>`_
  : Spec to describe properties of a tensor.

  Functions
  #########

  `broadcast_to(...) <dragon/broadcast_to.html>`_
  : Broadcast input according to a given shape.

  `cast(...) <tensorflow/cast.html>`_
  : Cast the data type of input.

  `clip_by_value(...) <tensorflow/clip_by_value.html>`_
  : Compute the clipped input according to the given bounds.

  `concat(...) <tensorflow/concat.html>`_
  : Concatenate the values along the given axis.

  `constant(...) <tensorflow/constant.html>`_
  : Return a tensor initialized from the value.

  `device(...) <tensorflow/device.html>`_
  : Context-manager to nest the the device spec.

  `expand_dims(...) <tensorflow/expand_dims.html>`_
  : Expand the dimensions of input with size 1.

  `eye(...) <tensorflow/eye.html>`_
  : Return a tensor constructed as the identity matrix.

  `fill(...) <tensorflow/fill.html>`_
  : Return a tensor filled with the scalar value.

  `gather(...) <tensorflow/gather.html>`_
  : Select the elements according to the index along the given axis.

  `function(...) <tensorflow/function.html>`_
  : Create a callable graph from the python function.

  `gradients(...) <tensorflow/gradients.html>`_
  : Compute the symbolic derivatives of ``ys`` w.r.t. ``xs`` .

  `identity(...) <tensorflow/identity.html>`_
  : Return a new tensor copying the content of input.

  `name_scope(...) <tensorflow/name_scope.html>`_
  : Context-manager to nest the name as prefix for operations.

  `ones(...) <tensorflow/ones.html>`_
  : Return a tensor filled with ones.

  `ones_like(...) <tensorflow/ones_like.html>`_
  : Return a tensor of ones with shape as the other.

  `one_hot(...) <tensorflow/one_hot.html>`_
  : Return the one-hot representation for input.

  `pad(...) <tensorflow/pad.html>`_
  : Pad the input according to the given sizes.

  `range(...) <tensorflow/range.html>`_
  : Return a tensor of evenly spaced values within a interval.

  `reshape(...) <tensorflow/reshape.html>`_
  : Change the dimensions of input.

  `shape(...) <tensorflow/shape.html>`_
  : Return the shape of input.

  `slice(...) <tensorflow/slice.html>`_
  : Select the elements according to the given sections.

  `split(...) <tensorflow/split.html>`_
  : Split input into chunks along the given axis.

  `squeeze(...) <tensorflow/squeeze.html>`_
  : Remove the dimensions of input with size 1.

  `transpose(...) <tensorflow/transpose.html>`_
  : Permute the dimensions of input.

  `unique(...) <tensorflow/unique.html>`_
  : Return the unique elements of input.

  `unique_with_counts(...) <tensorflow/unique_with_counts.html>`_
  : Return the unique elements of input with counts.

  `zeros(...) <tensorflow/zeros.html>`_
  : Return a tensor filled with zeros.

  `zeros_like(...) <tensorflow/zeros_like.html>`_
  : Return a tensor of zeros with shape as the other.

.. toctree::
  :hidden:

  tensorflow/broadcast_to
  tensorflow/cast
  tensorflow/clip_by_value
  tensorflow/concat
  tensorflow/constant
  tensorflow/device
  tensorflow/expand_dims
  tensorflow/eye
  tensorflow/fill
  tensorflow/function
  tensorflow/gather
  tensorflow/gradients
  tensorflow/GradientTape
  tensorflow/identity
  tensorflow/name_scope
  tensorflow/ones
  tensorflow/ones_like
  tensorflow/one_hot
  tensorflow/pad
  tensorflow/range
  tensorflow/reshape
  tensorflow/shape
  tensorflow/slice
  tensorflow/split
  tensorflow/squeeze
  tensorflow/TensorShape
  tensorflow/TensorSpec
  tensorflow/transpose
  tensorflow/unique
  tensorflow/unique_with_counts
  tensorflow/zeros
  tensorflow/zeros_like

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
