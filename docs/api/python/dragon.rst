dragon
======

.. only:: html

  Classes
  -------

  `class EagerTensor <dragon/EagerTensor.html>`_
  : Tensor abstraction for eager executing.

  `class GradientTape <dragon/GradientTape.html>`_
  : Record the operations for auto differentiation.

  `class Tensor <dragon/Tensor.html>`_
  : Tensor abstraction for graph executing.

  `class Workspace <dragon/Workspace.html>`_
  : Sandbox to isolate the resources and computations.

  Functions
  ---------

  `arange(...) <dragon/arange.html>`_
  : Return a tensor of evenly spaced values within a interval.

  `assign(...) <dragon/assign.html>`_
  : Assign the value to input.

  `broadcast_to(...) <dragon/broadcast_to.html>`_
  : Broadcast input according to a given shape.

  `cast(...) <dragon/cast.html>`_
  : Cast the data type of input.

  `channel_normalize(...) <dragon/channel_normalize.html>`_
  : Normalize channels with mean and standard deviation.

  `channel_shuffle(...) <dragon/channel_shuffle.html>`_
  : Shuffle channels between a given number of groups.

  `concat(...) <dragon/concat.html>`_
  : Concatenate the inputs along the given axis.

  `constant(...) <dragon/constant.html>`_
  : Return a tensor initialized from the value.

  `copy(...) <dragon/copy.html>`_
  : Copy the input.

  `create_function(...) <dragon/create_function.html>`_
  : Create a callable graph from the specified outputs.

  `device(...) <dragon/device.html>`_
  : Context-manager to nest the the device spec.

  `eager_mode(...) <dragon/eager_mode.html>`_
  : Context-manager set the eager execution mode.

  `eager_scope(...) <dragon/eager_mode.html>`_
  : Context-manager to nest the name for eager resources.

  `expand_dims(...) <dragon/expand_dims.html>`_
  : Expand the dimensions of input with size 1.

  `eye(...) <dragon/eye.html>`_
  : Return a tensor constructed as the identity matrix.

  `eye_like(...) <dragon/eye_like.html>`_
  :Return a tensor of identity matrix with shape as the other.

  `fill(...) <dragon/fill.html>`_
  : Return a tensor filled with the scalar value.

  `flatten(...) <dragon/flatten.html>`_
  : Flatten the input along the given axes.

  `function(...) <dragon/function.html>`_
  : Compile a function and return an executable.

  `get_workspace(...) <dragon/get_workspace.html>`_
  : Return the current default workspace.

  `gradients(...) <dragon/gradients.html>`_
  : Compute the symbolic derivatives of ``ys`` w.r.t. ``xs`` .

  `graph_mode(...) <dragon/graph_mode.html>`_
  : Context-manager set the graph execution mode.

  `index_select(...) <dragon/index_select.html>`_
  : Select the elements according to the indices along the given axis.

  `load_library(...) <dragon/load_library.html>`_
  : Load a shared library.

  `masked_assign(...) <dragon/masked_assign.html>`_
  : Assign the value to input where mask is 1.

  `masked_select(...) <dragon/masked_select.html>`_
  : Select the elements of input where mask is 1.

  `name_scope(...) <dragon/name_scope.html>`_
  : Context-manager to nest the name as prefix for operations.

  `nonzero(...) <dragon/nonzero.html>`_
  : Return the index of non-zero elements.

  `ones(...) <dragon/ones.html>`_
  : Return a tensor filled with ones.

  `ones_like(...) <dragon/ones_like.html>`_
  : Return a tensor of ones with shape as the other.

  `one_hot(...) <dragon/one_hot.html>`_
  : Return the one-hot representation for input.

  `pad(...) <dragon/pad.html>`_
  : Pad the input according to the given sizes.

  `python_plugin(...) <dragon/python_plugin.html>`_
  : Create a plugin operator from the python class.

  `repeat(...) <dragon/repeat.html>`_
  : Repeat the elements along the given axis.

  `reset_workspace(...) <dragon/reset_workspace.html>`_
  : Reset the current default workspace.

  `reshape(...) <dragon/reshape.html>`_
  : Change the dimensions of input.

  `shape(...) <dragon/shape.html>`_
  : Return the shape of input.

  `slice(...) <dragon/slice.html>`_
  : Select the elements according to the given sections.

  `split(...) <dragon/split.html>`_
  : Split the input into chunks along the given axis.

  `squeeze(...) <dragon/squeeze.html>`_
  : Remove the dimensions of input with size 1.

  `stack(...) <dragon/stack.html>`_
  : Stack the inputs along the given axis.

  `stop_gradient(...) <dragon/stop_gradient.html>`_
  : Return the identity of input with truncated gradient-flow.

  `tile(...) <dragon/tile.html>`_
  : Tile the input according to the given repeats.

  `transpose(...) <dragon/transpose.html>`_
  : Permute the dimensions of input.

  `where(...) <dragon/where.html>`_
  : Select the elements from two branches under the condition.

  `zeros(...) <dragon/zeros.html>`_
  : Return a tensor filled with zeros.

  `zeros_like(...) <dragon/zeros_like.html>`_
  : Return a tensor of zeros with shape as the other.

.. toctree::
  :hidden:

  dragon/arange
  dragon/assign
  dragon/broadcast_to
  dragon/cast
  dragon/channel_normalize
  dragon/channel_shuffle
  dragon/concat
  dragon/constant
  dragon/copy
  dragon/create_function
  dragon/device
  dragon/EagerTensor
  dragon/eager_mode
  dragon/eager_scope
  dragon/expand_dims
  dragon/eye
  dragon/eye_like
  dragon/fill
  dragon/flatten
  dragon/function
  dragon/get_workspace
  dragon/gradients
  dragon/GradientTape
  dragon/graph_mode
  dragon/index_select
  dragon/load_library
  dragon/masked_assign
  dragon/masked_select
  dragon/name_scope
  dragon/nonzero
  dragon/ones
  dragon/ones_like
  dragon/one_hot
  dragon/pad
  dragon/python_plugin
  dragon/repeat
  dragon/reset_workspace
  dragon/reshape
  dragon/shape
  dragon/slice
  dragon/split
  dragon/squeeze
  dragon/stack
  dragon/stop_gradient
  dragon/Tensor
  dragon/tile
  dragon/transpose
  dragon/where
  dragon/Workspace
  dragon/zeros
  dragon/zeros_like

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
