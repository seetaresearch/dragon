dragon
======

.. only:: html

  Classes
  -------

  `class DeviceSpec <dragon/DeviceSpec.html>`_
  : Describe a computation device.

  `class GradientTape <dragon/GradientTape.html>`_
  : Record the operations for auto differentiation.

  `class Tensor <dragon/Tensor.html>`_
  : A multi-dimensional array for computation

  `class Workspace <dragon/Workspace.html>`_
  : Standalone environment for resources and computations.

  Functions
  ---------

  `argsort(...) <dragon/argsort.html>`_
  : Return the index of sorted elements along the given axis.

  `assign(...) <dragon/assign.html>`_
  : Assign the value to input.

  `boolean_mask(...) <dragon/boolean_mask.html>`_
  : Return the elements of input where mask is true.

  `broadcast_to(...) <dragon/broadcast_to.html>`_
  : Broadcast input according to a given shape.

  `cast(...) <dragon/cast.html>`_
  : Cast the data type of input.

  `concat(...) <dragon/concat.html>`_
  : Concatenate the inputs along the given axis.

  `constant(...) <dragon/constant.html>`_
  : Return a tensor initialized from the value.

  `device(...) <dragon/device.html>`_
  : Context-manager to nest the device spec.

  `eager_mode(...) <dragon/eager_mode.html>`_
  : Context-manager set the eager execution mode.

  `variable_scope(...) <dragon/eager_mode.html>`_
  : Context-manager to nest the namespace for variables.

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

  `gather(...) <dragon/gather.html>`_
  : Gather elements along the given axis using index.

  `gather_elements(...) <dragon/gather_elements.html>`_
  : Gather elements along the given axis of index.

  `get_num_threads(...) <dragon/get_num_threads.html>`_
  : Return the number of threads for cpu parallelism.

  `get_workspace(...) <dragon/get_workspace.html>`_
  : Return the default workspace.

  `graph_mode(...) <dragon/graph_mode.html>`_
  : Context-manager set the graph execution mode.

  `identity(...) <dragon/identity.html>`_
  : Return a tensor copied from the input.

  `linspace(...) <dragon/linspace.html>`_
  : Generate evenly spaced values within intervals along the given axis.

  `load_library(...) <dragon/load_library.html>`_
  : Load a shared library.

  `name_scope(...) <dragon/name_scope.html>`_
  : Context-manager to nest the name as prefix for operations.

  `nonzero(...) <dragon/nonzero.html>`_
  : Return the index of non-zero elements.

  `ones(...) <dragon/ones.html>`_
  : Return a tensor filled with ones.

  `ones_like(...) <dragon/ones_like.html>`_
  : Return a tensor of ones with shape as the other.

  `one_hot(...) <dragon/one_hot.html>`_
  : Return the one-hot representation of input.

  `pad(...) <dragon/pad.html>`_
  : Pad the input according to the given sizes.

  `python_plugin(...) <dragon/python_plugin.html>`_
  : Create a plugin operator from the python class.

  `range(...) <dragon/range.html>`_
  : Return a tensor of evenly spaced values within a interval.

  `repeat(...) <dragon/repeat.html>`_
  : Repeat the elements along the given axis.

  `reset_workspace(...) <dragon/reset_workspace.html>`_
  : Reset the current default workspace.

  `reshape(...) <dragon/reshape.html>`_
  : Change the dimensions of input.

  `reverse(...) <dragon/reverse.html>`_
  : Reverse elements along the given axis.
 
  `roll(...) <dragon/roll.html>`_
  : Roll elements along the given axis.

  `scatter_add(...) <dragon/scatter_add.html>`_
  : Add elements along the given axis of index.

  `scatter_elements(...) <dragon/scatter_elements.html>`_
  : Update elements along the given axis of index.

  `set_num_threads(...) <dragon/set_num_threads.html>`_
  : Set the number of threads for cpu parallelism.

  `shape(...) <dragon/shape.html>`_
  : Return the shape of input.

  `slice(...) <dragon/slice.html>`_
  : Select the elements according to the given sections.

  `sort(...) <dragon/sort.html>`_
  : Return the sorted elements along the given axis.

  `split(...) <dragon/split.html>`_
  : Split the input into chunks along the given axis.

  `squeeze(...) <dragon/squeeze.html>`_
  : Remove the dimensions of input with size 1.

  `stack(...) <dragon/stack.html>`_
  : Stack the inputs along the given axis.

  `stop_gradient(...) <dragon/stop_gradient.html>`_
  : Return the identity of input with truncated gradient-flow.

  `tile(...) <dragon/tile.html>`_
  : Repeat elements along each axis of input.

  `transpose(...) <dragon/transpose.html>`_
  : Permute the dimensions of input.

  `tril(...) <dragon/tril.html>`_
  : Return the lower triangular part of input.

  `triu(...) <dragon/triu.html>`_
  : Return the upper triangular part of input.

  `unstack(...) <dragon/unstack.html>`_
  : Unpack input into chunks along the given axis.

  `unique(...) <dragon/unique.html>`_
  : Return the unique elements of input.

  `where(...) <dragon/where.html>`_
  : Select the elements from two branches under the condition.

  `zeros(...) <dragon/zeros.html>`_
  : Return a tensor filled with zeros.

  `zeros_like(...) <dragon/zeros_like.html>`_
  : Return a tensor of zeros with shape as the other.

.. toctree::
  :hidden:

  dragon/DeviceSpec
  dragon/GradientTape
  dragon/Tensor
  dragon/Workspace
  dragon/argsort
  dragon/assign
  dragon/boolean_mask
  dragon/broadcast_to
  dragon/cast
  dragon/concat
  dragon/constant
  dragon/device
  dragon/eager_mode
  dragon/expand_dims
  dragon/eye
  dragon/eye_like
  dragon/fill
  dragon/flatten
  dragon/function
  dragon/gather
  dragon/gather_elements
  dragon/get_num_threads
  dragon/get_workspace
  dragon/graph_mode
  dragon/identity
  dragon/linspace
  dragon/load_library
  dragon/name_scope
  dragon/nonzero
  dragon/ones
  dragon/ones_like
  dragon/one_hot
  dragon/pad
  dragon/python_plugin
  dragon/range
  dragon/repeat
  dragon/reset_workspace
  dragon/reshape
  dragon/reverse
  dragon/roll
  dragon/scatter_add
  dragon/scatter_elements
  dragon/set_num_threads
  dragon/shape
  dragon/slice
  dragon/sort
  dragon/split
  dragon/squeeze
  dragon/stack
  dragon/stop_gradient
  dragon/tile
  dragon/transpose
  dragon/tril
  dragon/triu
  dragon/unique
  dragon/unstack
  dragon/variable_scope
  dragon/where
  dragon/zeros
  dragon/zeros_like

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
