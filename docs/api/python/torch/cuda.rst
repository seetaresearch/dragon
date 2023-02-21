vm.torch.cuda
=============

.. only:: html

  Functions
  ---------

  `current_device(...) <cuda/current_device.html>`_
  : Return the index of current selected device.

  `device_count(...) <cuda/device_count.html>`_
  : Return the number of available devices.

  `get_device_capability(...) <cuda/get_device_capability.html>`_
  : Return the capability of specified device.

  `get_device_name(...) <cuda/get_device_name.html>`_
  : Return the name of specified device.

  `is_available(...) <cuda/is_available.html>`_
  : Return a bool reporting if runtime is available.

  `manual_seed(...) <cuda/manual_seed.html>`_
  : Set the random seed for cuda device.

  `manual_seed_all(...) <cuda/manual_seed_all.html>`_
  : Set the random seed for all cuda devices.

  `memory_allocated(...) <cuda/memory_allocated.html>`_
  : Return the size of memory used by tensors in current workspace.

  `set_device(...) <cuda/set_device.html>`_
  : Set the current device.

  `synchronize(...) <cuda/synchronize.html>`_
  : Synchronize all streams on a device.

.. toctree::
  :hidden:

  cuda/current_device
  cuda/device_count
  cuda/get_device_capability
  cuda/get_device_name
  cuda/is_available
  cuda/manual_seed
  cuda/manual_seed_all
  cuda/memory_allocated
  cuda/set_device
  cuda/synchronize

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
