vm.torch.mps
=============

.. only:: html

  Functions
  ---------

  `current_device(...) <mps/current_device.html>`_
  : Return the index of current selected device.

  `device_count(...) <mps/device_count.html>`_
  : Return the number of available devices.

  `get_device_family(...) <mps/get_device_family.html>`_
  : Return the family of specified device.

  `get_device_name(...) <mps/get_device_name.html>`_
  : Return the name of specified device.

  `is_available(...) <mps/is_available.html>`_
  : Return a bool reporting if runtime is available.

  `set_device(...) <mps/set_device.html>`_
  : Set the current device.

  `manual_seed(...) <mps/manual_seed.html>`_
  : Set the random seed for mps device.

  `manual_seed_all(...) <mps/manual_seed_all.html>`_
  : Set the random seed for all mps devices.

  `memory_allocated(...) <mps/memory_allocated.html>`_
  : Return the size of memory used by tensors in current workspace.

  `synchronize(...) <mps/synchronize.html>`_
  : Synchronize all streams on a device.

.. toctree::
  :hidden:

  mps/current_device
  mps/device_count
  mps/get_device_family
  mps/get_device_name
  mps/is_available
  mps/manual_seed
  mps/manual_seed_all
  mps/memory_allocated
  mps/set_device
  mps/synchronize

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
