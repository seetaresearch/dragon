vm.torch.mlu
=============

.. only:: html

  Functions
  ---------

  `current_device(...) <mlu/current_device.html>`_
  : Return the index of current selected device.

  `device_count(...) <mlu/device_count.html>`_
  : Return the number of available devices.

  `get_device_capability(...) <mlu/get_device_capability.html>`_
  : Return the capability of specified device.

  `get_device_name(...) <mlu/get_device_name.html>`_
  : Return the name of specified device.

  `is_available(...) <mlu/is_available.html>`_
  : Return a bool reporting if runtime is available.

  `manual_seed(...) <mlu/manual_seed.html>`_
  : Set the random seed for mlu device.

  `manual_seed_all(...) <mlu/manual_seed_all.html>`_
  : Set the random seed for all mlu devices.

  `memory_allocated(...) <mlu/memory_allocated.html>`_
  : Return the size of memory used by tensors in current workspace.

  `set_device(...) <mlu/set_device.html>`_
  : Set the current device.

  `synchronize(...) <mlu/synchronize.html>`_
  : Synchronize all streams on a device.

.. toctree::
  :hidden:

  mlu/current_device
  mlu/device_count
  mlu/get_device_capability
  mlu/get_device_name
  mlu/is_available
  mlu/manual_seed
  mlu/manual_seed_all
  mlu/memory_allocated
  mlu/set_device
  mlu/synchronize

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
