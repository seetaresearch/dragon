dragon.cuda
===========

.. only:: html

  Classes
  -------

  `class Stream <cuda/Stream.html>`_
  : The CUDA stream wrapper.

  Functions
  ---------

  `current_device(...) <cuda/current_device.html>`_
  : Return the index of current selected device.

  `get_device_capability(...) <cuda/get_device_capability.html>`_
  : Return the capability of specified device.

  `get_device_count(...) <cuda/get_device_count.html>`_
  : Return the number of available devices.

  `get_device_name(...) <cuda/get_device_name.html>`_
  : Return the name of specified device.

  `is_available(...) <cuda/is_available.html>`_
  : Return a bool reporting if runtime is available.

  `memory_allocated(...) <cuda/memory_allocated.html>`_
  : Return the size of memory used by tensors in current workspace.

  `set_cublas_flags(...) <cuda/set_cublas_flags.html>`_
  : Set the flags of cuBLAS library.

  `set_cudnn_flags(...) <cuda/set_cudnn_flags.html>`_
  : Set the flags of cuDNN library.

  `set_default_device(...) <cuda/set_default_device.html>`_
  : Set the default device.

  `set_device(...) <cuda/set_device.html>`_
  : Set the current device.

  `set_random_seed(...) <cuda/set_random_seed.html>`_
  : Set the random seed for cuda device.

  `synchronize(...) <cuda/synchronize.html>`_
  : Synchronize a specified CUDA stream.

.. toctree::
  :hidden:

  cuda/Stream
  cuda/current_device
  cuda/get_device_capability
  cuda/get_device_count
  cuda/get_device_name
  cuda/is_available
  cuda/memory_allocated
  cuda/set_cublas_flags
  cuda/set_cudnn_flags
  cuda/set_default_device
  cuda/set_device
  cuda/set_random_seed
  cuda/synchronize

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
