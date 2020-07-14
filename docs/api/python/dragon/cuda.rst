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

  `enable_cudnn(...) <cuda/enable_cudnn.html>`_
  : Enable the CuDNN library.

  `get_device_capability(...) <cuda/get_device_capability.html>`_
  : Return the capability of specified device.

  `is_available(...) <cuda/is_available.html>`_
  : Return a bool reporting if runtime is available.

  `set_default_device(...) <cuda/set_default_device.html>`_
  : Set the default device.

  `set_device(...) <cuda/set_device.html>`_
  : Set the current device.

  `synchronize(...) <cuda/synchronize.html>`_
  : Synchronize a specified CUDA stream.

.. toctree::
  :hidden:

  cuda/current_device
  cuda/enable_cudnn
  cuda/get_device_capability
  cuda/is_available
  cuda/set_default_device
  cuda/set_device
  cuda/Stream
  cuda/synchronize

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
