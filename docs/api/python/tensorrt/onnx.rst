vm.tensorrt.onnx
================

.. only:: html

  Classes
  -------

  `class BackendRep <onnx/BackendRep.html>`_
  : ONNX-TensorRT backend to execute repeatedly.

  Functions
  ---------

  `prepare_backend(...) <onnx/prepare_backend.html>`_
  : Create a backend to execute repeatedly.

  `run_model(...) <backend/run_model.html>`_
  : Execute an onnx model once.

  `run_node(...) <backend/run_node.html>`_
  : Execute an onnx node once.

  `supports_device(...) <backend/supports_device.html>`_
  : Query if the given device is supported to execute.

.. toctree::
  :hidden:

  onnx/BackendRep
  onnx/prepare
  onnx/run_model
  onnx/run_node
  onnx/supports_device

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
