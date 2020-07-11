Dragon - Python API
===================

Styles
------

Dragon is a computation graph based distributed deep learning framework.

For using it, import as follows:

.. code-block:: python

  import dragon

However, it will not help you much because you do not want to learn it.

To resolve this matter, we are concerned to design diverse styles for you:

Dragon
######

  *Dragon* is initially as a light-weight but professional style.

  Native interfaces are encouraged to manipulate the backend engine
  to perform the computation flexibly with data feeding or fetching.

  This style involves the following components:

  * `dragon <dragon.html>`_
  * `dragon.autograph <dragon/autograph.html>`_
  * `dragon.bitwise <dragon/bitwise.html>`_
  * `dragon.cuda <dragon/cuda.html>`_
  * `dragon.distributed <dragon/distributed.html>`_
  * `dragon.dlpack <dragon/dlpack.html>`_
  * `dragon.io <dragon/io.html>`_
  * `dragon.logging <dragon/logging.html>`_
  * `dragon.losses <dragon/losses.html>`_
  * `dragon.math <dragon/math.html>`_
  * `dragon.metrics <dragon/metrics.html>`_
  * `dragon.nn <dragon/nn.html>`_
  * `dragon.optimizers <dragon/optimizers.html>`_
  * `dragon.random <dragon/random.html>`_
  * `dragon.vision <dragon/vision.html>`_

Caffe
#####

  *Caffe* is the most famous framework for vision.

  Our work is very different from the official python wrappers, a.k.a,
  the *PyCaffe*, which comes from the exports of *BoostPython*
  based on C++ language.

  This style involves the following components:

  * `caffe <caffe.html>`_
  * `caffe.layers <caffe/layers.html>`_

TensorFlow
##########

  *TensorFlow* is an end-to-end open source platform for machine learning.

  It is challenging to make a consistency with graph and eager execution
  while is practical to cover most of *TensorFlow* operations using our backend.
  We have designed several simpler mechanisms to equip for our frontend to
  provide exactly the same inferfaces.

  This style involves the following components:

  * `tensorflow <tensorflow.html>`_
  * `tensorflow.bitwise <tensorflow/bitwise.html>`_
  * `tensorflow.dtypes <tensorflow/dtypes.html>`_
  * `tensorflow.initializers <tensorflow/initializers.html>`_
  * `tensorflow.keras <tensorflow/keras.html>`_
  * `tensorflow.linalg <tensorflow/linalg.html>`_
  * `tensorflow.losses <tensorflow/losses.html>`_
  * `tensorflow.math <tensorflow/math.html>`_
  * `tensorflow.nn <tensorflow/nn.html>`_
  * `tensorflow.optimizers <tensorflow/optimizers.html>`_
  * `tensorflow.random <tensorflow/random.html>`_

TensorLayer
###########

  *TensorLayer* takes a high-level layer abstraction to build complex models.

  Original *TensorLayer* project is restricted to execute *TensorFlow* operations,
  which is also known as a competitive tf-wrapper comparing to *tf.keras*.
  We transplant and remake the abstractions to match our engine, while keeping
  the compatibility with tf-based codes as possible.

  This style involves the following components:

  * `tensorlayer.initializers <tensorlayer/initializers.html>`_
  * `tensorlayer.layers <tensorlayer/layers.html>`_
  * `tensorlayer.models <tensorlayer/models.html>`_

PyTorch
#######

  *PyTorch* provides straight-forward operations on research prototyping.

  To bridge it, our *JIT* traces and dispatches the operations,
  as well as the rewriting of *GC* (Garbage Collection) to reuse
  the memories and operators by turns.

  We are still working hard to cover the original *PyTorch* operators,
  however, a bunch of extended operators in many other frameworks can be used.

  This style involves the following components:

  * `torch <torch.html>`_
  * `torch.autograd <torch/autograd.html>`_
  * `torch.distributed <torch/distributed.html>`_
  * `torch.jit <torch/jit.html>`_
  * `torch.nn <torch/nn.html>`_
  * `torch.nn.functional <torch/nn/functional.html>`_
  * `torch.onnx <torch/onnx.html>`_
  * `torch.optim <torch/optim.html>`_
  * `torch.vision.ops <torch/vision/ops.html>`_
  * `torch.utils.dlpack <torch/utils/dlpack.html>`_

Integrations
------------

DALI
####

  *DALI* is a graph-based framework optimized for data loading,
  pre-processing and augmentation with cpu or gpu device.

  We extend it to work with our backend as an optional data provider,
  which brings a better performance for both training and inference.

  This integration involves the following components:

  * `dali <dali.html>`_
  * `dali.ops <dali/ops.html>`_

ONNX
####

  *ONNX* provides an open source format for AI models.

  We extend it to translate our *GraphIR* for model exchanging and deployment,
  that contributes a potential compatibility between inference runtimes.

  This integration involves the following components:

  * `onnx <onnx.html>`_

TensorRT
########

  *TensorRT* optimizes the AI models for high-performance inference.

  We extend it to provide a cuda inference runtime for *ONNX* models.

  This integration involves the following components:

  * `tensorrt <tensorrt.html>`_
  * `tensorrt.backend <tensorrt/backend.html>`_

Modules
-------

.. only:: html

  `Module autograph <dragon/autograph.html>`_
  : Native API for ``dragon.autograph`` namespace.

  `Module bitwise <dragon/bitwise.html>`_
  : Native API for ``dragon.bitwise`` namespace.

  `Module cuda <dragon/cuda.html>`_
  : Native API for ``dragon.cuda`` namespace.

  `Module distributed <dragon/distributed.html>`_
  : Native API for ``dragon.distributed`` namespace.

  `Module dlpack <dragon/dlpack.html>`_
  : Native API for ``dragon.dlpack`` namespace.

  `Module io <dragon/io.html>`_
  : Native API for ``dragon.io`` namespace.

  `Module logging <dragon/logging.html>`_
  : Native API for ``dragon.logging`` namespace.

  `Module losses <dragon/losses.html>`_
  : Native API for ``dragon.losses`` namespace.

  `Module math <dragon/math.html>`_
  : Native API for ``dragon.math`` namespace.

  `Module metrics <dragon/metrics.html>`_
  : Native API for ``dragon.metrics`` namespace.

  `Module nn <dragon/nn.html>`_
  : Native API for ``dragon.nn`` namespace.

  `Module optimizers <dragon/optimizers.html>`_
  : Native API for ``dragon.optimizers`` namespace.

  `Module random <dragon/random.html>`_
  : Native API for ``dragon.random`` namespace.

  `Module vision <dragon/vision.html>`_
  : Native API for ``dragon.vision`` namespace.

  `Module vm.caffe <caffe.html>`_
  : Virtual API for ``caffe`` namespace.

  `Module vm.caffe.layers <caffe/layers.html>`_
  : Virtual API for ``caffe.layers`` namespace.

  `Module vm.dali <dali.html>`_
  : Virtual API for ``dali`` namespace.

  `Module vm.dali.ops <dali/ops.html>`_
  : Virtual API for ``dali.ops`` namespace.

  `Module vm.onnx <onnx.html>`_
  : Virtual API for ``onnx`` namespace.

  `Module vm.tensorflow <tensorflow.html>`_
  : Virtual API for ``tensorflow`` namespace.

  `Module vm.tensorflow.bitwise <tensorflow/bitwise.html>`_
  : Virtual API for ``tensorflow.bitwise`` namespace.

  `Module vm.tensorflow.dtypes <tensorflow/dtypes.html>`_
  : Virtual API for ``tensorflow.dtypes`` namespace.

  `Module vm.tensorflow.initializers <tensorflow/initializers.html>`_
  : Virtual API for ``tensorflow.initializers`` namespace.

  `Module vm.tensorflow.keras <tensorflow/keras.html>`_
  : Virtual API for ``tensorflow.keras`` namespace.

  `Module vm.tensorflow.linalg <tensorflow/linalg.html>`_
  : Virtual API for ``tensorflow.linalg`` namespace.

  `Module vm.tensorflow.losses <tensorflow/losses.html>`_
  : Virtual API for ``tensorflow.losses`` namespace.

  `Module vm.tensorflow.math <tensorflow/math.html>`_
  : Virtual API for ``tensorflow.math`` namespace.

  `Module vm.tensorflow.nn <tensorflow/nn.html>`_
  : Virtual API for ``tensorflow.nn`` namespace.

  `Module vm.tensorflow.optimizers <tensorflow/optimizers.html>`_
  : Virtual API for ``tensorflow.optimizers`` namespace.

  `Module vm.tensorflow.random <tensorflow/random.html>`_
  : Virtual API for ``tensorflow.random`` namespace.

  `Module vm.tensorlayer.initializers <tensorlayer/initializers.html>`_
  : Virtual API for ``tensorlayer.initializers`` namespace.

  `Module vm.tensorlayer.layers <tensorlayer/layers.html>`_
  : Virtual API for ``tensorlayer.layers`` namespace.

  `Module vm.tensorlayer.models <tensorlayer/models.html>`_
  : Virtual API for ``tensorlayer.models`` namespace.

  `Module vm.tensorrt.backend <tensorrt/backend.html>`_
  : Virtual API for ``tensorrt.backend`` namespace.

  `Module vm.torch <torch.html>`_
  : Virtual API for ``torch`` namespace.

  `Module vm.torch.autograd <torch/autograd.html>`_
  : Virtual API for ``torch.autograd`` namespace.

  `Module vm.torch.distributed <torch/distributed.html>`_
  : Virtual API for ``torch.distributed`` namespace.

  `Module vm.torch.jit <torch/jit.html>`_
  : Virtual API for ``torch.jit`` namespace.

  `Module vm.torch.nn <torch/nn.html>`_
  : Virtual API for ``torch.nn`` namespace.

  `Module vm.torch.nn.functional <torch/nn/functional.html>`_
  : Virtual API for ``torch.nn.functional`` namespace.

  `Module vm.torch.onnx <torch/onnx.html>`_
  : Virtual API for ``torch.onnx`` namespace.

  `Module vm.torch.optim <torch/optim.html>`_
  : Virtual API for ``torch.optim`` namespace.

  `Module vm.torch.vision.ops <torch/vision/ops.html>`_
  : Virtual API for ``torch.vision.ops`` namespace.

  `Module vm.torch.utils.dlpack <torch/utils/dlpack.html>`_
  : Virtual API for ``torch.utils.dlpack`` namespace.

.. toctree::
  :hidden:

  dragon
  dragon/autograph
  dragon/bitwise
  dragon/cuda
  dragon/distributed
  dragon/dlpack
  dragon/io
  dragon/logging
  dragon/losses
  dragon/math
  dragon/metrics
  dragon/nn
  dragon/optimizers
  dragon/random
  dragon/vision
  caffe
  caffe/layers
  dali
  dali/ops
  onnx
  tensorflow
  tensorflow/bitwise
  tensorflow/dtypes
  tensorflow/initializers
  tensorflow/keras
  tensorflow/linalg
  tensorflow/losses
  tensorflow/math
  tensorflow/nn
  tensorflow/optimizers
  tensorflow/random
  tensorlayer/initializers
  tensorlayer/layers
  tensorlayer/models
  tensorrt
  tensorrt/backend
  torch
  torch/autograd
  torch/distributed
  torch/jit
  torch/nn
  torch/nn/functional
  torch/onnx
  torch/optim
  torch/vision/ops
  torch/utils/dlpack
