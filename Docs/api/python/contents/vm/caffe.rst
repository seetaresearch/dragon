============
:mod:`Caffe`
============


Abstraction
-----------

|para| `Caffe`_ is one of the most famous deep learning framework for **Computer Vision**.

|para| However, it seems to be stuck due to the the poor memory optimization against very deep neural nets.
Besides, lacking of flexible automatic differentiation, it is awful to model multi-nets algorithms,
such as `GAN`_ and `Deterministic PolicyGradient`_.

|para| We implement it based on the backend of Dragon thorough native Python language.
Our work is very different from the official Python wrappers, a.k.a, the **PyCaffe**, which comes from
the exports of `BoostPython`_ based on C++ language.

|para| We demonstrate that even completely drop the **Net** topology proposed by `Jia et al. (2014)`_,
any **Graph** based deep learning frameworks(*Dragon*, *Theano*, *MXNet*, or *TensorFlow*)
can inherit almost specifics from `Caffe`_ also.

Related Work
------------

|paratitle| **Fast RCNN**

|para| Inspired by the `SolverWrapper`_, which disabled and reimplemented the
optimization process for object detection system, we tried to implement `Solver`_ completely in Python environment.
Benefited from the flexible **Updater** designed in the backend, we accomplished it with several python codes,
specializing in **Dynamic Learning Rate**, **Custom Training Loops**, and **Cross Validation**.

|paratitle| **Theano**

|para|  The **Net** used in `Caffe`_ is redundant, thus,
we propagate the data flows in the **Graph** equivalently.
Specifically, we use `theano.function(inputs, outputs)`_ to collect the inputs and outputs of `Net`_,
then, generate the **Computation Graph** descriptions for the backend to optimize.
Note that all DL frameworks in **VirtualBox** share the same descriptions,
leading the results to be deterministic and reproduceable.

|paratitle| **Keras**

|para| Referring to **Keras**, whose API is designed to wrap existing backends,
we **IMPLEMENT** most `Layers`_ of `Caffe`_ without any efforts.
Our backend has provided both the latest and optimized deep learning operators,
that outperforms the original `Caffe`_ or other forked ones.

Architectures
-------------

.. toctree::
   :hidden:

   caffe/layer
   caffe/net
   caffe/solver
   caffe/misc


|paratitle| **Layer**

|para| **Layer** is the basic structure for parsing text format definition.
By taking the advantages of python based `protobuf`_,
we can parse and document parameters of various layers.
Following the `Caffe`_ that before dismantling headers(such as `62ed0d2`_),
we categorize these layers into six sections:

|sectitle| □ |nbsp| `Data`_ - How to prefetch data before feeding to the Net.

|sectitle| □ |nbsp| `Vision`_ - Vision relevant layers, such as Convolution and Pooling.

|sectitle| □ |nbsp| `Neuron`_ - Neuron relevant layers, most of those are activations.

|sectitle| □ |nbsp| `Common`_ - Layers using BLAS or applying on NDArray.

|sectitle| □ |nbsp| `Loss`_ - Several frequently used loss functions.

|sectitle| □ |nbsp| `MPI`_ - MPI relevant layers, for model parallelism.

|context| For detailed Documentation, see: `Layer`_.


|paratitle| **Net**

|para| **Net** supports the most exporting interfaces used in **PyCaffe**.
We implement it completely in the python environment, which provides much conveniences,
especially when extending the modern architectures of **Convolutional Neural Networks**.

|context| For detailed Documentation, see: `Net`_.


|paratitle| **Solver**

|para| **Solver** merges updates to optimize the `Net`_.
We simplified it from the original C++ implementation, that brings more unconstrained tricks
for tuning hyper parameters, e.g., the learning rate.

|context| For detailed Documentation, see: `Solver`_.

|paratitle| **Misc**

|para| The internal settings(e.g. GPU, Random Seed) can be configured everywhere by singleton,
we contribute this to the designing of `Global Context`_, which was also taken by `Girshick (2015)`_.
In order to implement it, we bind these settings to the `dragon.config`_.

|context| For detailed Documentation, see: `Misc`_.


.. _Caffe: http://caffe.berkeleyvision.org
.. _62ed0d2: https://github.com/BVLC/caffe/tree/62ed0d2bd41a730397e718bae4354c9a5a722624
.. _BoostPython: http://www.boost.org/
.. _Jia et al. (2014): https://arxiv.org/abs/1408.5093
.. _Girshick (2015): https://github.com/rbgirshick/fast-rcnn/blob/master/lib/fast_rcnn/config.py
.. _GAN: https://arxiv.org/abs/1406.2661
.. _Deterministic PolicyGradient: http://proceedings.mlr.press/v32/silver14.html
.. _SolverWrapper: https://github.com/rbgirshick/fast-rcnn/blob/e68366925d18fde83e865b894022d1d278f3f758/lib/fast_rcnn/train.py#L20
.. _Global Context: https://github.com/BVLC/caffe/blob/ef2eb4b9369e4c0db5cfc92cc9c8e4d4497d421e/include/caffe/common.hpp#L102
.. _protobuf: https://github.com/google/protobuf
.. _dragon.config: ../config.html
.. _theano.function(inputs, outputs): theano/compile.html#dragon.vm.theano.compile.function.function
.. _Solver: caffe/solver.html
.. _Net: caffe/net.html
.. _Misc: caffe/misc.html
.. _Layers: caffe/layer.html
.. _Layer: caffe/layer.html
.. _Data: caffe/layer.html#data
.. _Vision: caffe/layer.html#vision
.. _Neuron: caffe/layer.html#neuron
.. _Common: caffe/layer.html#common
.. _Loss: caffe/layer.html#loss
.. _MPI: caffe/layer.html#mpi

.. |nbsp| raw:: html

    &nbsp

.. |br| raw:: html

    <br />

.. |paratitle| raw:: html

    <p style="font-size: 20px">

.. |sectitle| raw:: html

    <p style="text-indent:1em; font-size: 18px">

.. |para| raw:: html

    <p style="text-indent:1.5em; font-size: 18px; max-width: 830px;">

.. |context| raw:: html

    <p style="font-size: 18px; max-width: 830px;">


