=============
:mod:`Theano`
=============


Abstraction
-----------

|para| `Theano`_ is an inception of the modern deep learning frameworks.

|para| We find that the fundamentals of `compile`_ are quite useful to wrap the primitives of computation graph,
thus, instead of defining any primitives that are hard to remember, we still prefer the `Theano`_.

|para| We implement it based on the backend of Dragon thorough native Python language.
All operators are compiled offline, which is more similar to `TensorFlow`_ but not the original `Theano`_.

|para| Our implemented **Theano** is also easy to debug or extend, it could be a substitution for supporting
the early codes of Deep Learning.


Related Work
------------

|paratitle| **Caffe**

|para| `Caffe`_ takes the fine-grained layers to form the most operations of Deep Learning.

|para| Comparing with the real-time compiling, **Factory Pattern** provides a faster method
to create the computation graph based on the uncertain topology
while lacks optimizations both on the time and space. We proposed several simple algorithms
to optimize the raw graph by taking the text format topology of `protobuf`_,
which contributes an efficient `compile`_ module for **Theano**.

|paratitle| **Keras**

|para| `Keras`_ is smart enough to invent new fine-grained apis to unify various backends.

|para| We DO NOT follow it because the computations performed by different backends are confused.
Besides, the efforts to learn `Keras`_ are also expensive.

|para| Our implemented **Theano** completely takes the advantages of `VirtualTensor`_, while
still could be as light as `Keras`_.(*LESS THAN 2000 LINES!*)


Architectures
-------------

.. toctree::
   :hidden:

   theano/compile
   theano/tensor

|paratitle| **Compile**

|para| This module consists several crucial primitives to drive the backend of Dragon.
We find these primitives useful through injecting the intentions of programming
(such as **Feed/Fetch**, **Net Topology**, and **Control Flow**)
into the virtual machine, instead of explicitly declaring and creating.

|sectitle| □ |nbsp| `Function`_

|para| Unlike `Caffe2`_, we prefer this primitive to declaring a graph and stuffing operators,
as a graph can be confirmed and further optimized if only the inputs and outputs are deterministic.

|para| We also remove the naive arithmetic update operations, use the fine-grained `Updater`_ instead.
Our implemented `Updater`_ provides the speedup for large-scale distributed training,
which will enhance all frameworks in the VirtualBox.

|sectitle| □ |nbsp| `Shared`_

|para| This primitive is a simple wrapper of `FeedTensor`_.

|para| We remove the mechanism of `SharedVaraible`_ due to the **memory-storage** is taken by the backend.
Following the `Caffe2`_ and `TensorFlow`_, we attribute it to the **Feed** of data streams.

|sectitle| □ |nbsp| `Scan`_

|para| We use this primitive to create the dynamic computation graphs.

|para| By taking a template of the sub-graph, `Scan`_ unfolds it for specific loop steps,
which is very useful to model sentence-level **Recurrent Neural Networks**.

|context| For detailed Documentation, see: `Compile`_.

|paratitle| **Tensor**

|para| This module provides massive methods based on **Tensor**.

|para| The structure itself is obscure in `Theano`_, while **Symbolic Variable** is more frequently represented.
We simplify the messy representations by the unified `VirtualTensor`_ structure,
that **Tensor** is generally represented any n-dimensions arrays.

|para| All methods in this module will share the `Ops`_ library of Dragon.
We are sorry for removing some odd implementations supported by the original `Theano`_.

|para| Inheriting from the hierarchy of `Theano`_, we categorize these methods as follows:

|sectitle| □ |nbsp| `Basic`_

|sectitle| □ |nbsp| `Variable`_

|sectitle| □ |nbsp| `Initializer`_

|sectitle| □ |nbsp| `Operator`_

|sectitle| □ |nbsp| `NNet`_

|context| For detailed Documentation, see: `Tensor`_.


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

.. _Theano: http://deeplearning.net/software/theano
.. _TensorFlow: http://www.tensorflow.org
.. _compile: theano/compile.html
.. _Tensor: theano/tensor.html
.. _VirtualTensor: ../core/tensor.html
.. _protobuf: https://github.com/google/protobuf
.. _Caffe: caffe.html
.. _Caffe2: https://caffe2.ai
.. _Keras: https://keras.io

.. _Updater: ../updaters.html
.. _Ops: ../ops.html

.. _Function: theano/compile.html#dragon.vm.theano.compile.function.function
.. _Shared: theano/compile.html#dragon.vm.theano.compile.sharedvalue.shared
.. _Scan: theano/compile.html#dragon.vm.theano.compile.scan.scan
.. _FeedTensor: ../core/workspace.html#dragon.core.workspace.FeedTensor
.. _SharedVaraible: http://deeplearning.net/software/theano/library/compile/shared.html

.. _Basic: theano/tensor.html#basic
.. _Variable: theano/tensor.html#variable
.. _Initializer: theano/tensor.html#initializer
.. _Operator: theano/tensor.html#operator
.. _NNet: theano/tensor.html#nnet

