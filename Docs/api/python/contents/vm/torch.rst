============
:mod:`Torch`
============

Abstraction
-----------

|para| `PyTorch`_ provides straight-forward operations on research prototyping.

|para| We are aware that **Dragon** is a graph-based framework with strictly naming
for tensors, operators, and workspaces, while `Torch`_ is not.
A simple way to bridge their differences is **JIT**, which traces the anonymous expressions,
indicates a series of executions to the backend. If so, **AutoGrad** will just be a trick(Remember the *Chain Rule*).

|para| Rewriting the GC(*Garbage Collection*) is crucial in this role,
as the costly deconstruction on memories and operators must be avoided.
We could either persist a Operator(i.e. **Module**),
or reuse the several memories by turns(i.e. **MemoryPool**), if naming them formally.

|para| We are still working hard to cover the original PyTorch operators,
however, a bunch of extended operators in many other frameworks can be used.
Our **PyTorch** will be unique and more powerful than the official one.

Related Work
------------

|paratitle| **Proto-based Intermediate Representation**

|para| Recent years, several powerful frameworks choose the ProtocolBuffer to
describe the operators with various arguments, including `Caffe`_, `Caffe2`_, `TensorFlow`_, and `ONNX`_.
The most important reason is that, these descriptors can be easily serialized and sent to the backend.
With the help of **Factory Pattern**, we have had an elegant way to dispatch the executions, while not
call them imperatively. This way is also known as the **Declarative Programming**.

|para| Attaching the IR(Intermediate Representation) takes the following advantages:

* Traceable pipelines, much helpful for visualizing and debugging.

* Deterministic executions, detailed optimization can be applied.

* Efficient deployments, data-flows has been well organized.

|para| A good news is that, we can reduce the overhead of IR below 5% of computation time,
which means the dynamic graph could work as fast as the static graph while retain the flexibility.

|paratitle| **Caffe2**

|para| We have noticed that some developers discouraged the **Declarative Programming** in 2017 and early 2018,
due to the counter-intuitive building of computation graph. Actually, `Caffe2`_ had published Operator-Wise execution
(a.k.a, *workspace.RunOperator()*) since 2016. In other words, **Imperative Programming** is the subset of **Declarative Programming**,
if we process the declaration implicitly. This mechanism is sometimes called **JIT** by someone.

Architectures
-------------

.. toctree::
   :hidden:

.. _Torch: http://torch.ch
.. _PyTorch: https://pytorch.org
.. _Caffe: http://caffe.berkeleyvision.org
.. _Caffe2: http://caffe2.ai
.. _TensorFlow: https://www.tensorflow.org
.. _ONNX: https://onnx.ai

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


