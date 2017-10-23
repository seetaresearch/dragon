Dragon - Python APIs
====================

Dragon is a computation graph based distributed deep learning framework.

For using it, import as follows:

.. code-block:: python

   import dragon

Style Orientation
-----------------

However, it will not help you much because Dragon is designed without systemic APIs.

We have extended it with **FOUR** Programming Styles:

TinyDragon
##########

    *TinyDragon* takes a very light weight programming style.

    Our goal is to reduce extra unnecessary structures or interfaces. Therefore,
    in addition to feed or fetch, the last thing is designing a objective function through available operators.

    It mainly uses the following components:

    * `Tensor`_
    * `Operators`_
    * `Updaters`_
    * `Workspace`_
    * `Function`_
    * `Grad`_

Caffe
#####

    *Caffe* is implemented basing on the backend of Dragon thorough native Python language.

    Our work is very different from the official Python wrappers, a.k.a, the **PyCaffe**, which comes from
    the exports of **BoostPython** based on C++ language.

    It mainly uses the following components:

    * `Tensor`_
    * `Layer`_
    * `Net`_
    * `Solver`_
    * `Misc`_

Theano
######

    *Theano* is an inception of the modern deep learning frameworks.

    We implement it based on the backend of Dragon thorough native Python language.
    All operators are compiled offline, which is more similar to *TensorFlow* but not the original *Theano*.

    It mainly uses the following components:

    * `Tensor`_
    * `T`_
    * `Updaters`_
    * `Function`_
    * `Grad`_

TensorFlow
##########

     COMING SOON......


Table of Contents
-----------------

.. toctree::
   :hidden:

   contents/config
   contents/ops
   contents/updaters
   contents/memonger
   contents/core
   contents/io
   contents/operators
   contents/vm
   contents/tools

Quick Shortcut
##############

=====================    ======================================================================
`dragon.config`_         The global configures.
`dragon.ops`_            The exhibition of available operators.
`dragon.updaters`_       The exhibition of available updaters.
`dragon.memonger`_       The extreme memory optimizer.
=====================    ======================================================================

Packages
########

=====================      =====================================================================
`dragon.core`_             The core package.
`dragon.io`_               The io package.
`dragon.operators`_        The operators package.
`dragon.tools`_            The tools package.
`dragon.vm`_               The vm package.
=====================      =====================================================================

.. _dragon.config: contents/config.html
.. _dragon.ops: contents/ops.html
.. _dragon.core: contents/core.html
.. _dragon.core.tensor.Tensor: contents/core/tensor.html
.. _dragon.core.workspace: contents/core/workspace.html
.. _dragon.io:   contents/io.html
.. _dragon.operators: contents/operators.html
.. _dragon.updaters: contents/updaters.html
.. _dragon.memonger: contents/memonger.html
.. _dragon.tools: contents/tools.html
.. _dragon.vm: contents/vm.html

.. _Tensor: contents/core/tensor.html
.. _Workspace: contents/core/workspace.html
.. _Operators: contents/ops.html
.. _Updaters: contents/updaters.html

.. _Layer: contents/vm/caffe/layer.html
.. _Net: contents/vm/caffe/net.html
.. _Solver: contents/vm/caffe/solver.html
.. _Misc: contents/vm/caffe/misc.html

.. _Function: contents/vm/theano/compile.html#dragon.vm.theano.compile.function.function
.. _Grad: contents/vm/theano/tensor.html#dragon.vm.theano.gradient.grad
.. _T: contents/vm/theano/tensor.html

.. _Caffe: contents/vm/caffe.html


.. |br| raw:: html

    <br />