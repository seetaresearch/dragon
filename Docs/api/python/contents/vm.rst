================
:mod:`dragon.vm`
================

Overview
--------

|para| Although the proposed `TinyDragon`_ can contribute a framework, we still desire it to be humanized.
An interesting idea is that: basic primitives of `Theano`_ can be used for building `Caffe`_ or `TensorFlow`_,
thus, these modern frameworks can share a common backend if providing enough operator kernels.

|para| In this section, we demonstrate a cross-frameworks frontend is feasible, and further more, will get
benefit from all participating crucial interfaces especially when one is not reasonable.

VirtualBox
----------

.. toctree::
   :hidden:

   vm/caffe
   vm/theano
   vm/torch

====================    ====================================================================================
List                    Brief
====================    ====================================================================================
`Theano`_               **Theano** is an inception of the modern deep learning frameworks.
`Caffe`_                **Caffe** is one of the most famous deep learning framework for Computer Vision.
`PyTorch`_              **PyTorch** provides straight-forward operations on research prototyping.
====================    ====================================================================================

.. |para| raw:: html

    <p style="text-indent:1.5em; font-size: 18px; max-width: 830px;">

.. _TinyDragon: ../index.html#tinydragon
.. _Theano:  vm/theano.html
.. _Caffe: vm/caffe.html
.. _PyTorch: vm/torch.html
.. _TensorFlow: ../index.html#tensorflow

