==================
:mod:`dragon.core`
==================

Data Structure
--------------

.. toctree::
   :hidden:

   core/tensor
   core/scope

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`dragon.core.scope`_                The Scope and Namespace.
`dragon.core.tensor`_               The basic structure of VM.
==============================      =======================================================================

C++ Binding Wrapper
-------------------

.. toctree::
   :hidden:

   core/workspace
   core/tensor_utils
   core/mpi
   core/gradient_maker

==============================      =======================================================================
List                                Brief
==============================      =======================================================================
`dragon.core.workspace`_            The interfaces of Workspace, mostly are the wrappers of C++.
`dragon.core.gradient_maker`_       The generator of GradientOps.
`dragon.core.tensor_utils`_         The Tensor utilities.
`dragon.core.mpi`_                  The MPI utilities.
==============================      =======================================================================

.. _dragon.core.mpi: core/mpi.html
.. _dragon.core.scope: core/scope.html
.. _dragon.core.tensor: core/tensor.html
.. _dragon.core.tensor_utils: core/tensor_utils.html
.. _dragon.core.workspace: core/workspace.html
.. _dragon.core.gradient_maker: core/gradient_maker.html