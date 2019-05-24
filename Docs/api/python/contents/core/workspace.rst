================
:mod:`Workspace`
================

.. toctree::
   :hidden:


Workspace
---------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Workspace(object)`_              A wrapper for the C implemented workspace.
`get_default_workspace`_          Return the current default workspace.
`reset_default_workspace`_        Reset the global default workspace.
==============================    =============================================================================

Tensor
------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`HasTensor`_                      Query whether tensor has registered in current workspace.
`CreateFiller`_                   Create the filler in the backend.
`GetTensorName`_                  Query the name represented in current workspace.
`SetTensorAlias`_                 Bind a alias to a existed tensor.
`FeedTensor`_                     Feed the values to the given tensor.
`FetchTensor`_                    Fetch the values of given tensor.
`ResetTensor`_                    Reset the memory of given tensor.
==============================    =============================================================================


Operator
--------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`RunOperator`_                    Run the operator in the VM backend.
==============================    =============================================================================


Graph
-----

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`CreateGraph`_                    Create the graph in the backend.
`RunGraph`_                       Run the specific graph.
==============================    =============================================================================

I/O
---

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Snapshot`_                       Snapshot tensors into a binary file.
`Restore`_                        Restore tensors from a binary file.
==============================    =============================================================================

API Reference
-------------

.. automodule:: dragon.core.workspace
    :members:
    :undoc-members:

.. autoclass:: Workspace
    :members:

    .. automethod:: __init__

.. _Workspace(object): #dragon.core.workspace.Workspace
.. _get_default_workspace: #dragon.core.workspace.get_default_workspace
.. _reset_default_workspace: #dragon.core.workspace.reset_default_workspace
.. _CreateGraph: #dragon.core.workspace.CreateGraph
.. _HasTensor: #dragon.core.workspace.HasTensor
.. _GetTensorName: #dragon.core.workspace.GetTensorName
.. _SetTensorAlias: #dragon.core.workspace.SetTensorAlias
.. _CreateFiller: #dragon.core.workspace.CreateFiller
.. _FetchTensor: #dragon.core.workspace.FetchTensor
.. _FeedTensor: #dragon.core.workspace.FeedTensor
.. _ResetTensor: #dragon.core.workspace.ResetTensor
.. _RunOperator: #dragon.core.workspace.RunOperator
.. _RunGraph: #dragon.core.workspace.RunGraph
.. _Snapshot: #dragon.core.workspace.Snapshot
.. _Restore: #dragon.core.workspace.Restore

.. _theano.function(*args, **kwargs): ../vm/theano/compile.html#dragon.vm.theano.compile.function.function