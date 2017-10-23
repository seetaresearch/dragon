================
:mod:`Workspace`
================

.. toctree::
   :hidden:

Tensor
------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`HasTensor`_                      Query whether tensor has registered in current workspace.
`GetTensorName`_                  Query the name represented in current workspace.
`CreateFiller`_                   Create the filler in the backend.
`FetchTensor`_                    Fetch the values of given tensor.
`FeedTensor`_                     Feed the values to the given tensor.
==============================    =============================================================================

Graph
-----

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`CreateGraph`_                    Create the graph in the backend.
`RunGraph`_                       Run the specific graph.
==============================    =============================================================================

Misc
----

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Snapshot`_                       Snapshot tensors into a binary file.
`Restore`_                        Restore tensors from a binary file.
`SwitchWorkspace`_                Switch to the specific Workspace.
`PrintRawGraphDef`_               Print the raw prototxt.
`PrintOptimizedGraph`_            Print the optimized prototxt.
`WriteOptimizedGraph`_            Generate the optimized prototxt into a file.
==============================    =============================================================================

API Reference
-------------

.. automodule:: dragon.core.workspace
    :members:
    :undoc-members:
    :show-inheritance:

.. _SwitchWorkspace: #dragon.core.workspace.SwitchWorkspace
.. _CreateGraph: #dragon.core.workspace.CreateGraph
.. _HasTensor: #dragon.core.workspace.HasTensor
.. _GetTensorName: #dragon.core.workspace.GetTensorName
.. _CreateFiller: #dragon.core.workspace.CreateFiller
.. _FetchTensor: #dragon.core.workspace.FetchTensor
.. _FeedTensor: #dragon.core.workspace.FeedTensor
.. _RunGraph: #dragon.core.workspace.RunGraph
.. _Snapshot: #dragon.core.workspace.Snapshot
.. _Restore: #dragon.core.workspace.Restore
.. _PrintRawGraphDef: #dragon.core.workspace.PrintRawGraphDef
.. _PrintOptimizedGraph: #dragon.core.workspace.PrintOptimizedGraph
.. _WriteOptimizedGraph: #dragon.core.workspace.WriteOptimizedGraph

.. _theano.function(*args, **kwargs): ../vm/theano/compile.html#dragon.vm.theano.compile.function.function