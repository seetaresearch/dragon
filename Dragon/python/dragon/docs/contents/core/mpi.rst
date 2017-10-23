==========
:mod:`MPI`
==========

.. toctree::
   :hidden:

Basic
-----

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Init`_                           Init the MPI env.
`Is_Init`_                        Whether the MPI env has initialized.
`Rank`_                           The world rank of current MPI node.
`Size`_                           The world size of current MPI env.
`CreateGroup`_                    Construct a MPIGroup with specific members.
`Finalize`_                       Finalize the MPI env.
==============================    =============================================================================

Parallelism
-----------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Snapshot`_                       Set the specific MPI nodes to snapshot.
`Parallel`_                       Set the specific MPI nodes for data parallelism.
`AllowSnapshot`_                  Whether this node can snapshot.
`AllowParallel`_                  Whether this node was set for data parallelism.
`SetParallelMode`_                Set the mode of data parallelism.
`GetParallelMode`_                Get the current mode of data parallelism.
==============================    =============================================================================

.. automodule:: dragon.core.mpi
    :members:

.. _Init: #dragon.core.mpi.Init
.. _Is_Init: #dragon.core.mpi.Is_Init
.. _Rank: #dragon.core.mpi.Rank
.. _Size: #dragon.core.mpi.Size
.. _CreateGroup: #dragon.core.mpi.CreateGroup
.. _Finalize:  #dragon.core.mpi.Finalize

.. _Snapshot: #dragon.core.mpi.Snapshot
.. _Parallel: #dragon.core.mpi.Parallel
.. _AllowSnapshot: #dragon.core.mpi.AllowSnapshot
.. _AllowParallel: #dragon.core.mpi.AllowParallel
.. _SetParallelMode: #dragon.core.mpi.SetParallelMode
.. _GetParallelMode: #dragon.core.mpi.GetParallelMode

.. _workspace.Snapshot(*args, **kwargs): workspace.html#dragon.core.workspace.Snapshot