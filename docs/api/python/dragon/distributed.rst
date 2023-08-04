dragon.distributed
==================

.. only:: html

  Functions
  ---------

  `all_gather(...) <distributed/all_gather.html>`_
  : Gather input across all nodes.

  `all_reduce(...) <distributed/all_reduce.html>`_
  : Reduce input across all nodes.

  `broadcast(...) <distributed/broadcast.html>`_
  : Broadcast input from root node.

  `is_cncl_available(...) <distributed/is_cncl_available.html>`_
  : Return whether the CNCL backend is available.

  `is_initialized(...) <distributed/is_initialized.html>`_
  : Return whether the distributed environment is initialized.

  `is_mpi_available(...) <distributed/is_mpi_available.html>`_
  : Return whether the MPI backend is available.

  `is_nccl_available(...) <distributed/is_nccl_available.html>`_
  : Return whether the NCCL backend is available.

  `get_backend(...) <distributed/get_backend.html>`_
  : Return the backend of given process group.

  `get_group(...) <distributed/get_group.html>`_
  : Return the current default process group.

  `get_rank(...) <distributed/get_rank.html>`_
  : Return the rank of current process.

  `get_world_size(...) <distributed/get_world_size.html>`_
  : Return the world size of environment.

  `new_group(...) <distributed/new_group.html>`_
  : Create a new communication group.

  `reduce_scatter(...) <distributed/reduce_scatter.html>`_
  : Reduce and scatter input across all nodes.

.. toctree::
  :hidden:
  
  distributed/all_gather
  distributed/all_reduce
  distributed/broadcast
  distributed/is_cncl_available
  distributed/is_initialized
  distributed/is_mpi_available
  distributed/is_nccl_available
  distributed/get_backend
  distributed/get_group
  distributed/get_rank
  distributed/get_world_size
  distributed/new_group
  distributed/reduce_scatter

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
