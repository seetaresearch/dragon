dragon.distributed
==================

.. only:: html

  Functions
  ---------

  `all_reduce(...) <distributed/all_reduce.html>`_
  : Reduce the input across all nodes in a group.

  `broadcast(...) <distributed/broadcast.html>`_
  : Broadcast the input from root node in a group.

  `is_initialized(...) <distributed/is_initialized.html>`_
  : Whether the distributed environment has initialized.

  `init(...) <distributed/init.html>`_
  : Initialize the distributed environment.

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

.. toctree::
  :hidden:
  
  distributed/all_reduce
  distributed/broadcast
  distributed/init
  distributed/is_initialized
  distributed/get_backend
  distributed/get_group
  distributed/get_rank
  distributed/get_world_size
  distributed/new_group

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
