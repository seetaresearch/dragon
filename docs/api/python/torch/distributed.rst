vm.torch.distributed
====================

.. only:: html

  Functions
  ---------

  `all_gather(...) <distributed/all_gather.html>`_
  : Gather tensor across all nodes and output to a tensor list.

  `all_gather_into_tensor(...) <distributed/all_gather_into_tensor.html>`_
  : Gather tensor across all nodes and output to a tensor.

  `all_reduce(...) <distributed/all_reduce.html>`_
  : Reduce tensor across all nodes.

  `broadcast(...) <distributed/broadcast.html>`_
  : Broadcast tensor from the source node.

  `reduce_scatter(...) <distributed/reduce_scatter.html>`_
  : Reduce and scatter the tensor list across all nodes.

  `reduce_scatter_tensor(...) <distributed/reduce_scatter_tensor.html>`_
  : Reduce and scatter the tensor across all nodes.

.. toctree::
  :hidden:

  distributed/all_gather
  distributed/all_gather_into_tensor
  distributed/all_reduce
  distributed/broadcast
  distributed/reduce_scatter
  distributed/reduce_scatter_tensor

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
