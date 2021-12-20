cudnn
=====

Properties
----------

allow_tf32
##########
.. data:: dragon.vm.torch.backends.cudnn.allow_tf32
  :annotation: = False

  The flag that allows cuDNN TF32 math type or not.

benchmark
#########
.. data:: dragon.vm.torch.backends.cudnn.benchmark
  :annotation: = False

  The flag that benchmarks fastest cuDNN algorithms or not.

deterministic
#############
.. data:: dragon.vm.torch.backends.cudnn.deterministic
  :annotation: = False

  The flag that selects deterministic cuDNN algorithms or not.

enabled
#######
.. data:: dragon.vm.torch.backends.cudnn.enabled
  :annotation: = True

  The flag that uses cuDNN or not.

Functions
---------

is_available
############
.. automethod:: dragon.vm.torch.backends.cudnn.is_available

version
#######
.. automethod:: dragon.vm.torch.backends.cudnn.version

.. raw:: html

  <style>
  h1:before {
    content: "torch.backends.";
    color: #103d3e;
  }
  </style>
