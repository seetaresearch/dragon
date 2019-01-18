===========
:mod:`Misc`
===========

.. toctree::
   :hidden:

Quick Reference
---------------

=========================      ============================================================================
List                           Brief
=========================      ============================================================================
`caffe.set_mode_cpu`_          Set to the CPU mode.
`caffe.set_mode_gpu`_          Set to the GPU mode.
`caffe.set_device`_            Set the active device.
`caffe.set_random_seed`_       Set the global random seed.
`caffe.root_solver`_           Whether this node is root.
`caffe.set_root_solver`_       Set this node to the root.
=========================      ============================================================================

API Reference
-------------

.. automodule:: dragon.vm.caffe.misc
    :members:

.. _caffe.set_mode_cpu: #dragon.vm.caffe.misc.set_mode_cpu
.. _caffe.set_mode_gpu: #dragon.vm.caffe.misc.set_mode_gpu
.. _caffe.set_device: #dragon.vm.caffe.misc.set_device
.. _caffe.set_random_seed: #dragon.vm.caffe.misc.set_random_seed
.. _caffe.root_solver: #dragon.vm.caffe.misc.root_solver
.. _caffe.set_root_solver: #dragon.vm.caffe.misc.set_root_solver

.. _set_mode_cpu(_caffe.cpp, L51): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L51
.. _set_mode_gpu(_caffe.cpp, L52): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L52
.. _SetDevice(common.cpp, L65): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/common.cpp#L65
.. _set_random_seed(_caffe.cpp, L71): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L71
.. _root_solver(common.hpp, L164): https://github.com/BVLC/caffe/blob/b9ea0267851ccc7f782327408fe7953ba0f13c53/include/caffe/common.hpp#L164
.. _set_root_solver(common.hpp, L165): https://github.com/BVLC/caffe/blob/b9ea0267851ccc7f782327408fe7953ba0f13c53/include/caffe/common.hpp#L165
