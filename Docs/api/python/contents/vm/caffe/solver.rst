=============
:mod:`Solver`
=============

.. toctree::
   :hidden:

Quick Shortcut
--------------

====================    =============================================================================
List                    Brief
====================    =============================================================================
`Solver.step`_          Step the train net.
`Solver.snapshot`_      Snapshot the parameters of train net.
`Solver.net`_           Return the train net.
`Solver.test_nets`_     Return the test nets.
`Solver.iter`_          Return or Set the current iteration.
`Solver.lr`_            Return or Set the current learning rate.
====================    =============================================================================

API Reference
-------------

.. currentmodule:: dragon.vm.caffe.solver

.. autoclass:: Solver
    :members:

    .. automethod:: __init__

.. _Net.function(*args, **kwargs): net.html#dragon.vm.caffe.net.Net.function
.. _workspace.Snapshot(*args, **kwargs): ../../core/workspace.html#dragon.core.workspace.Snapshot

.. _Solver.step: #dragon.vm.caffe.solver.Solver.step
.. _Solver.snapshot: #dragon.vm.caffe.solver.Solver.snapshot
.. _Solver.net: #dragon.vm.caffe.solver.Solver.net
.. _Solver.test_nets: #dragon.vm.caffe.solver.Solver.test_nets
.. _Solver.iter: #dragon.vm.caffe.solver.Solver.iter
.. _Solver.lr: #dragon.vm.caffe.solver.Solver.lr

.. _[LeCun et.al, 1998]: http://yann.lecun.com/exdb/publis/#lecun-98b
.. _[Sutskever et.al, 2012]: http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf
.. _[Hinton et.al, 2013]: http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf
.. _[Kingma & Ba, 2014]: https://arxiv.org/abs/1412.6980

.. _SolverWrapper: https://github.com/rbgirshick/py-faster-rcnn/blob/4e199d792f625cf712ca9b9a16278bafe0806201/lib/fast_rcnn/train.py#L20``
.. _InitTrainNet(solver.cpp, L63): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solver.cpp#L63
.. _InitTestNets(solver.cpp, L104): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solver.cpp#L104
.. _GetLearningRate(solver.cpp, L27): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solvers/sgd_solver.cpp#L27
.. _Step(solver.cpp, L180): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solver.cpp#L180
.. _Test(solver.cpp, L328): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solver.cpp#L328
.. _Snapshot(solver.cpp, L403): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/solver.cpp#L403
.. _SolverParameter.base_lr: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L146
.. _SolverParameter.momentum: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L174
.. _SolverParameter.momentum2: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L217
.. _SolverParameter.rms_decay: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L221
.. _SolverParameter.delta: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L215


.. autoclass:: SGDSolver
    :members:

.. autoclass:: NesterovSolver
    :members:

.. autoclass:: RMSPropSolver
    :members:

.. autoclass:: AdamSolver
    :members: