==========
:mod:`Net`
==========

.. toctree::
   :hidden:

Quick Shortcut
--------------

=========================   =============================================================================
List                        Brief
=========================   =============================================================================
`Net.copy_from`_            Copy the parameters from the binary proto file.
`Net.forward`_              Forward Pass.
`Net.backward`_             Backward Pass.
`Net.function`_             Forward + Backward Pass.
`Net.save`_                 Save the parameters into a binary file.
`Net.blobs`_                Return the blobs.
`Net.params`_               Return the parameters.
`Net.trainable_params`_     Return the trainable parameters.
`Net.inputs`_               Return the inputs of net.
`Net.outputs`_              Return the outputs of net.
=========================   =============================================================================

API Reference
-------------

.. currentmodule:: dragon.vm.caffe.net

.. autoclass:: Net
    :members:

    .. automethod:: __init__

.. autoclass:: PartialNet
    :members:

.. _Net.copy_from: #dragon.vm.caffe.net.Net.copy_from
.. _Net.forward: #dragon.vm.caffe.net.Net.forward
.. _Net.backward: #dragon.vm.caffe.net.Net.backward
.. _Net.save: #dragon.vm.caffe.net.Net.save
.. _Net.blobs: #dragon.vm.caffe.net.Net.blobs
.. _Net.params: #dragon.vm.caffe.net.Net.params
.. _Net.trainable_params: #dragon.vm.caffe.net.Net.trainable_params
.. _Net.inputs: #dragon.vm.caffe.net.Net.inputs
.. _Net.outputs: #dragon.vm.caffe.net.Net.outputs
.. _Net.function: #dragon.vm.caffe.net.Net.function

.. _NetInit(proto_txt, phase): #dragon.vm.caffe.net.Net.NetInit
.. _NetInitLoad(proto_txt, model, phase): #dragon.vm.caffe.net.Net.NetInitLoad
.. _workspace.Snapshot(*args, **kwargs): ../../core/workspace.html#dragon.core.workspace.Snapshot
.. _workspace.Restore(*args, **kwargs): ../../core/workspace.html#dragon.core.workspace.Restore

.. _Net_Init(_caffe.cpp, L109): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L109
.. _Net_Init_Load(_caffe.cpp, L137): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L137
.. _FilterNet(net.cpp, L259): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/net.cpp#L259
.. _Init(net.cpp, L44): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/net.cpp#L44
.. _ForwardBackward(net.cpp, L85): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/include/caffe/net.hpp#L85
.. _CopyTrainedLayersFromBinaryProto(net.cpp, L780): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/net.cpp#L780
.. _Net_forward(pycaffe.py, L88): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L88
.. _Net_backward(pycaffe.py, L137): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L137
.. _Net_Save(_caffe.cpp, L153): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/_caffe.cpp#L153
.. _Net_blobs(pycaffe.py, L25): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L25
.. _Net_params(pycaffe.py, L58): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L58
.. _Net_inputs(pycaffe.py, L73): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L73
.. _Net_outputs(pycaffe.py, L81): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/python/caffe/pycaffe.py#L81

.. _theano.function(*args, **kwargs): ../theano/compile.html#dragon.vm.theano.compile.function.function