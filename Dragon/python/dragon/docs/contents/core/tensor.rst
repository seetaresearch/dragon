=============
:mod:`Tensor`
=============

.. toctree::
   :hidden:

Quick Shortcut
--------------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Tensor.name`_                    Return or Set the name.
`Tensor.shape`_                   Return or Set the shape.
`Tensor.dtype`_                   Return or Set the data type.
`Tensor.set_value`_               Feed the values to C++ backend.
`Tensor.get_value`_               Fetch the values from C++ backend.
`Tensor.copy`_                    Return a Tensor with same content.
`Tensor.reshape`_                 Reshape the dimensions of input.
`Tensor.dimshuffle`_              Shuffle the dimen`sions.
`Tensor.CreateOperator`_          Construct a new Tensor with specific operator descriptor.
`Tensor.Fill`_                    Fill self with the specific type of filler.
`Tensor.PrintExpressions`_        Return the stringified internal expressions.
==============================    =============================================================================

Register
--------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Tensor.Variable`_                Register as an empty variable.
`Tensor.Placeholder`_             Register as a placeholder.
`Tensor.Constant`_                Register as a variable with constant initializer.
`Tensor.Uniform`_                 Register as a variable with uniform initializer.
`Tensor.Normal`_                  Register as a variable with normal initializer.
`Tensor.TruncatedNormal`_         Register as a variable with truncated normal initializer.
`Tensor.Gaussian`_                Register as a variable with gaussian initializer.
`Tensor.Xavier`_                  Register as a variable with xavier initializer.
`Tensor.MSRA`_                    Register as a variable with msra initializer.
`Tensor.GlorotUniform`_           Register as a variable with glorot uniform initializer.
`Tensor.GlorotNormal`_            Register as a variable with glorot normal initializer.
==============================    =============================================================================

Override
--------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Tensor.__add__`_                 x.__add__(y) <=> x + y
`Tensor.__radd__`_                 x.__radd__(y) <=> y + x
`Tensor.__sub__`_                 x.__sub__(y) <=> x - y
`Tensor.__rsub__`_                x.__rsub__(y) <=> y - x
`Tensor.__mul__`_                 x.__mul__(y) <=> x * y
`Tensor.__rmul__`_                x.__rmul__(y) <=> y * x
`Tensor.__div__`_                 x.__div__(y) <=> x / y
`Tensor.__rdiv__`_                x.__rdiv__(y) <=> y / x
`Tensor.__neg__`_                 x.__neg__()  <=> -x
`Tensor.__str__`_                 Return the information(name/shape).
`Tensor.__getitem__`_             Return a Tensor with specific indices.
`Tensor.__call__`_                Print the expressions.
==============================    =============================================================================


API Reference
-------------

.. currentmodule:: dragon.core.tensor

.. autoclass:: Tensor
    :members:

    .. automethod:: __init__

.. _Tensor.Variable: #dragon.core.tensor.Tensor.Variable
.. _Tensor.Placeholder: #dragon.core.tensor.Tensor.Placeholder
.. _Tensor.Constant: #dragon.core.tensor.Tensor.Constant
.. _Tensor.Uniform: #dragon.core.tensor.Tensor.Uniform
.. _Tensor.Normal: #dragon.core.tensor.Tensor.Normal
.. _Tensor.TruncatedNormal: #dragon.core.tensor.Tensor.TruncatedNormal
.. _Tensor.Gaussian: #dragon.core.tensor.Tensor.Gaussian
.. _Tensor.Xavier: #dragon.core.tensor.Tensor.Xavier
.. _Tensor.MSRA: #dragon.core.tensor.Tensor.MSRA
.. _Tensor.GlorotUniform: #dragon.core.tensor.Tensor.GlorotUniform
.. _Tensor.GlorotNormal: #dragon.core.tensor.Tensor.GlorotNormal

.. _Tensor.__add__: #dragon.core.tensor.Tensor.__add__
.. _Tensor.__radd__: #dragon.core.tensor.Tensor.__radd__
.. _Tensor.__sub__: #dragon.core.tensor.Tensor.__sub__
.. _Tensor.__rsub__: #dragon.core.tensor.Tensor.__rsub__
.. _Tensor.__mul__: #dragon.core.tensor.Tensor.__mul__
.. _Tensor.__rmul__: #dragon.core.tensor.Tensor.__rmul__
.. _Tensor.__div__: #dragon.core.tensor.Tensor.__div__
.. _Tensor.__rdiv__: #dragon.core.tensor.Tensor.__rdiv__
.. _Tensor.__neg__: #dragon.core.tensor.Tensor.__neg__
.. _Tensor.__str__: #dragon.core.tensor.Tensor.__str__
.. _Tensor.__getattr__: #dragon.core.tensor.Tensor.__getattr__
.. _Tensor.__getitem__: #dragon.core.tensor.Tensor.__getitem__
.. _Tensor.__call__: #dragon.core.tensor.Tensor.__call__

.. _Tensor.name: #dragon.core.tensor.Tensor.name
.. _Tensor.shape: #dragon.core.tensor.Tensor.shape
.. _Tensor.dtype: #dragon.core.tensor.Tensor.dtype
.. _Tensor.set_value: #dragon.core.tensor.Tensor.set_value
.. _Tensor.get_value: #dragon.core.tensor.Tensor.get_value
.. _Tensor.copy: #dragon.core.tensor.Tensor.copy
.. _Tensor.reshape: #dragon.core.tensor.Tensor.reshape
.. _Tensor.dimshuffle: #dragon.core.tensor.Tensor.dimshuffle
.. _Tensor.CreateOperator: #dragon.core.tensor.Tensor.CreateOperator
.. _Tensor.Fill: #dragon.core.tensor.Tensor.Fill
.. _Tensor.PrintExpressions: #dragon.core.tensor.Tensor.PrintExpressions

.. _workspace.FeedTensor(*args, **kwargs): workspace.html#dragon.core.workspace.FeedTensor
.. _workspace.FetchTensor(*args, **kwargs): workspace.html#dragon.core.workspace.FetchTensor
.. _ops.Copy(*args, **kwargs): ../operators/control_flow.html#dragon.operators.control_flow.Copy
