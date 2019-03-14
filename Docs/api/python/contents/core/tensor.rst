=============
:mod:`Tensor`
=============

.. toctree::
   :hidden:

Quick Reference
---------------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Tensor.name`_                    Return or Set the name.
`Tensor.shape`_                   Return or Set the shape.
`Tensor.get_shape`_               Return the shape.
`Tensor.dtype`_                   Return or Set the data type.
`Tensor.set_value`_               Feed the values to C++ backend.
`Tensor.get_value`_               Fetch the values from C++ backend.
`Tensor.copy`_                    Return a Tensor with same content.
`Tensor.reshape`_                 Reshape the dimensions of input.
`Tensor.dimshuffle`_              Shuffle the dimensions.
`Tensor.eval`_                    Run and return the computing results of this tensor.
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
`Tensor.GlorotUniform`_           Register as a variable with glorot uniform initializer.
`Tensor.GlorotNormal`_            Register as a variable with glorot normal initializer.
==============================    =============================================================================

Override
--------

==============================    =============================================================================
List                              Brief
==============================    =============================================================================
`Tensor.__add__`_                 x.__add__(y) <=> x + y
`Tensor.__radd__`_                x.__radd__(y) <=> y + x
`Tensor.__sub__`_                 x.__sub__(y) <=> x - y
`Tensor.__rsub__`_                x.__rsub__(y) <=> y - x
`Tensor.__mul__`_                 x.__mul__(y) <=> x * y
`Tensor.__rmul__`_                x.__rmul__(y) <=> y * x
`Tensor.__div__`_                 x.__div__(y) <=> x / y
`Tensor.__rdiv__`_                x.__rdiv__(y) <=> y / x
`Tensor.__neg__`_                 x.__neg__()  <=> -x
`Tensor.__gt__`_                  x.__gt__()  <=> x > y
`Tensor.__ge__`_                  x.__ge__()  <=> x >= y
`Tensor.__lt__`_                  x.__lt__()  <=> x < y
`Tensor.__le__`_                  x.__le__()  <=> x <= y
`Tensor.__eq__`_                  x.__eq__()  <=> x == y
`Tensor.__repr__`_                Return the information(name/shape).
`Tensor.__getitem__`_             Return a Tensor with specific indices.
`Tensor.__call__`_                Return the expressions for displaying.
==============================    =============================================================================


API Reference
-------------

.. currentmodule:: dragon.core.tensor

.. autoclass:: Tensor
    :members:

    .. automethod:: __init__
    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __div__
    .. automethod:: __rdiv__
    .. automethod:: __neg__
    .. automethod:: __gt__
    .. automethod:: __ge__
    .. automethod:: __lt__
    .. automethod:: __le__
    .. automethod:: __eq__
    .. automethod:: __eq__
    .. automethod:: __repr__
    .. automethod:: __getitem__
    .. automethod:: __call__

.. _Tensor.Variable: #dragon.core.tensor.Tensor.Variable
.. _Tensor.Placeholder: #dragon.core.tensor.Tensor.Placeholder
.. _Tensor.Constant: #dragon.core.tensor.Tensor.Constant
.. _Tensor.Uniform: #dragon.core.tensor.Tensor.Uniform
.. _Tensor.Normal: #dragon.core.tensor.Tensor.Normal
.. _Tensor.TruncatedNormal: #dragon.core.tensor.Tensor.TruncatedNormal
.. _Tensor.Gaussian: #dragon.core.tensor.Tensor.Gaussian
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
.. _Tensor.__gt__: #dragon.core.tensor.Tensor.__gt__
.. _Tensor.__ge__: #dragon.core.tensor.Tensor.__ge__
.. _Tensor.__lt__: #dragon.core.tensor.Tensor.__lt__
.. _Tensor.__le__: #dragon.core.tensor.Tensor.__le__
.. _Tensor.__eq__: #dragon.core.tensor.Tensor.__eq__
.. _Tensor.__repr__: #dragon.core.tensor.Tensor.__repr__
.. _Tensor.__getitem__: #dragon.core.tensor.Tensor.__getitem__
.. _Tensor.__call__: #dragon.core.tensor.Tensor.__call__

.. _Tensor.name: #dragon.core.tensor.Tensor.name
.. _Tensor.shape: #dragon.core.tensor.Tensor.shape
.. _Tensor.get_shape: #dragon.core.tensor.Tensor.get_shape
.. _Tensor.dtype: #dragon.core.tensor.Tensor.dtype
.. _Tensor.set_value: #dragon.core.tensor.Tensor.set_value
.. _Tensor.get_value: #dragon.core.tensor.Tensor.get_value
.. _Tensor.copy: #dragon.core.tensor.Tensor.copy
.. _Tensor.reshape: #dragon.core.tensor.Tensor.reshape
.. _Tensor.dimshuffle: #dragon.core.tensor.Tensor.dimshuffle
.. _Tensor.eval: #dragon.core.tensor.Tensor.eval

.. _workspace.FeedTensor(*args, **kwargs): workspace.html#dragon.core.workspace.FeedTensor
.. _workspace.FetchTensor(*args, **kwargs): workspace.html#dragon.core.workspace.FetchTensor
.. _ops.Copy(*args, **kwargs): ../operators/control_flow.html#dragon.operators.control_flow.Copy
