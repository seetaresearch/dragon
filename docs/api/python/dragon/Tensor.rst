Tensor
======

.. autoclass:: dragon.Tensor

__init__
--------
.. automethod:: dragon.Tensor.__init__

Properties
----------

dtype
#####
.. autoattribute:: dragon.Tensor.dtype

id
##
.. autoattribute:: dragon.Tensor.id

name
####
.. autoattribute:: dragon.Tensor.name

ndim
####
.. autoattribute:: dragon.Tensor.ndim

shape
#####
.. autoattribute:: dragon.Tensor.shape

size
#####
.. autoattribute:: dragon.Tensor.size

Methods
-------

astype
######
.. automethod:: dragon.Tensor.astype

constant
########
.. automethod:: dragon.Tensor.constant

copy
####
.. automethod:: dragon.Tensor.copy

from_value
##########
.. automethod:: dragon.Tensor.from_value

get_value
##########
.. automethod:: dragon.Tensor.get_value

glorot_normal
#############
.. automethod:: dragon.Tensor.glorot_normal

glorot_uniform 
##############
.. automethod:: dragon.Tensor.glorot_uniform

normal
######
.. automethod:: dragon.Tensor.normal

reshape
#######
.. automethod:: dragon.Tensor.reshape

set_value
#########
.. automethod:: dragon.Tensor.set_value

truncated_normal
################
.. automethod:: dragon.Tensor.truncated_normal

uniform
#######
.. automethod:: dragon.Tensor.uniform

Overrides
---------

__add__
#######
.. automethod:: dragon.Tensor.__add__

__float__
#########
.. automethod:: dragon.Tensor.__float__

__ge__
######
.. automethod:: dragon.Tensor.__ge__

__getitem__
###########
.. automethod:: dragon.Tensor.__getitem__

__gt__
######
.. automethod:: dragon.Tensor.__gt__

__int__
#######
.. automethod:: dragon.Tensor.__int__

__le__
######
.. automethod:: dragon.Tensor.__le__

__lt__
######
.. automethod:: dragon.Tensor.__lt__

__mul__
#######
.. automethod:: dragon.Tensor.__mul__

__neg__
#######
.. automethod:: dragon.Tensor.__neg__

__radd__
########
.. automethod:: dragon.Tensor.__radd__

__rmul__
########
.. automethod:: dragon.Tensor.__rmul__

__rsub__
########
.. automethod:: dragon.Tensor.__rsub__

__setitem__
###########
.. automethod:: dragon.Tensor.__setitem__

__sub__
#######
.. automethod:: dragon.Tensor.__sub__

__rtruediv__
############
.. automethod:: dragon.Tensor.__rtruediv__

__truediv__
############
.. automethod:: dragon.Tensor.__truediv__

.. _dragon.assign(...): assign.html
.. _dragon.cast(...): cast.html
.. _dragon.fill(...): fill.html
.. _dragon.identity(...): identity.html
.. _dragon.masked_assign(...): masked_assign.html
.. _dragon.masked_select(...): masked_select.html
.. _dragon.math.add(...): math/add.html
.. _dragon.math.div(...): math/div.html
.. _dragon.math.greater(...): math/greater.html
.. _dragon.math.greater_equal(...): math/greater_equal.html
.. _dragon.math.less(...): math/less.html
.. _dragon.math.less_equal(...): math/less_equal.html
.. _dragon.math.mul(...): math/mul.html
.. _dragon.math.negative(...): math/negative.html
.. _dragon.math.sub(...): math/sub.html
.. _dragon.random.glorot_normal(...): random/glorot_normal.html
.. _dragon.random.glorot_uniform(...): random/glorot_uniform.html
.. _dragon.random.normal(...): random/normal.html
.. _dragon.random.truncated_normal(...): random/truncated_normal.html
.. _dragon.random.uniform(...): random/uniform.html
.. _dragon.reshape(...): reshape.html
.. _dragon.slice(...): slice.html

.. raw:: html

  <style>
    h1:before {
      content: "dragon.";
      color: #103d3e;
    }
  </style>
