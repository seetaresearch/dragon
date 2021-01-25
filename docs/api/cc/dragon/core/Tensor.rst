Tensor
======

.. doxygenclass:: dragon::Tensor

Constructors
------------

.. doxygenfunction:: dragon::Tensor::Tensor()
.. doxygenfunction:: dragon::Tensor::Tensor(const string &name)
.. doxygenfunction:: dragon::Tensor::Tensor(const vec64_t &dims)
.. doxygenfunction:: dragon::Tensor::Tensor(const vec32_t &dims)
.. doxygenfunction:: dragon::Tensor::Tensor(const TypeMeta &meta)

Public Functions
----------------

CopyFrom
########
.. doxygenfunction:: dragon::Tensor::CopyFrom(const Tensor &other, Context *ctx)

CopyFrom
########
.. doxygenfunction:: dragon::Tensor::CopyFrom(const vector<VectorType> &other)

CopyTo
######
.. doxygenfunction:: dragon::Tensor::CopyTo

DimString
#########
.. doxygenfunction:: dragon::Tensor::DimString() const

DimString
#########
.. doxygenfunction:: dragon::Tensor::DimString(const vector<int64_t> &dims)

IsType
######
.. doxygenfunction:: dragon::Tensor::IsType

Reset
#####
.. doxygenfunction:: dragon::Tensor::Reset

Reshape
#######
.. doxygenfunction:: dragon::Tensor::Reshape

ReshapeLike
###########
.. doxygenfunction:: dragon::Tensor::ReshapeLike

Share
#####
.. doxygenfunction:: dragon::Tensor::Share

axis
####
.. doxygenfunction:: dragon::Tensor::axis

capacity
########
.. doxygenfunction:: dragon::Tensor::capacity

conut
#####
.. doxygenfunction:: dragon::Tensor::count() const

conut
#####
.. doxygenfunction:: dragon::Tensor::count(int64_t start) const

conut
#####
.. doxygenfunction:: dragon::Tensor::count(int64_t start, int64_t end) const

data
####
.. doxygenfunction:: dragon::Tensor::data

dim
###
.. doxygenfunction:: dragon::Tensor::dim

dims
####
.. doxygenfunction:: dragon::Tensor::dims

empty
#####
.. doxygenfunction:: dragon::Tensor::empty

has_memory
##########
.. doxygenfunction:: dragon::Tensor::has_memory

has_name
########
.. doxygenfunction:: dragon::Tensor::has_name

meta
####
.. doxygenfunction:: dragon::Tensor::meta

memory
######
.. doxygenfunction:: dragon::Tensor::memory

memory_state
############
.. doxygenfunction:: dragon::Tensor::memory_state

mutable_data
############
.. doxygenfunction:: dragon::Tensor::mutable_data

name
####
.. doxygenfunction:: dragon::Tensor::name

nbytes
######
.. doxygenfunction:: dragon::Tensor::nbytes

ndim
####
.. doxygenfunction:: dragon::Tensor::ndim

raw_data
########
.. doxygenfunction:: dragon::Tensor::raw_data

raw_mutable_data
################
.. doxygenfunction:: dragon::Tensor::raw_mutable_data()

raw_mutable_data
################
.. doxygenfunction:: dragon::Tensor::raw_mutable_data(const TypeMeta &meta)

size
####
.. doxygenfunction:: dragon::Tensor::size

stride
######
.. doxygenfunction:: dragon::Tensor::stride

strides
#######
.. doxygenfunction:: dragon::Tensor::strides

version
#######
.. doxygenfunction:: dragon::Tensor::version

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
