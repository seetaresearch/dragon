MLUContext
===========

.. doxygenclass:: dragon::MLUContext

Constructors
------------

.. doxygenfunction:: dragon::MLUContext::MLUContext()
.. doxygenfunction:: dragon::MLUContext::MLUContext(int device)
.. doxygenfunction:: dragon::MLUContext::MLUContext(const DeviceOption &option)

Public Properties
-----------------

cnnl_handle
###########
.. doxygenfunction:: dragon::MLUContext::cnnl_handle

cnrand_generator
################
.. doxygenfunction:: dragon::MLUContext::cnrand_generator

current_device
##############
.. doxygenfunction:: dragon::MLUContext::current_device

device
######
.. doxygenfunction:: dragon::MLUContext::device

mlu_stream
##########
.. doxygenfunction:: dragon::MLUContext::mlu_stream()

mlu_stream
##########
.. doxygenfunction:: dragon::MLUContext::mlu_stream(int device, int stream)

mutex
#####
.. doxygenfunction:: dragon::MLUContext::mutex

objects
#######
.. doxygenfunction:: dragon::MLUContext::objects

random_seed
###########
.. doxygenfunction:: dragon::MLUContext::random_seed

stream
######
.. doxygenfunction:: dragon::MLUContext::stream

workspace
#########
.. doxygenfunction:: dragon::MLUContext::workspace()

workspace
#########
.. doxygenfunction:: dragon::MLUContext::workspace(int device, int stream)

set_stream
##########
.. doxygenfunction:: dragon::MLUContext::set_stream

Public Functions
----------------

Copy
####
.. doxygenfunction:: dragon::MLUContext::Copy

Delete
######
.. doxygenfunction:: dragon::MLUContext::Delete

FinishDeviceComputation
#######################
.. doxygenfunction:: dragon::MLUContext::FinishDeviceComputation

Memset
######
.. doxygenfunction:: dragon::MLUContext::Memset

MemsetAsync
###########
.. doxygenfunction:: dragon::MLUContext::MemsetAsync

Memcpy
######
.. doxygenfunction:: dragon::MLUContext::Memcpy(size_t n, void *dest, const void *src)

Memcpy
######
.. doxygenfunction:: dragon::MLUContext::Memcpy(size_t n, void *dest, const void *src, int device)

MemcpyAsync
###########
.. doxygenfunction:: dragon::MLUContext::MemcpyAsync

New
###
.. doxygenfunction:: dragon::MLUContext::New

SwitchToDevice
##############
.. doxygenfunction:: dragon::MLUContext::SwitchToDevice

SynchronizeStream
#################
.. doxygenfunction:: dragon::MLUContext::SynchronizeStream

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
