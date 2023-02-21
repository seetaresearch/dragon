MPSContext
==========

.. doxygenclass:: dragon::MPSContext

Constructors
------------

.. doxygenfunction:: dragon::MPSContext::MPSContext()
.. doxygenfunction:: dragon::MPSContext::MPSContext(int device)
.. doxygenfunction:: dragon::MPSContext::MPSContext(const DeviceOption &option)

Public Properties
-----------------

current_device
##############
.. doxygenfunction:: dragon::MPSContext::current_device

device
######
.. doxygenfunction:: dragon::MPSContext::device

mps_stream
##########
.. doxygenfunction:: dragon::MPSContext::mps_stream

mutex
#####
.. doxygenfunction:: dragon::MPSContext::mutex

objects
#######
.. doxygenfunction:: dragon::MPSContext::objects

random_seed
###########
.. doxygenfunction:: dragon::MPSContext::random_seed

stream
######
.. doxygenfunction:: dragon::MPSContext::stream

workspace
#########
.. doxygenfunction:: dragon::MPSContext::workspace()

workspace
#########
.. doxygenfunction:: dragon::MPSContext::workspace(int device, int stream)

set_stream
##########
.. doxygenfunction:: dragon::MPSContext::set_stream

Public Functions
----------------

Delete
######
.. doxygenfunction:: dragon::MPSContext::Delete

FinishDeviceComputation
#######################
.. doxygenfunction:: dragon::MPSContext::FinishDeviceComputation

Memset
######
.. doxygenfunction:: dragon::MPSContext::Memset

MemsetAsync
###########
.. doxygenfunction:: dragon::MPSContext::MemsetAsync

Memcpy
######
.. doxygenfunction:: dragon::MPSContext::Memcpy(size_t n, void *dest, const void *src)

Memcpy
######
.. doxygenfunction:: dragon::MPSContext::Memcpy(size_t n, void *dest, const void *src, int device)

MemcpyAsync
###########
.. doxygenfunction:: dragon::MPSContext::MemcpyAsync

New
###
.. doxygenfunction:: dragon::MPSContext::New

NewShared
#########
.. doxygenfunction:: dragon::MPSContext::NewShared

NewSharedFromBytes
##################
.. doxygenfunction:: dragon::MPSContext::NewSharedFromBytes

NewSharedFromBuffer
###################
.. doxygenfunction:: dragon::MPSContext::NewSharedFromBuffer

SwitchToDevice
##############
.. doxygenfunction:: dragon::MPSContext::SwitchToDevice

SynchronizeStream
#################
.. doxygenfunction:: dragon::MPSContext::SynchronizeStream

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
