CPUContext
==========

.. doxygenclass:: dragon::CPUContext

Constructors
------------

.. doxygenfunction:: dragon::CPUContext::CPUContext()
.. doxygenfunction:: dragon::CPUContext::CPUContext(int random_seed)
.. doxygenfunction:: dragon::CPUContext::CPUContext(const DeviceOption &option)

Public Properties
-----------------

device
######
.. doxygenfunction:: dragon::CPUContext::device

mutex
#####
.. doxygenfunction:: dragon::CPUContext::mutex

objects
#######
.. doxygenfunction:: dragon::CPUContext::objects

random_seed
###########
.. doxygenfunction:: dragon::CPUContext::random_seed

rand_generator
##############
.. doxygenfunction:: dragon::CPUContext::rand_generator

stream
######
.. doxygenfunction:: dragon::CPUContext::stream

workspace
#########
.. doxygenfunction:: dragon::CPUContext::workspace

set_stream
##########
.. doxygenfunction:: dragon::CPUContext::set_stream

Public Functions
----------------

Copy
####
.. doxygenfunction:: dragon::CPUContext::Copy

Delete
######
.. doxygenfunction:: dragon::CPUContext::Delete

FinishDeviceComputation
#######################
.. doxygenfunction:: dragon::CPUContext::FinishDeviceComputation

Memset
######
.. doxygenfunction:: dragon::CPUContext::Memset

MemsetAsync
###########
.. doxygenfunction:: dragon::CPUContext::MemsetAsync

Memcpy
######
.. doxygenfunction:: dragon::CPUContext::Memcpy

MemcpyAsync
###########
.. doxygenfunction:: dragon::CPUContext::MemcpyAsync

New
###
.. doxygenfunction:: dragon::CPUContext::New

SwitchToDevice
##############
.. doxygenfunction:: dragon::CPUContext::SwitchToDevice

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
