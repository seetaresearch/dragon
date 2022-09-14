CPUContext
==========

.. doxygenclass:: dragon::CPUContext

Constructors
------------

.. doxygenfunction:: dragon::CPUContext::CPUContext()
.. doxygenfunction:: dragon::CPUContext::CPUContext(unsigned int random_seed)
.. doxygenfunction:: dragon::CPUContext::CPUContext(const DeviceOption &option)

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

device
######
.. doxygenfunction:: dragon::CPUContext::device

rand_generator
##############
.. doxygenfunction:: dragon::CPUContext::rand_generator

set_stream
##########
.. doxygenfunction:: dragon::CPUContext::set_stream

stream
######
.. doxygenfunction:: dragon::CPUContext::stream

workspace
#########
.. doxygenfunction:: dragon::CPUContext::workspace

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
