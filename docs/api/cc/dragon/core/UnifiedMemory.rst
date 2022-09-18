UnifiedMemory
=============

.. doxygenclass:: dragon::UnifiedMemory

Constructors
------------

.. doxygenfunction:: dragon::UnifiedMemory::UnifiedMemory()
.. doxygenfunction:: dragon::UnifiedMemory::UnifiedMemory(const TypeMeta &meta, size_t size)

Public Types
------------

State
#####
.. doxygenenum:: dragon::UnifiedMemory::State

Public Properties
-----------------

cpu_data
########
.. doxygenfunction:: dragon::UnifiedMemory::cpu_data

cuda_data
#########
.. doxygenfunction:: dragon::UnifiedMemory::cuda_data

device
######
.. doxygenfunction:: dragon::UnifiedMemory::device

info
####
.. doxygenfunction:: dragon::UnifiedMemory::info

mps_data
########
.. doxygenfunction:: dragon::UnifiedMemory::mps_data

order
#####
.. doxygenfunction:: dragon::UnifiedMemory::order

size
####
.. doxygenfunction:: dragon::UnifiedMemory::size() const

size
####
.. doxygenfunction:: dragon::UnifiedMemory::size(const string &device_type, int device_id) const

state
#####
.. doxygenfunction:: dragon::UnifiedMemory::state

mutable_cpu_data
################
.. doxygenfunction:: dragon::UnifiedMemory::mutable_cpu_data

mutable_cuda_data
#################
.. doxygenfunction:: dragon::UnifiedMemory::mutable_cuda_data

set_cpu_data
############
.. doxygenfunction:: dragon::UnifiedMemory::set_cpu_data

set_cuda_data
#############
.. doxygenfunction:: dragon::UnifiedMemory::set_cuda_data

set_order
#########
.. doxygenfunction:: dragon::UnifiedMemory::set_order

Public Functions
----------------

SwitchToCUDADevice
##################
.. doxygenfunction:: dragon::UnifiedMemory::SwitchToCUDADevice

SwitchToMPSDevice
##################
.. doxygenfunction:: dragon::UnifiedMemory::SwitchToMPSDevice

ToCPU
#####
.. doxygenfunction:: dragon::UnifiedMemory::ToCPU

ToCUDA
######
.. doxygenfunction:: dragon::UnifiedMemory::ToCUDA

ToMPS
#####
.. doxygenfunction:: dragon::UnifiedMemory::ToMPS

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
