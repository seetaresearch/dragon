CUDAContext
===========

.. doxygenclass:: dragon::CUDAContext

Constructors
------------

.. doxygenfunction:: dragon::CUDAContext::CUDAContext()
.. doxygenfunction:: dragon::CUDAContext::CUDAContext(int device)
.. doxygenfunction:: dragon::CUDAContext::CUDAContext(const DeviceOption &option)

Public Functions
----------------

Copy
####
.. doxygenfunction:: dragon::CUDAContext::Copy

Delete
######
.. doxygenfunction:: dragon::CUDAContext::Delete

FinishDeviceComputation
#######################
.. doxygenfunction:: dragon::CUDAContext::FinishDeviceComputation

Memset
######
.. doxygenfunction:: dragon::CUDAContext::Memset

MemsetAsync
###########
.. doxygenfunction:: dragon::CUDAContext::MemsetAsync

Memcpy
######
.. doxygenfunction:: dragon::CUDAContext::Memcpy(size_t n, void *dest, const void *src)

Memcpy
######
.. doxygenfunction:: dragon::CUDAContext::Memcpy(size_t n, void *dest, const void *src, int device)

MemcpyAsync
###########
.. doxygenfunction:: dragon::CUDAContext::MemcpyAsync

New
###
.. doxygenfunction:: dragon::CUDAContext::New

SwitchToDevice
##############
.. doxygenfunction:: dragon::CUDAContext::SwitchToDevice()

SwitchToDevice
##############
.. doxygenfunction:: dragon::CUDAContext::SwitchToDevice(int stream)

SynchronizeStream
#################
.. doxygenfunction:: dragon::CUDAContext::SynchronizeStream

cublas_handle
#############
.. doxygenfunction:: dragon::CUDAContext::cublas_handle

cuda_stream
###########
.. doxygenfunction:: dragon::CUDAContext::cuda_stream()

cuda_stream
###########
.. doxygenfunction:: dragon::CUDAContext::cuda_stream(int device, int stream)

cudnn_handle
############
.. doxygenfunction:: dragon::CUDAContext::cudnn_handle

curand_generator
################
.. doxygenfunction:: dragon::CUDAContext::curand_generator

rand_generator
##############
.. doxygenfunction:: dragon::CUDAContext::rand_generator

device
######
.. doxygenfunction:: dragon::CUDAContext::device

set_stream
##########
.. doxygenfunction:: dragon::CUDAContext::set_stream

stream
######
.. doxygenfunction:: dragon::CUDAContext::stream

workspace
#########
.. doxygenfunction:: dragon::CUDAContext::workspace()

workspace
#########
.. doxygenfunction:: dragon::CUDAContext::workspace(int device, int stream)

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
