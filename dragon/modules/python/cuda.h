/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_CUDA_H_
#define DRAGON_MODULES_PYTHON_CUDA_H_

#include <dragon/core/context_cuda.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_cuda(py::module& m) {
  /*! \brief Return whether CUDA driver is sufficient */
  m.def("cudaIsDriverSufficient", []() {
#ifdef USE_CUDA
    int count;
    auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorInsufficientDriver) return false;
    return true;
#else
    return false;
#endif
  });

  /*! \brief Return whether NCCL is available */
  m.def("ncclIsAvailable", []() {
#ifdef USE_NCCL
#ifdef USE_CUDA
    int count;
    auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorInsufficientDriver) return false;
    return true;
#else // USE_CUDA
    return false;
#endif
#else // USE_NCCL
    return false;
#endif
  });

  /*! \brief Set the flags of cuBLAS library */
  m.def("cublasSetFlags", [](int allow_tf32) {
#ifdef USE_CUDA
    auto& ctx = CUDAContext::objects();
    if (allow_tf32 >= 0) ctx.cublas_allow_tf32_ = allow_tf32;
#endif
  });

  /*! \brief Set the flags of cuDNN library */
  m.def(
      "cudnnSetFlags",
      [](int enabled, int benchmark, int deterministic, int allow_tf32) {
#ifdef USE_CUDA
        auto& ctx = CUDAContext::objects();
        if (enabled >= 0) ctx.cudnn_enabled_ = enabled;
        if (benchmark >= 0) ctx.cudnn_benchmark_ = benchmark;
        if (deterministic >= 0) ctx.cudnn_deterministic_ = deterministic;
        if (allow_tf32 >= 0) ctx.cudnn_allow_tf32_ = allow_tf32;
#endif
      });

  /*! \brief Return the index of current device */
  m.def("cudaGetDevice", []() {
#ifdef USE_CUDA
    return CUDAContext::current_device();
#else
    return 0;
#endif
  });

  /*! \brief Return the number of available devices */
  m.def("cudaGetDeviceCount", []() {
#ifdef USE_CUDA
    return CUDAGetDeviceCount();
#else
    return 0;
#endif
  });

  /*! \brief Return the name of specified device */
  m.def("cudaGetDeviceName", [](int device_id) {
#ifdef USE_CUDA
    if (device_id < 0) device_id = CUDAContext::current_device();
    auto& prop = CUDAGetDeviceProp(device_id);
    return string(prop.name);
#else
    return string("");
#endif
  });

  /*! \brief Return the capability of specified device */
  m.def("cudaGetDeviceCapability", [](int device_id) {
#ifdef USE_CUDA
    if (device_id < 0) device_id = CUDAContext::current_device();
    auto& prop = CUDAGetDeviceProp(device_id);
    return std::tuple<int, int>(prop.major, prop.minor);
#else
    return std::tuple<int, int>(0, 0);
#endif
  });

  /*! \brief Set the active cuda device */
  m.def("cudaSetDevice", [](int device_id) {
#ifdef USE_CUDA
    CUDAContext::objects().SetDevice(device_id);
#endif
  });

  /*! \brief Set the random seed for cuda device */
  m.def("cudaSetRandomSeed", [](int device_id, int seed) {
#ifdef USE_CUDA
    if (device_id < 0) device_id = CUDAContext::current_device();
    CUDAContext::objects().SetRandomSeed(device_id, seed);
#endif
  });

  /*! \brief Synchronize the specified cuda stream */
  m.def("cudaStreamSynchronize", [](int device_id, int stream_id) {
#ifdef USE_CUDA
    if (device_id < 0) device_id = CUDAContext::current_device();
    auto stream = CUDAContext::objects().stream(device_id, stream_id);
    CUDAContext::SynchronizeStream(stream);
#endif
  });

  /*! \brief CUDAStream class */
  py::class_<CUDAStream>(m, "CUDAStream")
      /*! \brief Default constructor */
      .def(py::init<int>())

      /*! \brief Return the stream index */
      .def_property_readonly("index", &CUDAStream::stream_id)

      /*! \brief Return the stream pointer */
      .def_property_readonly(
          "ptr", [](CUDAStream* self) { return (intptr_t)self->impl(); })

      /*! \brief Query the completion status */
      .def("Query", &CUDAStream::Query)

      /*! \brief Wait for the dispatched kernels to complete */
      .def("Synchronize", &CUDAStream::Synchronize);

  /*! \brief CUDAGraph class */
  py::class_<CUDAGraph>(m, "CUDAGraph")
      /*! \brief Default constructor */
      .def(py::init<>())

      /*! \brief Begin the graph capture */
      .def("BeginCapture", &CUDAGraph::BeginCapture)

      /*! \brief End the graph capture */
      .def("EndCapture", &CUDAGraph::EndCapture)

      /*! \brief Launch graph on the captured stream */
      .def("Launch", &CUDAGraph::Launch)

      /*! \brief Reset the graph capture */
      .def("Reset", &CUDAGraph::Reset);
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_CUDA_H_
