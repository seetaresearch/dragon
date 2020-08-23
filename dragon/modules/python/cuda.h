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

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

namespace cuda {

class CUDAStream {
 public:
  explicit CUDAStream(int device_id) : device_id_(device_id) {
#ifdef USE_CUDA
    CUDADeviceGuard guard(device_id);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
#endif
  }

  ~CUDAStream() {
#ifdef USE_CUDA
    CUDA_CHECK(cudaStreamDestroy(stream_));
#endif
  }

  int device_id() {
    return device_id_;
  }

  intptr_t ptr() {
#ifdef USE_CUDA
    return (intptr_t)stream_;
#else
    return (intptr_t) nullptr;
#endif
  }

  void Synchronize() {
#ifdef USE_CUDA
    cudaStreamSynchronize(stream_);
    auto error = cudaGetLastError();
    CHECK_EQ(error, cudaSuccess)
        << "\nCUDA Error: " << cudaGetErrorString(error);
#endif
  }

 protected:
  int device_id_;
#ifdef USE_CUDA
  cudaStream_t stream_;
#endif
};

void RegisterModule(py::module& m) {
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

  /*! \brief Activate the CuDNN engine */
  m.def("cudaEnableDNN", [](bool enabled, bool benchmark) {
#ifdef USE_CUDA
    auto& cuda_objects = CUDAContext::objects();
    cuda_objects.cudnn_enabled_ = enabled;
    cuda_objects.cudnn_benchmark_ = benchmark;
#endif
  });

  /*! \brief Return the index of current device */
  m.def("cudaGetDevice", []() { return CUDAContext::current_device(); });

  /*! \brief Return the capability of specified device */
  m.def("cudaGetDeviceCapability", [](int device_id) {
#ifdef USE_CUDA
    if (device_id < 0) device_id = CUDAContext::current_device();
    auto& prop = GetCUDADeviceProp(device_id);
    return std::tuple<int, int>(prop.major, prop.minor);
#else
    return std::tuple<int, int>(0, 0);
#endif
  });

  /*! \brief Set the active cuda device */
  m.def("cudaSetDevice", [](int device_id) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaSetDevice(device_id));
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

  /*! \brief Export the stream class */
  py::class_<CUDAStream>(m, "CUDAStream")
      /*! \brief Default constructor */
      .def(py::init<int>())

      /*! \brief Return the device index */
      .def_property_readonly("device_id", &CUDAStream::device_id)

      /*! \brief Return the stream pointer */
      .def_property_readonly("ptr", &CUDAStream::ptr)

      /*! \brief Synchronize the stream */
      .def("Synchronize", &CUDAStream::Synchronize);
}

} // namespace cuda

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_CUDA_H_
