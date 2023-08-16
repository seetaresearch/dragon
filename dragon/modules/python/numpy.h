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

#ifndef DRAGON_MODULES_PYTHON_NUMPY_H_
#define DRAGON_MODULES_PYTHON_NUMPY_H_

#include <dragon/core/context_cuda.h>
#include <dragon/core/context_mlu.h>
#include <dragon/core/tensor.h>

#include "dragon/modules/python/common.h"
#include "dragon/modules/python/types.h"

namespace dragon {

namespace python {

class NumpyWrapper {
 public:
  NumpyWrapper(Tensor* tensor) : tensor_(tensor) {}

  py::object To(bool copy) {
    const auto& meta = tensor_->meta();
    const auto& dtype = ::dragon::dtypes::to_string(meta);
    CHECK_NE(dtype, "unknown") << "\nAn empty tensor to numpy.";
    CHECK_GT(tensor_->count(), 0) << "\nAn empty tensor to numpy.";
    if (dtype == "string") {
      CHECK_EQ(tensor_->count(), 1) << "\nNon-scalar string tensor to numpy.";
      return py::bytes(tensor_->data<string, CPUContext>()[0]);
    }
    auto npy_dtype = dtypes::to_npy(meta);
    if (npy_dtype == NPY_NOTYPE) {
      // Fallback to NumPy RTTI.
      CHECK_NE((npy_dtype = PyArray_TypeNumFromName(dtype.c_str())), NPY_NOTYPE)
          << "\nUnsupported " << dtype << " tensor to numpy.";
    }
    vector<npy_intp> dims({tensor_->dims().begin(), tensor_->dims().end()});
    if (copy) {
      auto* memory = tensor_->memory();
      CHECK(memory) << "\nAn empty tensor to numpy.";
      auto device_type = memory ? memory->info()["device_type"] : "cpu";
      auto* array = PyArray_SimpleNew(dims.size(), dims.data(), npy_dtype);
      if (device_type == "cuda") {
#ifdef USE_CUDA
        CUDADeviceGuard guard(memory->device());
        CUDAContext::Memcpy<CPUContext, CUDAContext>(
            tensor_->nbytes(),
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            tensor_->raw_data<CUDAContext>(),
            memory->device());
#else
        CUDA_NOT_COMPILED;
#endif
      } else if (device_type == "mlu") {
#ifdef USE_MLU
        MLUDeviceGuard guard(memory->device());
        MLUContext::Memcpy<CPUContext, MLUContext>(
            tensor_->nbytes(),
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            tensor_->raw_data<MLUContext>(),
            memory->device());
#else
        MLU_NOT_COMPILED;
#endif
      } else {
        CPUContext::Memcpy<CPUContext, CPUContext>(
            tensor_->nbytes(),
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            tensor_->raw_data<CPUContext>());
      }
      return py::reinterpret_steal<py::object>(array);
    }
    return py::reinterpret_steal<py::object>(PyArray_SimpleNewFromData(
        dims.size(),
        dims.data(),
        npy_dtype,
        const_cast<void*>(tensor_->raw_data<CPUContext>())));
  }

  Tensor* From(py::object obj, bool copy) {
    auto* array = PyArray_GETCONTIGUOUS((PyArrayObject*)obj.ptr());
    const auto* meta = &dtypes::from_npy(PyArray_TYPE(array));
    if (meta->id() == 0) {
      // Fallback to NumPy RTTI.
      if (PyArray_TYPE(array) == PyArray_TypeNumFromName("bfloat16")) {
        meta = &::dragon::dtypes::to_meta("bfloat16");
      }
    }
    CHECK(meta->id() != 0) << "\nUnsupported numpy type to tensor.";
    auto* npy_dims = PyArray_DIMS(array);
    auto* data = static_cast<void*>(PyArray_DATA(array));
    vector<int64_t> dims(npy_dims, npy_dims + PyArray_NDIM(array));
    tensor_->set_meta(*meta)->Reshape(dims);
    auto* memory = tensor_->MapFrom(nullptr)->memory();
    auto device_type = memory ? memory->info()["device_type"] : "cpu";
    if (copy) {
      if (device_type == "cuda") {
#ifdef USE_CUDA
        CUDADeviceGuard guard(memory->device());
        CUDAContext::Memcpy<CUDAContext, CPUContext>(
            tensor_->nbytes(),
            tensor_->raw_mutable_data<CUDAContext>(),
            data,
            memory->device());
#else
        CUDA_NOT_COMPILED;
#endif
      } else if (device_type == "mlu") {
#ifdef USE_MLU
        MLUDeviceGuard guard(memory->device());
        MLUContext::Memcpy<MLUContext, CPUContext>(
            tensor_->nbytes(),
            tensor_->raw_mutable_data<MLUContext>(),
            data,
            memory->device());
#else
        MLU_NOT_COMPILED;
#endif
      } else {
        CPUContext::Memcpy<CPUContext, CPUContext>(
            tensor_->nbytes(), tensor_->raw_mutable_data<CPUContext>(), data);
      }
      Py_XDECREF(array);
    } else {
      memory = memory ? memory : new UnifiedMemory();
      if (memory->set_cpu_data(data, tensor_->nbytes())) {
        if (tensor_->ExternalDeleter) tensor_->ExternalDeleter();
        tensor_->ExternalDeleter = [array]() -> void { Py_XDECREF(array); };
      } else {
        if (tensor_->ExternalDeleter) tensor_->ExternalDeleter();
        tensor_->ExternalDeleter = nullptr;
      }
      tensor_->set_memory(memory);
    }
    return tensor_;
  }

 private:
  Tensor* tensor_;
};

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_NUMPY_H_
