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

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

class NumpyWrapper {
 public:
  NumpyWrapper(Tensor* tensor) : tensor_(tensor) {}

  py::object To(bool copy) {
    const auto& meta = tensor_->meta();
    const auto& typestr = ::dragon::dtypes::to_string(meta);
    CHECK_GT(tensor_->count(), 0) << "\nConvert an empty tensor.";
    CHECK(typestr != "unknown") << "\nConvert an empty tensor.";
    if (typestr == "string") {
      CHECK_EQ(tensor_->count(), 1);
      return py::bytes(tensor_->data<string, CPUContext>()[0]);
    }
    auto typenum = dtypes::to_npy(meta);
    vector<npy_intp> dims({tensor_->dims().begin(), tensor_->dims().end()});
    if (copy) {
      auto* memory = tensor_->memory();
      CHECK(memory) << "\nConvert an empty tensor.";
      auto device_type = memory ? memory->info()["device_type"] : "cpu";
      auto* array = PyArray_SimpleNew(tensor_->ndim(), dims.data(), typenum);
      if (device_type == "cuda") {
        CUDAContext::Memcpy<CPUContext, CUDAContext>(
            tensor_->nbytes(),
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            tensor_->raw_data<CUDAContext>(),
            memory->device());
      } else {
        CPUContext::Memcpy<CPUContext, CPUContext>(
            tensor_->nbytes(),
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            tensor_->raw_data<CPUContext>());
      }
      return py::reinterpret_steal<py::object>(array);
    }
    auto* data = const_cast<void*>(tensor_->raw_data<CPUContext>());
    auto* array = PyArray_SimpleNewFromData(
        tensor_->ndim(), dims.data(), dtypes::to_npy(meta), data);
    return py::reinterpret_steal<py::object>(array);
  }

  Tensor* From(py::object obj, bool copy) {
    auto* array =
        PyArray_GETCONTIGUOUS(reinterpret_cast<PyArrayObject*>(obj.ptr()));
    const auto& meta = dtypes::from_npy(PyArray_TYPE(array));
    CHECK(meta.id() != 0) << "\nUnsupported numpy array type.";
    auto* npy_dims = PyArray_DIMS(array);
    auto* data = static_cast<void*>(PyArray_DATA(array));
    vector<int64_t> dims(npy_dims, npy_dims + PyArray_NDIM(array));
    auto* memory = tensor_->set_meta(meta)->Reshape(dims)->memory();
    if (copy) {
      auto device_type = memory ? memory->info()["device_type"] : "cpu";
      if (device_type == "cuda") {
        CUDAContext::Memcpy<CUDAContext, CPUContext>(
            tensor_->nbytes(),
            tensor_->raw_mutable_data<CUDAContext>(),
            data,
            memory->device());
      } else {
        CPUContext::Memcpy<CPUContext, CPUContext>(
            tensor_->nbytes(), tensor_->raw_mutable_data<CPUContext>(), data);
      }
      Py_XDECREF(array);
    } else {
      memory = memory ? memory : new UnifiedMemory();
      memory->set_cpu_data(data, tensor_->nbytes());
      tensor_->set_memory(memory);
      if (tensor_->ExternalDeleter) tensor_->ExternalDeleter();
      tensor_->ExternalDeleter = [array]() -> void { Py_XDECREF(array); };
    }
    return tensor_;
  }

 private:
  Tensor* tensor_;
};

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_NUMPY_H_
