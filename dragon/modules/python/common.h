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

#ifndef DRAGON_MODULES_PYTHON_COMMON_H_
#define DRAGON_MODULES_PYTHON_COMMON_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "dragon/core/common.h"
#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/graph_gradient.h"
#include "dragon/core/operator.h"
#include "dragon/core/operator_gradient.h"
#include "dragon/core/registry.h"
#include "dragon/core/workspace.h"
#include "dragon/modules/python/types.h"
#include "dragon/onnx/onnx_backend.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace dragon {

namespace python {

namespace py = pybind11;

class TensorFetcherBase {
 public:
  virtual ~TensorFetcherBase() {}
  virtual py::object Fetch(const Tensor& tensor) = 0;
};

class TensorFeederBase {
 public:
  virtual ~TensorFeederBase() {}
  virtual void
  Feed(const DeviceOption& option, PyArrayObject* array, Tensor* tensor) = 0;
};

DECLARE_REGISTRY(TensorFetcherRegistry, TensorFetcherBase);

class NumpyFetcher : public TensorFetcherBase {
 public:
  py::object Fetch(const Tensor& tensor) override {
    CHECK_GT(tensor.count(), 0);
    vector<npy_intp> npy_dims;
    for (auto dim : tensor.dims())
      npy_dims.push_back(dim);
    int npy_type = types::to_npy(tensor.meta());
    CHECK(npy_type != -1) << "\nThe data type of Tensor(" << tensor.name()
                          << ") is unknown. Have you solved it?";
    CHECK(tensor.memory()) << "\nIllegal memory access.";
    // Create a empty array with the same shape
    auto* array = PyArray_SimpleNew(tensor.ndim(), npy_dims.data(), npy_type);
    // Copy the tensor data to the numpy array
    if (tensor.memory_state() == UnifiedMemory::STATE_AT_CUDA) {
      CUDAContext::Memcpy<CPUContext, CUDAContext>(
          tensor.nbytes(),
          PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
          tensor.raw_data<CUDAContext>(),
          tensor.memory()->device());
    } else {
      CPUContext::Memcpy<CPUContext, CPUContext>(
          tensor.nbytes(),
          PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
          tensor.raw_data<CPUContext>());
    }
    return py::reinterpret_steal<py::object>(array);
  }
};

class StringFetcher : public TensorFetcherBase {
 public:
  py::object Fetch(const Tensor& tensor) override {
    CHECK_EQ(tensor.count(), 1);
    return py::bytes(tensor.data<string, CPUContext>()[0]);
  }
};

class NumpyFeeder : public TensorFeederBase {
 public:
  void Feed(
      const DeviceOption& option,
      PyArrayObject* original_array,
      Tensor* tensor) override {
    auto* array = PyArray_GETCONTIGUOUS(original_array);
    const auto& meta = types::from_npy(PyArray_TYPE(array));
    if (meta.id() == 0) {
      LOG(FATAL) << "Type <" << ::dragon::types::to_string(meta)
                 << "> is not supported to feed.";
    }
    tensor->set_meta(meta);
    int ndim = PyArray_NDIM(array);
    vec64_t dims(ndim);
    auto* npy_dims = PyArray_DIMS(array);
    for (int i = 0; i < ndim; i++) {
      dims[i] = npy_dims[i];
    }
    tensor->Reshape(dims);
    if (option.device_type() == PROTO_CUDA) {
#ifdef USE_CUDA
      CUDAContext::Memcpy<CUDAContext, CPUContext>(
          tensor->nbytes(),
          tensor->raw_mutable_data<CUDAContext>(),
          static_cast<void*>(PyArray_DATA(array)),
          option.device_id());
#else
      LOG(FATAL) << "CUDA was not compiled.";
#endif
    } else {
      CPUContext::Memcpy<CPUContext, CPUContext>(
          tensor->nbytes(),
          tensor->raw_mutable_data<CPUContext>(),
          static_cast<void*>(PyArray_DATA(array)));
    }
    Py_XDECREF(array);
  }
};

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_COMMON_H_
