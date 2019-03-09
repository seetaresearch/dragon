/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_PYTHON_PY_DRAGON_H_
#define DRAGON_PYTHON_PY_DRAGON_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "py_types.h"
#include "core/common.h"
#include "core/registry.h"
#include "core/context.h"
#include "core/context_cuda.h"
#include "core/operator.h"
#include "core/operator_gradient.h"
#include "core/graph_gradient.h"
#include "core/workspace.h"
#include "utils/caffemodel.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace dragon {

namespace python {

class TensorFetcherBase {
 public:
    virtual ~TensorFetcherBase() {}
    virtual pybind11::object Fetch(const Tensor& tensor) = 0;
};

class TensorFeederBase {
 public:
    virtual ~TensorFeederBase() {}
    virtual void Feed(
        const DeviceOption&             option,
        PyArrayObject*                  array,
        Tensor*                         tensor) = 0;
};

DECLARE_TYPED_REGISTRY(TensorFetcherRegistry, TypeId, TensorFetcherBase);

#define REGISTER_TENSOR_FETCHER(type, ...) \
    REGISTER_TYPED_CLASS(TensorFetcherRegistry, type, __VA_ARGS__)

inline TensorFetcherBase* CreateFetcher(TypeId type) {
    return TensorFetcherRegistry()->Create(type);
}

DECLARE_TYPED_REGISTRY(TensorFeederRegistry, TypeId, TensorFeederBase);

#define REGISTER_TENSOR_FEEDER(type, ...) \
    REGISTER_TYPED_CLASS(TensorFeederRegistry, type, __VA_ARGS__)

class NumpyFetcher : public TensorFetcherBase {
 public:
    pybind11::object Fetch(const Tensor& tensor) override {
        CHECK_GT(tensor.count(), 0);
        vector<npy_intp> npy_dims;
        for (const auto dim : tensor.dims()) npy_dims.push_back(dim);
        int npy_type = TypeMetaToNPY(tensor.meta());
        if (npy_type == -1) {
            LOG(FATAL) <<  "The data type of Tensor(" +
                tensor.name() + ") is unknown. Have you solved it ?";
        }
        CHECK(tensor.memory()) << "\nIllegal memory access.";
        // Create a empty array with the same shape
        PyObject* array = PyArray_SimpleNew(
            tensor.ndim(), npy_dims.data(), npy_type);
        // Copy the tensor data to the numpy array
        if (tensor.memory_state() == MixedMemory::STATE_AT_CUDA) {
            CUDAContext::MemcpyEx<CPUContext, CUDAContext>(tensor.nbytes(),
                     PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
                                            tensor.raw_data<CUDAContext>(),
                                             tensor.memory()->device_id());
        } else {
            CPUContext::Memcpy<CPUContext, CPUContext>(tensor.nbytes(),
                 PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
                                        tensor.raw_data<CPUContext>());
        }
        return pybind11::reinterpret_steal<pybind11::object>(array);
    }
};

class StringFetcher : public TensorFetcherBase {
 public:
    pybind11::object Fetch(const Tensor& tensor) override {
        CHECK_EQ(tensor.count(), 1);
        return pybind11::bytes(tensor.data<string, CPUContext>()[0]);
    }
};

class NumpyFeeder : public TensorFeederBase {
 public:
    void Feed(
        const DeviceOption&         option,
        PyArrayObject*              original_array,
        Tensor*                     tensor) override {
        PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
        const TypeMeta& meta = TypeNPYToMeta(PyArray_TYPE(array));
        if (meta.id() == 0) LOG(FATAL) << "Unsupported data type.";
        tensor->SetMeta(meta);
        int ndim = PyArray_NDIM(array);
        npy_intp* npy_dims = PyArray_DIMS(array);
        vector<int64_t> dims;
        for (int i = 0; i < ndim; i++) dims.push_back(npy_dims[i]);
        tensor->Reshape(dims);
        if (option.device_type() == PROTO_CUDA) {
#ifdef WITH_CUDA
            CUDAContext::MemcpyEx<CUDAContext, CPUContext>(
                                          tensor->nbytes(),
                   tensor->raw_mutable_data<CUDAContext>(),
                   static_cast<void*>(PyArray_DATA(array)),
                                       option.device_id());
#else
            LOG(FATAL) << "CUDA was not compiled.";
#endif
        } else {
            auto* data = tensor->raw_mutable_data<CPUContext>();
            CPUContext::Memcpy<CPUContext, CPUContext>(
                                      tensor->nbytes(),
                tensor->raw_mutable_data<CPUContext>(),
              static_cast<void*>(PyArray_DATA(array)));
        }
        Py_XDECREF(array);
    }
};

Workspace* ws();

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_DRAGON_H_