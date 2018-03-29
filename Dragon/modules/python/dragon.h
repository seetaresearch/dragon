// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_MODULES_PYTHON_DRAGON_H_
#define DRAGON_MODULES_PYTHON_DRAGON_H_

#include <sstream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "core/common.h"
#include "core/registry.h"
#include "core/context.h"
#include "core/context_cuda.h"
#include "core/operator.h"
#include "core/operator_gradient.h"
#include "core/workspace.h"

#ifdef WITH_PYTHON3
#define PyString_AsString PyUnicode_AsUTF8
#endif

using namespace dragon;

inline std::string PyBytesToStdString(PyObject* pystring) {
    return std::string(PyBytes_AsString(pystring), PyBytes_Size(pystring));
}

inline PyObject* StdStringToPyBytes(const std::string& str) {
    return PyBytes_FromStringAndSize(str.c_str(), str.size());
}

inline PyObject* StdStringToPyUnicode(const std::string& str) {
#ifdef WITH_PYTHON3
    return PyUnicode_FromStringAndSize(str.c_str(), str.size());
#else
    return PyBytes_FromStringAndSize(str.c_str(), str.size());
#endif
}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) { ss << t; }

template <typename T,typename ... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args& ... args) {
    MakeStringInternal(ss, t);
    MakeStringInternal(ss, args...);
}

template <typename ... Args>
std::string MakeString(const Args&... args) {
    std::stringstream ss;
    MakeStringInternal(ss, args...);
    return std::string(ss.str());
}

inline void PrErr_SetString(PyObject* type, const std::string& str) { 
    PyErr_SetString(type, str.c_str()); 
}

class TensorFetcherBase {
 public:
    virtual ~TensorFetcherBase() {}
    virtual PyObject* Fetch(const Tensor& tensor) = 0;
};

class TensorFeederBase {
 public:
    virtual ~TensorFeederBase() {}
    virtual PyObject* Feed(const DeviceOption& option, 
                           PyArrayObject* array, 
                           Tensor* tensor) = 0;
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

int DragonToNumpyType(const TypeMeta& meta);
const TypeMeta& NumpyTypeToDragon(int numpy_type);

class NumpyFetcher : public TensorFetcherBase {
 public:
    PyObject* Fetch(const Tensor& tensor) override {
        CHECK_GT(tensor.count(), 0);
        vector<npy_intp> npy_dims;
        for (const auto dim : tensor.dims()) npy_dims.push_back(dim);
        int numpy_type = DragonToNumpyType(tensor.meta());
        if (numpy_type == -1) {
            string s = "The data type of Tensor(" + tensor.name() + ") is unknown. Have you solved it ?";
            PyErr_SetString(PyExc_RuntimeError, s.c_str());
            return nullptr;
        }
        //  create a empty array with r shape
        PyObject* array = PyArray_SimpleNew(tensor.ndim(), npy_dims.data(), numpy_type);
        //  copy the tensor data to the numpy array
        if (tensor.memory_state() == MixedMemory::STATE_AT_CUDA) {
            CUDAContext::Memcpy<CPUContext, CUDAContext>(tensor.nbytes(),
                                                         PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), 
                                                                               tensor.raw_data<CUDAContext>());
        } else {
            CPUContext::Memcpy<CPUContext, CPUContext>(tensor.nbytes(),
                                                       PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), 
                                                                              tensor.raw_data<CPUContext>());
        }
        return array;
    }
};

class StringFetcher : public TensorFetcherBase {
 public:
    PyObject* Fetch(const Tensor& tensor) override {
        CHECK_GT(tensor.count(), 0);
        return StdStringToPyBytes(*tensor.data<string, CPUContext>());
    }
};

class NumpyFeeder : public TensorFeederBase {
 public:
    PyObject* Feed(const DeviceOption& option, 
                   PyArrayObject* original_array, 
                   Tensor* tensor) override {
        PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
        const TypeMeta& meta = NumpyTypeToDragon(PyArray_TYPE(array));
        if (meta.id() == 0) {
            PyErr_SetString(PyExc_TypeError, "Unsupported data type.");
            return nullptr;
        }
        if (meta.id() != tensor->meta().id() && tensor->meta().id() != 0)
            LOG(WARNING) << "Feed Tensor(" << tensor->name() << ")"
                         << " with different data type from original one.";
        tensor->SetMeta(meta);
        int ndim = PyArray_NDIM(array);
        npy_intp* npy_dims = PyArray_DIMS(array);
        vector<TIndex> dims;
        for (int i = 0; i < ndim; i++) dims.push_back(npy_dims[i]);
        tensor->Reshape(dims);
        if (option.device_type() == CUDA) {
#ifdef WITH_CUDA
            CUDAContext context(option);
            context.SwitchToDevice();
            context.Memcpy<CUDAContext, CPUContext>(tensor->nbytes(), 
                                                    tensor->raw_mutable_data<CUDAContext>(), 
                                                    static_cast<void*>(PyArray_DATA(array)));
#else   
            LOG(FATAL) << "CUDA was not compiled.";
#endif
        } else{
            CPUContext::Memcpy<CPUContext, CPUContext>(tensor->nbytes(), 
                                                       tensor->raw_mutable_data<CPUContext>(),
                                                       static_cast<void*>(PyArray_DATA(array)));
        }
        Py_XDECREF(array);
        Py_RETURN_TRUE;
    }
};

#endif    // DRAGON_MODULES_PYTHON_DRAGON_H_