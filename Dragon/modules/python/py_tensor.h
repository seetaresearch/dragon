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

#ifndef DRAGON_PYTHON_PY_TENSOR_H_
#define DRAGON_PYTHON_PY_TENOSR_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddTensorMethods(pybind11::module& m) {
    /*! \brief Export the Tensor class */
    pybind11::class_<Tensor>(m, "Tensor")
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("dims", &Tensor::dims)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("dtype", [](Tensor* self) {
            return TypeMetaToString(self->meta());
      }).def_property_readonly("device", [](Tensor* self) {
            if (self->has_memory()) {
                Map<string, string> mem_info = self->memory()->info();
                return std::tuple<string, int>(
                    mem_info["device_type"], atoi(
                        mem_info["device_id"].c_str()));
            } else {
                return std::tuple<string, int>("Unknown", 0);
            }
      }).def("ToCPU", [](Tensor* self) {
            CHECK(self->has_memory()) << "\nTensor(" << self->name()
              << ") does not initialize or had been reset.";
            self->memory()->ToCPU();
      }).def("ToCUDA", [](Tensor* self, const int device_id) {
#ifdef WITH_CUDA
           CHECK(self->has_memory()) << "\nTensor(" << self->name()
               << ") does not initialize or had been reset.";
           self->memory()->SwitchToCUDADevice(device_id);
#else
           CUDA_NOT_COMPILED;
#endif
     });

    /*! \brief List all the existing tensors */
    m.def("Tensors", []() { return ws()->GetTensors(); });

    /*! \brief Indicate whether the given tensor is existing */
    m.def("HasTensor", [](
        const string& name) -> bool {
        return ws()->HasTensor(name);
    });

    /*! \brief Return the unique name of given tensor */
    m.def("GetTensorName", [](
        const string& name) -> string {
        return ws()->GetTensorName(name);
    });

    /*! \brief Create a tensor with the given name */
    m.def("CreateTensor", [](
        const string& name) -> void {
        ws()->CreateTensor(name);
    });

    /*! \brief Create a tensor with the given name */
    m.def("ResetTensor", [](
        const string& name) -> void {
        ws()->ResetTensor(name);
    });

    /*! \brief Create a tensor with the given shape */
    m.def("TensorFromShape", [](
        const string&               name,
        const vector<int64_t>&      shape,
        const string&               dtype) {
        const TypeMeta& meta = TypeStringToMeta(dtype);
        if (meta.id() == 0) {
            LOG(FATAL) << "Unsupported data type: " + dtype + ".";
        }
        Tensor* tensor = ws()->CreateTensor(name);
        if (meta.id() != tensor->meta().id() && tensor->meta().id() != 0)
            LOG(WARNING) << "Set Tensor(" << tensor->name() << ")"
            << " with different data type from original one.";
        tensor->Reshape(shape);
        tensor->raw_mutable_data<CPUContext>(meta);
    });

    /*! \brief Create a tensor with the given array */
    m.def("TensorFromPyArray", [](
        const string&               name,
        pybind11::object            py_array) {
        PyArrayObject* array = PyArray_GETCONTIGUOUS(
            reinterpret_cast<PyArrayObject*>(py_array.ptr()));
        const TypeMeta& meta = TypeNPYToMeta(PyArray_TYPE(array));
        if (meta.id() == 0) LOG(FATAL) << "Unsupported data type.";
        Tensor* tensor = ws()->CreateTensor(name);
        tensor->SetMeta(meta);
        int ndim = PyArray_NDIM(array);
        npy_intp* npy_dims = PyArray_DIMS(array);
        vector<int64_t> dims;
        for (int i = 0; i < ndim; i++) dims.push_back(npy_dims[i]);
        tensor->Reshape(dims);
        auto* data = static_cast<void*>(PyArray_DATA(array));
        if (!tensor->has_memory()) {
            MixedMemory* memory(new MixedMemory());
            memory->set_cpu_data(data, tensor->nbytes());
            tensor->set_memory(memory);
        } else {
            if (tensor->DECREFPyArray) tensor->DECREFPyArray();
            tensor->memory()->set_cpu_data(data, tensor->nbytes());
        }
        // Follow the codes of PyTorch
        // Here we bind the DECREF to Tensor
        // ResetTensor() or ResetWorkspace() can trigger it
        tensor->DECREFPyArray = [array]()->void { Py_XDECREF(array); };
    });

    /*! \brief Create a tensor copied from an existing one */
    m.def("TensorFromTensor", [](
        const string&               name,
        const string&               other,
        const string&               dev1,
        const string&               dev2) {
        DeviceOption dst_ctx, src_ctx;
        dst_ctx.ParseFromString(dev1);
        src_ctx.ParseFromString(dev2);
        Tensor* srcT = ws()->GetTensor(other);
        Tensor* dstT = ws()->CreateTensor(name);
        dstT->ReshapeLike(*srcT);
        const TypeMeta& meta = srcT->meta();
        if (dst_ctx.device_type() == PROTO_CUDA) {
            if (src_ctx.device_type() == PROTO_CUDA) {
                // CUDA <- CUDA
                CUDAContext::MemcpyEx<CUDAContext, CUDAContext>(
                    srcT->nbytes(),
                        dstT->raw_mutable_data<CUDAContext>(meta),
                            srcT->raw_data<CUDAContext>(),
                                src_ctx.device_id());
            } else {
                // CUDA <- CPU
                CUDAContext::MemcpyEx<CUDAContext, CPUContext>(
                    srcT->nbytes(),
                        dstT->raw_mutable_data<CUDAContext>(meta),
                            srcT->raw_data<CPUContext>(),
                                dst_ctx.device_id());
            }
        } else {
            if (src_ctx.device_type() == PROTO_CUDA) {
                // CPU <- CUDA
                CUDAContext::MemcpyEx<CPUContext, CUDAContext>(
                    srcT->nbytes(),
                        dstT->raw_mutable_data<CPUContext>(meta),
                            srcT->raw_data<CUDAContext>(),
                                src_ctx.device_id());
            } else {
                // CPU <- CPU
                CPUContext::Memcpy<CUDAContext, CUDAContext>(
                    srcT->nbytes(),
                        dstT->raw_mutable_data<CPUContext>(meta),
                            srcT->raw_data<CPUContext>());
            }
        }
    });

    /*! \brief Return a array zero-copied from an existing tensor */
    m.def("TensorToPyArray", [](
        const string&               name,
        const bool                  readonly) {
        Tensor* tensor = ws()->GetTensor(name);
        CHECK_GT(tensor->count(), 0);
        vector<npy_intp> dims;
        for (const auto dim : tensor->dims()) dims.push_back(dim);
        int npy_type = TypeMetaToNPY(tensor->meta());
        if (npy_type == -1) {
            LOG(FATAL) << "Tensor(" + tensor->name() + ") "
                "with dtype." + TypeMetaToString(tensor->meta()) +
                " is not supported by numpy.";
        }
        auto* data = readonly ?
            const_cast<void*>(tensor->raw_data<CPUContext>()) :
                tensor->raw_mutable_data<CPUContext>();
        PyObject* array = PyArray_SimpleNewFromData(
            tensor->ndim(), dims.data(), npy_type, data);
        return pybind11::reinterpret_steal<pybind11::object>(array);
    });

    /*! \brief Create a tensor from the specified filler */
    m.def("CreateFiller", [](
        const string&               serialized) {
        TensorFillerProto filler_proto;
        if (!filler_proto.ParseFromString(serialized))
            LOG(FATAL) << "Failed to parse the TensorFiller.";
        ws()->CreateFiller(filler_proto);
        ws()->CreateTensor(filler_proto.tensor());
    });

    /*! \brief Return the filler type of a tensor */
    m.def("GetFillerType", [](const string& name) {
        return ws()->GetFiller(name)->type();
    });

    /* \brief Set an alias for the tensor */
    m.def("SetTensorAlias", [](
        const string&               name,
        const string&               alias) {
        if (!ws()->HasTensor(name)) {
            LOG(FATAL) << "Tensor(" + name << ") has not "
                "been registered in the current workspace.";
        }
        ws()->SetTensorAlias(name, alias);
    });

    /*! \brief Return the CXX Tensor reference */
    m.def("GetTensor", [](
        const string&               name) {
        return ws()->GetTensor(name);
    }, pybind11::return_value_policy::reference_internal);
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_TENSOR_H_