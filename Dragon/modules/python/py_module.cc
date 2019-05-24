#include "py_autograd.h"
#include "py_operator.h"
#include "py_tensor.h"
#include "py_cuda.h"
#include "py_mpi.h"
#include "py_config.h"
#include "py_proto.h"

namespace dragon {

namespace python {

DEFINE_TYPED_REGISTRY(TensorFetcherRegistry, TypeId, TensorFetcherBase);
DEFINE_TYPED_REGISTRY(TensorFeederRegistry, TypeId, TensorFeederBase);

TypeId CTypeToFetcher(TypeId type) {
    static Map<TypeId,TypeId> c_type_map {
        { TypeMeta::Id<bool>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int8_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<uint8_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int64_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<float16>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<float>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<double>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<string>(), TypeMeta::Id<StringFetcher>() }};
    return c_type_map.count(type) ? c_type_map[type] : 0;
}

REGISTER_TENSOR_FETCHER(TypeMeta::Id<NumpyFetcher>(), NumpyFetcher);
REGISTER_TENSOR_FETCHER(TypeMeta::Id<StringFetcher>(), StringFetcher);
REGISTER_TENSOR_FEEDER(TypeMeta::Id<NumpyFeeder>(), NumpyFeeder);

void OnImportModule() { []() { import_array1(); }(); }

PYBIND11_MODULE(libdragon, m) {
    /*! \brief Export the Workspace class */
    pybind11::class_<Workspace>(m, "Workspace")
        .def(pybind11::init<const string&>())

        /*! \brief Return the name of this workspace */
        .def_property_readonly("name", &Workspace::name)

        /*! \brief Return the name of stored tensors */
        .def_property_readonly("tensors", &Workspace::tensors)

        /*! \brief Return the name of stored graphs */
        .def_property_readonly("graphs", &Workspace::graphs)

        /*! \brief Destory all the tensors */
        .def("Clear", &Workspace::Clear)

        /*! \brief Merge a external workspace into self */
        .def("MergeFrom", &Workspace::MergeFrom)

        /*! \brief Return a unique dummy name */
        .def("GetDummyName", &Workspace::GetDummyName)

        /*! \brief Return the unique name of given tensor */
        .def("GetTensorName", &Workspace::GetTensorName)

        /*! \brief Reset a tensor with the given name */
        .def("ResetTensor", &Workspace::ResetTensor)

        /*! \brief Indicate whether the given tensor is existing */
        .def("HasTensor", [](
            Workspace*                  self,
            const string&               name) {
            return self->HasTensor(name);
        })

        /*! \brief Create a tensor with the given name */
        .def("CreateTensor", [](
            Workspace*                  self,
            const string&               name) {
            self->CreateTensor(name);
        })

        /*! \brief Create a tensor from the specified filler */
        .def("CreateFiller", [](
            Workspace*                  self,
            const string&               serialized) {
            TensorFillerProto filler_proto;
            if (!filler_proto.ParseFromString(serialized))
                LOG(FATAL) << "Failed to parse the TensorFiller.";
            self->CreateFiller(filler_proto);
            self->CreateTensor(filler_proto.tensor());
        })

        /*! \brief Create a tensor with the given shape */
        .def("TensorFromShape", [](
            Workspace*                  self,
            const string&               name,
            const vector<int64_t>&      shape,
            const string&               dtype) {
            const TypeMeta& meta = TypeStringToMeta(dtype);
            CHECK(meta.id() != 0)
                << "\nUnsupported data type: " + dtype + ".";
            Tensor* tensor = self->CreateTensor(name);
            tensor->Reshape(shape);
            tensor->raw_mutable_data<CPUContext>(meta);
        })

        /*! \brief Create a tensor with the given array */
        .def("TensorFromArray", [](
            Workspace*                  self,
            const string&               name,
            pybind11::object            object) {
            PyArrayObject* array = PyArray_GETCONTIGUOUS(
                reinterpret_cast<PyArrayObject*>(object.ptr()));
            const TypeMeta& meta = TypeNPYToMeta(PyArray_TYPE(array));
            if (meta.id() == 0) LOG(FATAL) << "Unsupported DType.";
            Tensor* tensor = self->CreateTensor(name);
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
        })

        /*! \brief Create a tensor copied from an existing one */
        .def("TensorFromTensor", [](
            Workspace*                  self,
            const string&               name,
            const string&               other,
            const string&               dev1,
            const string&               dev2) {
            DeviceOption dst_ctx, src_ctx;
            dst_ctx.ParseFromString(dev1);
            src_ctx.ParseFromString(dev2);
            auto* src = self->GetTensor(other);
            auto* dst = self->CreateTensor(name);
            const auto& meta = src->meta();
            dst->ReshapeLike(*src);
            if (dst_ctx.device_type() == PROTO_CUDA) {
                if (src_ctx.device_type() == PROTO_CUDA) {
                    // CUDA <- CUDA
                    CUDAContext::MemcpyEx<CUDAContext, CUDAContext>(
                        src->nbytes(),
                        dst->raw_mutable_data<CUDAContext>(meta),
                        src->raw_data<CUDAContext>(),
                        src_ctx.device_id()
                    );
                } else {
                    // CUDA <- CPU
                    CUDAContext::MemcpyEx<CUDAContext, CPUContext>(
                        src->nbytes(),
                        dst->raw_mutable_data<CUDAContext>(meta),
                        src->raw_data<CPUContext>(),
                        dst_ctx.device_id()
                    );
                }
            } else {
                if (src_ctx.device_type() == PROTO_CUDA) {
                    // CPU <- CUDA
                    CUDAContext::MemcpyEx<CPUContext, CUDAContext>(
                        src->nbytes(),
                        dst->raw_mutable_data<CPUContext>(meta),
                        src->raw_data<CUDAContext>(),
                        src_ctx.device_id()
                    );
                } else {
                    // CPU <- CPU
                    CPUContext::Memcpy<CUDAContext, CUDAContext>(
                        src->nbytes(),
                        dst->raw_mutable_data<CPUContext>(meta),
                        src->raw_data<CPUContext>()
                    );
                }
            }
        })

        /*! \brief Return a array zero-copied from an existing tensor */
        .def("TensorToArray", [](
            Workspace*                  self,
            const string&               name,
            const bool                  readonly) {
            Tensor* tensor = self->GetTensor(name);
            CHECK_GT(tensor->count(), 0);
            vector<npy_intp> dims;
            for (auto dim : tensor->dims()) dims.push_back(dim);
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
        })

        /*! \brief Return the CXX Tensor reference */
        .def("GetTensor", [](
            Workspace*                  self,
            const string&               name) {
            return self->GetTensor(name);
        }, pybind11::return_value_policy::reference_internal)

        /* \brief Set an alias for the tensor */
        .def("SetTensorAlias", [](
            Workspace*                  self,
            const string&               name,
            const string&               alias) {
            CHECK(self->HasTensor(name))
                << "\nTensor(" + name << ") has not been "
                << "registered in the current workspace.";
            self->SetTensorAlias(name, alias);
        })

        /*! \brief Copy the array data to tensor */
        .def("FeedTensor", [](
            Workspace*                  self,
            const string&               name,
            pybind11::object            value,
            const string&               ctx) {
            DeviceOption dev;
            if (!ctx.empty()) {
                CHECK(dev.ParseFromString(ctx))
                    << "\nFailed to parse the DeviceOption.";
            }
            Tensor* tensor = self->CreateTensor(name);
            unique_ptr<TensorFeederBase> feeder(
                TensorFeederRegistry()->Create(
                    TypeMeta::Id<NumpyFeeder>()));
            feeder->Feed(dev, reinterpret_cast
                <PyArrayObject*>(value.ptr()), tensor);
        })

        /*! \brief Copy the tensor data to the array */
       .def("FetchTensor", [](
            Workspace*                  self,
            const string&               name) {
            CHECK(self->HasTensor(name))
                << "\nTensor(" + name + ") does not exist.\n"
                << "Have you registered it?";
            Tensor* tensor = self->GetTensor(name);
            TypeId type_id = CTypeToFetcher(tensor->meta().id());
            CHECK(type_id != 0)
                << "\nTensor(" << tensor->name()
                << ") does not initialize or had been reset.";
            unique_ptr<TensorFetcherBase> fetcher(CreateFetcher(type_id));
            if (fetcher.get()) {
                // Copy the tensor data to a numpy object
                return fetcher->Fetch(*tensor);
            } else {
                LOG(FATAL) << name << " is not a C++ native type.";
                return pybind11::object();
            }
        })

        /*! \brief Run a operator from the def reference */
        .def("RunOperator", [](
            Workspace*                  self,
            OperatorDef*                def,
            const bool                  verbose) {
            pybind11::gil_scoped_release g;
            if (verbose) {
                // It is not a good design to print the debug string
                std::cout << def->DebugString() << std::endl;
            }
            self->RunOperator(*def);
        })

        /*! \brief Run a operator from the serialized def */
        .def("RunOperator", [](
            Workspace*                  self,
            const string&               serialized,
            const bool                  verbose) {
            OperatorDef def;
            CHECK(def.ParseFromString(serialized));
            pybind11::gil_scoped_release g;
            if (verbose) {
                // It is not a good design to print the debug string
                std::cout << def.DebugString() << std::endl;
            }
            self->RunOperatorOnce(def);
        })

        /*! \brief Create a graph from the serialized def */
        .def("CreateGraph", [](
            Workspace*                  self,
            const string&               serialized,
            const bool                  verbose) {
            GraphDef graph_def;
            CHECK(graph_def.ParseFromString(serialized))
                << "\nFailed to parse the GraphDef.";
            auto* graph = self->CreateGraph(graph_def);
            if (verbose) {
                bool could_be_serialized = true;
                const auto& def = graph->opt_def();
                for (auto& op : def.op())
                    if (op.type() == "GivenTensorFill")
                        could_be_serialized = false;
                if (could_be_serialized) {
                    // It is not a good design to print the debug string
                    std::cout << def.DebugString() << std::endl;
                }
            }
            // Return the graph name may be different from the def
            // We will make a unique dummy name on creating the graph
            return graph->name();
        })

        /*! \brief Run an existing graph */
        .def("RunGraph", [](
            Workspace*                  self,
            const string&               name,
            const string&               include,
            const string&               exclude) {
            pybind11::gil_scoped_release g;
            self->RunGraph(name, include, exclude);
        })

        .def("Backward", [](
            Workspace*                      self,
            const vector<OperatorDef*>&     forward_ops,
            const vector<string>&           targets,
            const vector<string>&           input_grads,
            const vector<string>&           ignore_grads,
            const bool                      is_sharing,
            const bool                      verbose) {
            // Make => Optimize => Run
            GraphDef backward_ops;
            GraphGradientMaker maker;
            for (auto& e : input_grads) maker.AddExternalGrad(e);
            for (auto& e : ignore_grads) maker.AddIgnoreGrad(e);
            maker.Make(forward_ops, targets, backward_ops);
            pybind11::gil_scoped_release g;
            if (is_sharing) backward_ops = maker.Share(backward_ops);
            for (auto& op : backward_ops.op()) {
                if (verbose) std::cout << op.DebugString() << std::endl;
                if (op.has_uid()) self->RunOperator(op);
                else self->RunOperatorOnce(op);
            }
        })

        /*! \brief Serialize tensors into a binary file */
        .def("Snapshot", [](
            Workspace*                  self,
            const string&               filename,
            const vector<string>&       tensors,
            const int                   format) {
            vector<Tensor*> refs;
            switch (format) {
                case 0:  // Pickle
                    LOG(FATAL) << "Format depends on Pickle. "
                                  "Can't be used in C++.";
                    break;
                case 1:  // CaffeModel
                    for (const auto& e : tensors)
                        refs.emplace_back(self->GetTensor(e));
                    SavaCaffeModel(filename, refs);
                    break;
                default:
                    LOG(FATAL) << "Unknwon format, code: " << format;
            }
        })

        /*! \brief Load tensors from a binary file */
        .def("Restore", [](
            Workspace*                  self,
            const string&               filename,
            const int                   format) {
                switch (format) {
                case 0:  // Pickle
                    LOG(FATAL) << "Format depends on Pickle. "
                                  "Can't be used in C++.";
                    break;
                case 1:  // CaffeModel
                    LoadCaffeModel(filename, self);
                    break;
                default:
                    LOG(FATAL) << "Unknwon format, code: " << format;
            }
        })

        /*! \brief Load tensors and graph from a ONNX model */
        .def("ImportONNXModel", [](
            Workspace*                  self,
            const string&               model_path) {
            GraphDef init_graph, pred_graph;
            onnx::ONNXBackend onnx_backend;
            onnx_backend.Prepare(model_path, &init_graph, &pred_graph);
            // Serializing to Python is intractable
            // We should apply the initializer immediately
            self->RunGraph(self->CreateGraph(init_graph)->name(), "", "");
            return pybind11::bytes(pred_graph.SerializeAsString());
        });

    AddMPIMethods(m);
    AddCUDAMethods(m);
    AddProtoMethods(m);
    AddTensorMethods(m);
    AddConfigMethods(m);
    AddGradientMethods(m);
    AddOperatorMethods(m);
    OnImportModule();
}

}  // namespace python

}  // namespace dragon