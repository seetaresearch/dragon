#include "dragon/modules/python/autograd.h"
#include "dragon/modules/python/cuda.h"
#include "dragon/modules/python/dlpack.h"
#include "dragon/modules/python/mpi.h"
#include "dragon/modules/python/operator.h"
#include "dragon/modules/python/proto.h"
#include "dragon/modules/python/sysconfig.h"
#include "dragon/modules/python/tensor.h"

namespace dragon {

namespace python {

#define REGISTER_MODULE(namespace) namespace ::RegisterModule(m)

#define REGISTER_TENSOR_FETCHER(type, ...) \
  REGISTER_CLASS(TensorFetcherRegistry, type, __VA_ARGS__)

DEFINE_REGISTRY(TensorFetcherRegistry, TensorFetcherBase);

REGISTER_TENSOR_FETCHER(bool, NumpyFetcher);
REGISTER_TENSOR_FETCHER(int8, NumpyFetcher);
REGISTER_TENSOR_FETCHER(uint8, NumpyFetcher);
REGISTER_TENSOR_FETCHER(int32, NumpyFetcher);
REGISTER_TENSOR_FETCHER(int64, NumpyFetcher);
REGISTER_TENSOR_FETCHER(float16, NumpyFetcher);
REGISTER_TENSOR_FETCHER(float32, NumpyFetcher);
REGISTER_TENSOR_FETCHER(float64, NumpyFetcher);
REGISTER_TENSOR_FETCHER(string, StringFetcher);
#undef REGISTER_TENSOR_FETCHER

PYBIND11_MODULE(libdragon_python, m) {
  /*! \brief Export the Workspace class */
  py::class_<Workspace>(m, "Workspace")
      /*! \brief Default constructor */
      .def(py::init<const string&>())

      /*! \brief Return the name of this workspace */
      .def_property_readonly("name", &Workspace::name)

      /*! \brief Return the name of stored tensors */
      .def_property_readonly("tensors", &Workspace::tensors)

      /*! \brief Return the name of stored graphs */
      .def_property_readonly("graphs", &Workspace::graphs)

      /*! \brief Merge resources from another workspace */
      .def("MergeFrom", &Workspace::MergeFrom)

      /*! \brief Clear the cached resources */
      .def("Clear", &Workspace::Clear)

      /*! \brief Return an unique name */
      .def("UniqueName", &Workspace::UniqueName)

      /*! \brief Reset the tensor */
      .def("ResetTensor", &Workspace::ResetTensor)

      /*! \brief Return whether the tensor is existing */
      .def(
          "HasTensor",
          [](Workspace* self, const string& name) {
            return self->HasTensor(name);
          })

      /*! \brief Create the tensor */
      .def(
          "CreateTensor",
          [](Workspace* self, const string& name, const string& filler_str) {
            if (!filler_str.empty()) {
              FillerInfo filler_info;
              if (!filler_info.ParseFromString(filler_str)) {
                LOG(FATAL) << "Failed to parse the FillerInfo.";
              }
              return self->CreateTensor(name, &filler_info);
            }
            return self->CreateTensor(name);
          },
          py::return_value_policy::reference)

      /*! \brief Return the tensor */
      .def(
          "GetTensor",
          [](Workspace* self, const string& name) {
            return self->TryGetTensor(name);
          },
          py::return_value_policy::reference)

      /* \brief Register an alias for the name */
      .def(
          "RegisterAlias",
          [](Workspace* self, const string& name, const string& alias) {
            return self->RegisterAlias(name, alias);
          })

      /*! \brief Copy the array data to tensor */
      .def(
          "FeedTensor",
          [](Workspace* self,
             const string& name,
             py::object value,
             const string& ctx) {
            DeviceOption dev;
            if (!ctx.empty()) {
              CHECK(dev.ParseFromString(ctx))
                  << "\nFailed to parse the DeviceOption.";
            }
            auto* tensor = self->CreateTensor(name);
            unique_ptr<TensorFeederBase> feeder(new NumpyFeeder());
            feeder->Feed(
                dev, reinterpret_cast<PyArrayObject*>(value.ptr()), tensor);
          })

      /*! \brief Copy the tensor data to array */
      .def(
          "FetchTensor",
          [](Workspace* self, const string& name) {
            CHECK(self->HasTensor(name))
                << "\nTensor(" + name + ") does not exist.\n"
                << "Have you registered it?";
            auto* tensor = self->GetTensor(name);
            string type = ::dragon::types::to_string(tensor->meta());
            if (type == "unknown") {
              LOG(FATAL) << "Tensor(" << tensor->name() << ") "
                         << "does not initialize or had been reset.";
              return py::object();
            } else {
              if (TensorFetcherRegistry()->Has(type)) {
                return TensorFetcherRegistry()->Create(type)->Fetch(*tensor);
              } else {
                LOG(FATAL) << "Type of tensor(" << tensor->name() << ") "
                           << "is not supported to fetch.";
                return py::object();
              }
            }
          })

      /*! \brief Return the size of memory used by tensors on given device */
      .def(
          "MemoryAllocated",
          [](Workspace* self, const string& device_type, int device_id) {
            size_t size = 0;
            for (const auto& name : self->tensors(false)) {
              auto* memory = self->GetTensor(name)->memory(false, true);
              if (memory) {
                if (device_type == "cpu") {
                  size += memory->size();
                } else {
                  size += memory->size(device_type, device_id);
                }
              }
            }
            return size;
          })

      /*! \brief Run the operator */
      .def(
          "RunOperator",
          [](Workspace* self, OperatorDef* def, const bool verbose) {
            py::gil_scoped_release g;
            if (verbose) {
              auto msg = string("\n") + def->DebugString();
              msg.pop_back();
              PRINT(INFO) << "op {" << str::replace_all(msg, "\n", "\n  ")
                          << "\n}\n";
            }
            self->RunOperator(*def);
          })

      /*! \brief Run the operators */
      .def(
          "RunOperator",
          [](Workspace* self, vector<OperatorDef*>& defs, const bool verbose) {
            py::gil_scoped_release g;
            for (auto* def : defs) {
              if (verbose) {
                auto msg = string("\n") + def->DebugString();
                msg.pop_back();
                PRINT(INFO)
                    << "op {" << str::replace_all(msg, "\n", "\n  ") << "\n}\n";
              }
              self->RunOperator(*def);
            }
          })

      /*! \brief Run the operator from serialized def */
      .def(
          "RunOperator",
          [](Workspace* self, const string& serialized, const bool verbose) {
            OperatorDef def;
            CHECK(def.ParseFromString(serialized));
            py::gil_scoped_release g;
            if (verbose) {
              auto msg = string("\n") + def.DebugString();
              msg.pop_back();
              PRINT(INFO) << "op {" << str::replace_all(msg, "\n", "\n  ")
                          << "\n}\n";
            }
            self->RunOperator(def);
          })

      /*! \brief Create the graph */
      .def(
          "CreateGraph",
          [](Workspace* self, const string& serialized, const bool verbose) {
            GraphDef graph_def;
            CHECK(graph_def.ParseFromString(serialized))
                << "\nFailed to parse the GraphDef.";
            auto* graph = self->CreateGraph(graph_def);
            if (verbose) {
              bool could_be_serialized = true;
              const auto& def = graph->optimized_def();
              for (auto& op : def.op()) {
                if (op.type() == "GivenTensorFill") {
                  could_be_serialized = false;
                }
              }
              if (could_be_serialized) {
                auto msg = string("\n") + def.DebugString();
                msg.pop_back();
                LOG(INFO) << "\ngraph {" << str::replace_all(msg, "\n", "\n  ")
                          << "\n}\n";
              }
            }
            // Return the graph name may be different from the def
            // We will make a unique dummy name on creating the graph
            return graph->name();
          })

      /*! \brief Run the graph */
      .def(
          "RunGraph",
          [](Workspace* self,
             const string& name,
             const string& include,
             const string& exclude) {
            py::gil_scoped_release g;
            self->RunGraph(name, include, exclude);
          })

      /*! \brief Run the backward */
      .def(
          "RunBackward",
          [](Workspace* self,
             const vector<OperatorDef*>& op_defs,
             const vector<string>& targets,
             const vector<string>& sources,
             const vector<string>& input_grads,
             const vector<string>& empty_grads,
             const bool retain_grads,
             const bool verbose) {
            GraphDef graph_def;
            GraphGradientMaker maker;
            for (const auto& name : empty_grads) {
              maker.add_empty_grad(name);
            }
            for (const auto& name : sources) {
              maker.add_retained_grad(name + "_grad");
            }
            maker.Make(op_defs, targets, input_grads, graph_def);
            py::gil_scoped_release g;
            if (!retain_grads) {
              graph_def = maker.Optimize(graph_def);
            }
            for (const auto& op_def : graph_def.op()) {
              if (verbose) {
                auto msg = string("\n") + op_def.DebugString();
                msg.pop_back();
                PRINT(INFO)
                    << "op {" << str::replace_all(msg, "\n", "\n  ") << "\n}\n";
              }
              self->RunOperator(op_def);
            }
          })

      /*! \brief Load tensors and graph from a ONNX model */
      .def("PrepareONNXModel", [](Workspace* self, const string& model_path) {
        GraphDef init_graph, pred_graph;
        onnx::ONNXBackend onnx_backend;
        onnx_backend.Prepare(model_path, &init_graph, &pred_graph);
        // Serializing to Python is intractable
        // We should apply the initializer immediately
        self->RunGraph(self->CreateGraph(init_graph)->name(), "", "");
        return py::bytes(pred_graph.SerializeAsString());
      });

  // Initialization once importing
  []() { import_array1(); }();

  REGISTER_MODULE(autograd);
  REGISTER_MODULE(cuda);
  REGISTER_MODULE(mpi);
  REGISTER_MODULE(ops);
  REGISTER_MODULE(proto);
  REGISTER_MODULE(sysconfig);
  REGISTER_MODULE(tensor);
#undef REGISTER_MODULE
}

} // namespace python

} // namespace dragon
