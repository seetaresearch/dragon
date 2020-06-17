#include "dragon/modules/python/autograd.h"
#include "dragon/modules/python/config.h"
#include "dragon/modules/python/cuda.h"
#include "dragon/modules/python/dlpack.h"
#include "dragon/modules/python/mpi.h"
#include "dragon/modules/python/operator.h"
#include "dragon/modules/python/proto.h"
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
      .def(
          "HasTensor",
          [](Workspace* self, const string& name) {
            return self->HasTensor(name);
          })

      /*! \brief Create a tensor with the given name */
      .def(
          "CreateTensor",
          [](Workspace* self, const string& name) {
            return self->CreateTensor(name);
          },
          py::return_value_policy::reference_internal)

      /*! \brief Create a tensor from the specified filler */
      .def(
          "CreateFiller",
          [](Workspace* self, const string& serialized) {
            TensorFillerProto filler_proto;
            if (!filler_proto.ParseFromString(serialized))
              LOG(FATAL) << "Failed to parse the TensorFiller.";
            self->CreateFiller(filler_proto);
            self->CreateTensor(filler_proto.tensor());
          })

      /*! \brief Return the CXX Tensor reference */
      .def(
          "GetTensor",
          [](Workspace* self, const string& name) {
            return self->TryGetTensor(name);
          },
          py::return_value_policy::reference_internal)

      /* \brief Set an alias for the tensor */
      .def(
          "SetTensorAlias",
          [](Workspace* self, const string& name, const string& alias) {
            return self->ActivateAlias(name, alias);
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

      /*! \brief Copy the tensor data to the array */
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

      /*! \brief Run a operator from the def reference */
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

      /*! \brief Run operators from the def reference */
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

      /*! \brief Run a operator from the serialized def */
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

      /*! \brief Create a graph from the serialized def */
      .def(
          "CreateGraph",
          [](Workspace* self, const string& serialized, const bool verbose) {
            GraphDef graph_def;
            CHECK(graph_def.ParseFromString(serialized))
                << "\nFailed to parse the GraphDef.";
            auto* graph = self->CreateGraph(graph_def);
            if (verbose) {
              bool could_be_serialized = true;
              const auto& def = graph->opt_def();
              for (auto& op : def.op())
                if (op.type() == "GivenTensorFill") could_be_serialized = false;
              if (could_be_serialized) {
                auto msg = string("\n") + def.DebugString();
                msg.pop_back();
                PRINT(INFO) << "graph {" << str::replace_all(msg, "\n", "\n  ")
                            << "\n}\n";
              }
            }
            // Return the graph name may be different from the def
            // We will make a unique dummy name on creating the graph
            return graph->name();
          })

      /*! \brief Run an existing graph */
      .def(
          "RunGraph",
          [](Workspace* self,
             const string& name,
             const string& incl,
             const string& excl) {
            py::gil_scoped_release g;
            self->RunGraph(name, incl, excl);
          })

      .def(
          "RunBackward",
          [](Workspace* self,
             const vector<OperatorDef*>& forward_ops,
             const vector<string>& targets,
             const vector<string>& sources,
             const vector<string>& input_grads,
             const vector<string>& ignored_grads,
             const bool is_sharing,
             const bool verbose) {
            GraphDef backward_ops;
            GraphGradientMaker maker;
            for (auto& name : ignored_grads)
              maker.add_ignored_grad(name);
            for (auto& name : sources)
              maker.add_hooked_grad(name + "_grad");
            maker.Make(forward_ops, targets, input_grads, backward_ops);
            py::gil_scoped_release g;
            if (is_sharing) backward_ops = maker.Share(backward_ops);
            for (auto& def : backward_ops.op()) {
              if (verbose) {
                auto msg = string("\n") + def.DebugString();
                msg.pop_back();
                PRINT(INFO)
                    << "op {" << str::replace_all(msg, "\n", "\n  ") << "\n}\n";
              }
              self->RunOperator(def);
            }
          })

      /*! \brief Serialize tensors into a binary file */
      .def(
          "Save",
          [](Workspace* self,
             const string& filename,
             const vector<string>& tensors,
             const int format) {
            vector<Tensor*> refs;
            switch (format) {
              case 0: // Pickle
                LOG(FATAL) << "Format depends on Pickle. "
                           << "Can't be used in C++.";
                break;
              case 1: // CaffeModel
                for (const auto& e : tensors)
                  refs.emplace_back(self->GetTensor(e));
                SavaCaffeModel(filename, refs);
                break;
              default:
                LOG(FATAL) << "Unknown format, code: " << format;
            }
          })

      /*! \brief Load tensors from a binary file */
      .def(
          "Load",
          [](Workspace* self, const string& filename, const int format) {
            switch (format) {
              case 0: // Pickle
                LOG(FATAL) << "Format depends on Pickle. "
                           << "Can't be used in C++.";
                break;
              case 1: // CaffeModel
                LoadCaffeModel(filename, self);
                break;
              default:
                LOG(FATAL) << "Unknown format, code: " << format;
            }
          })

      /*! \brief Load tensors and graph from a ONNX model */
      .def("ImportONNXModel", [](Workspace* self, const string& model_path) {
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
  REGISTER_MODULE(config);
  REGISTER_MODULE(cuda);
  REGISTER_MODULE(mpi);
  REGISTER_MODULE(ops);
  REGISTER_MODULE(proto);
  REGISTER_MODULE(tensor);
#undef REGISTER_MODULE
}

} // namespace python

} // namespace dragon
