#include <dragon/core/workspace.h>
#include <dragon/onnx/onnx_backend.h>

#include "dragon/modules/python/cuda.h"
#include "dragon/modules/python/gradient.h"
#include "dragon/modules/python/mlu.h"
#include "dragon/modules/python/mpi.h"
#include "dragon/modules/python/mps.h"
#include "dragon/modules/python/proto.h"
#include "dragon/modules/python/sysconfig.h"
#include "dragon/modules/python/tensor.h"

namespace dragon {

namespace python {

#define REGISTER_MODULE(module_name) RegisterModule_##module_name(m)

inline string GetDebugDef(const string& def_str, const string& type) {
  auto s = string("\n") + def_str;
  s.pop_back();
  return type + " {" + str::replace_all(s, "\n", "\n  ") + "\n}\n";
}

PYBIND11_MODULE(libdragon_python, m) {
  /*! \brief Workspace class */
  py::class_<Workspace>(m, "Workspace")
      /*! \brief Default constructor */
      .def(py::init<const string&>())

      /*! \brief Return the name of this workspace */
      .def_property_readonly("name", &Workspace::name)

      /*! \brief Return the name of stored tensors */
      .def_property_readonly(
          "tensors", [](Workspace* self) { return self->tensors(); })

      /*! \brief Return the name of stored graphs */
      .def_property_readonly("graphs", &Workspace::graphs)

      /*! \brief Merge resources from another workspace */
      .def("MergeFrom", &Workspace::MergeFrom)

      /*! \brief Release the created resources */
      .def("Clear", &Workspace::Clear)

      /* \brief Set an alias for the target */
      .def("SetAlias", &Workspace::SetAlias)

      /*! \brief Return an unique name */
      .def("UniqueName", &Workspace::UniqueName)

      /*! \brief Create a tensor */
      .def(
          "CreateTensor",
          &Workspace::CreateTensor,
          py::return_value_policy::reference)

      /*! \brief Return the tensor */
      .def(
          "GetTensor",
          [](Workspace* self, const string& name) {
            return self->TryGetTensor(name);
          },
          py::return_value_policy::reference)

      /*! \brief Return the size of memory used by tensors on given device */
      .def(
          "MemoryAllocated",
          [](Workspace* self, const string& device_type, int device_id) {
            size_t size = 0;
            for (const auto& name : self->tensors(false)) {
              auto* memory = self->GetTensor(name)->memory(false, true);
              if (memory == nullptr) continue;
              if (device_type == "cpu") {
                size += memory->size();
              } else {
                size += memory->size(device_type, device_id);
              }
            }
            return size;
          })

      /*! \brief Run an operator */
      .def(
          "RunOperator",
          [](Workspace* self, OperatorDef* def, int stream, bool verbose) {
            py::gil_scoped_release g;
            if (verbose) PRINT(INFO) << GetDebugDef(def->DebugString(), "op");
            self->RunOperator(*def, stream);
          })

      /*! \brief Run a list of operators */
      .def(
          "RunOperator",
          [](Workspace* self,
             vector<OperatorDef*>& defs,
             int stream,
             bool verbose) {
            py::gil_scoped_release g;
            for (auto* def : defs) {
              if (verbose) PRINT(INFO) << GetDebugDef(def->DebugString(), "op");
              self->RunOperator(*def, stream);
            }
          })

      /*! \brief Run a serialized operator */
      .def(
          "RunOperator",
          [](Workspace* self,
             const string& serialized,
             int stream,
             bool verbose) {
            OperatorDef def;
            CHECK(def.ParseFromString(serialized));
            py::gil_scoped_release g;
            if (verbose) PRINT(INFO) << GetDebugDef(def.DebugString(), "op");
            self->RunOperator(def, stream);
          })

      /*! \brief Create a graph */
      .def(
          "CreateGraph",
          [](Workspace* self, const string& serialized, bool verbose) {
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
                PRINT(INFO) << GetDebugDef(def.DebugString(), "graph");
              }
            }
            // Return the graph name may be different from the def.
            // We will make a unique dummy name on creating the graph.
            return graph->name();
          })

      /*! \brief Run a graph */
      .def(
          "RunGraph",
          [](Workspace* self, const string& name, int stream) {
            py::gil_scoped_release g;
            self->RunGraph(name, stream);
          })

      /*! \brief Run a backward pass */
      .def(
          "RunBackward",
          [](Workspace* self,
             const vector<OperatorDef*>& op_defs,
             const vector<string>& targets,
             const vector<string>& grad_grads,
             const vector<string>& sources,
             int stream,
             bool optimize,
             bool verbose) {
            GradientTape tape;
            tape.CreateGradientDefs(op_defs, targets, grad_grads);
            py::gil_scoped_release g;
            if (optimize) tape.Optimize(sources);
            for (const auto& op : tape.def().op()) {
              if (verbose) PRINT(INFO) << GetDebugDef(op.DebugString(), "op");
              self->RunOperator(op, stream);
            }
          })

      /*! \brief Load tensors and graph from a ONNX model */
      .def("PrepareONNXModel", [](Workspace* self, const string& model_path) {
        GraphDef init_graph, pred_graph;
        onnx::ONNXBackend onnx_backend;
        onnx_backend.Prepare(model_path, &init_graph, &pred_graph);
        // Serializing to Python is intractable.
        // We should apply the initializer immediately.
        self->RunGraph(self->CreateGraph(init_graph)->name());
        return py::bytes(pred_graph.SerializeAsString());
      });

  // Initialization once importing
  []() { import_array1(); }();

  REGISTER_MODULE(cuda);
  REGISTER_MODULE(gradient);
  REGISTER_MODULE(mlu);
  REGISTER_MODULE(mpi);
  REGISTER_MODULE(mps);
  REGISTER_MODULE(proto);
  REGISTER_MODULE(sysconfig);
  REGISTER_MODULE(tensor);
}

#undef REGISTER_MODULE

} // namespace python

} // namespace dragon
