#include <dragon/core/workspace.h>
#include <dragon/onnx/onnx_backend.h>
#include <dragon/utils/proto_utils.h>

#include "dragon/modules/runtime/dragon_runtime.h"

namespace dragon {

std::mutex g_mutex;
Map<string, unique_ptr<Workspace>> g_workspaces;
Map<string, vector<string>> g_workspace_map;

Workspace* CreateWorkspace(const string& name) {
  std::unique_lock<std::mutex> lock(g_mutex);
  LOG(INFO) << "Create the Workspace(" << name << ").";
  if (g_workspaces.count(name)) return g_workspaces[name].get();
  unique_ptr<Workspace> new_workspace(new Workspace(name));
  g_workspaces[name] = std::move(new_workspace);
  g_workspace_map[name] = vector<string>();
  return g_workspaces[name].get();
}

Workspace* ResetWorkspace(const string& name) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CHECK(g_workspaces.count(name))
      << "\nWorkspace(" << name << ") does not exist.";
  LOG(INFO) << "Reset the Workspace(" << name << ").";
  g_workspaces[name].reset(new Workspace(name));
  for (const auto& child : g_workspace_map[name]) {
    if (g_workspaces.count(child) > 0) {
      g_workspaces[name]->MergeFrom(g_workspaces[child].get());
    }
  }
  return g_workspaces[name].get();
}

Workspace* ResetWorkspace(Workspace* ws) {
  CHECK(ws) << "\nGiven workspace is invalid.";
  return ResetWorkspace(ws->name());
}

void MoveWorkspace(Workspace* dest, Workspace* src) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CHECK(src) << "\nGiven source workspace is invalid.";
  CHECK(dest) << "\nGiven destination workspace is invalid.";
  dest->MergeFrom(src);
  g_workspace_map[dest->name()].push_back(src->name());
  LOG(INFO) << "Move the Workspace(" << src->name() << ") "
            << "into the Workspace(" << dest->name() << ").";
}

void DestroyWorkspace(const string& name) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CHECK(g_workspaces.count(name))
      << "\nWorkspace(" << name << ") does not exist."
      << "\nCan not be released.";
  LOG(INFO) << "Destroy the Workspace(" << name << ").";
  g_workspaces.erase(name);
}

void DestroyWorkspace(Workspace* ws) {
  CHECK(ws) << "\nGiven workspace is invalid.";
  return DestroyWorkspace(ws->name());
}

string CreateGraph(const GraphDef_t def, const Device& device, Workspace* ws) {
  auto def_v2(*def);
  auto* device_option = def_v2.mutable_device_option();
  if (device.device_type() == "CUDA") {
    device_option->set_device_type(DeviceTypeProto::PROTO_CUDA);
  } else if (device.device_type() == "MPS") {
    device_option->set_device_type(DeviceTypeProto::PROTO_MPS);
  } else {
    device_option->set_device_type(DeviceTypeProto::PROTO_CPU);
  }
  device_option->set_device_id(device.device_index());
  auto* graph = ws->CreateGraph(def_v2);
  if (!graph) LOG(FATAL) << "Can not create the graph.";
  return graph->name();
}

string CreateGraph(const string& file, const Device& device, Workspace* ws) {
  GraphDef graph_def;
  ParseProtoFromText(file.c_str(), &graph_def);
  return CreateGraph(&graph_def, device, ws);
}

void RunGraph(const string& name, Workspace* ws, int stream) {
  ws->RunGraph(name, "", "", stream);
}

void CreateTensor(const string& name, Workspace* ws) {
  ws->CreateTensor(name);
}

template <typename T>
T* FetchTensor(
    const string& name,
    vector<int64_t>& shape,
    Workspace* ws,
    bool copy) {
  auto* tensor = ws->GetTensor(name);
  auto* memory = tensor->memory();
  CHECK(memory) << "\nConvert an empty tensor.";
  shape = tensor->dims();
  if (copy) {
    void* data = malloc(tensor->nbytes());
    auto device_type = memory ? memory->info()["device_type"] : "cpu";
    if (device_type == "cuda") {
#ifdef USE_CUDA
      CUDADeviceGuard guard(memory->device());
      CUDAContext::Memcpy<CPUContext, CUDAContext>(
          tensor->nbytes(),
          data,
          tensor_->raw_data<CUDAContext>(),
          memory->device());
#else
      CUDA_NOT_COMPILED;
#endif
    } else {
      CPUContext::Memcpy<CPUContext, CPUContext>(
          tensor->nbytes(), data, tensor->raw_data<CPUContext>());
    }
    return static_cast<T*>(data);
  } else {
    return const_cast<T*>(
        static_cast<const T*>(tensor->raw_data<CPUContext>()));
  }
}

template <typename T>
void FeedTensor(
    const string& name,
    const vector<int64_t>& shape,
    const T* data,
    const Device& device,
    Workspace* ws) {
  auto* tensor = ws->CreateTensor(name)->Reshape(shape);
  if (device.device_type() == "CUDA") {
#ifdef USE_CUDA
    CUDADeviceGuard guard(device.device_index());
    tensor->mutable_data<T, CUDAContext>();
    CUDAContext::Memcpy<CUDAContext, CPUContext>(
        tensor->nbytes(),
        tensor->raw_mutable_data<CUDAContext>(),
        static_cast<const void*>(data));
#else
    CUDA_NOT_COMPILED;
#endif
  } else {
    tensor->mutable_data<T, CPUContext>();
    CPUContext::Memcpy<CPUContext, CPUContext>(
        tensor->nbytes(),
        tensor->raw_mutable_data<CPUContext>(),
        static_cast<const void*>(data));
  }
}

void CreateGraphDef(GraphDef** def) {
  *def = new GraphDef();
}

void DestroyGraphDef(GraphDef* def) {
  if (def) delete def;
}

void LoadONNXModel(
    const string& model_file,
    GraphDef* init_def,
    GraphDef* pred_def,
    vector<string>& inputs,
    vector<string>& outputs) {
  LOG(INFO) << "Load Model: " << model_file << "......";
  LOG(INFO) << "Format: ONNX";
  onnx::ONNXBackend onnx_backend;
  onnx_backend.Prepare(model_file, init_def, pred_def);
  inputs.clear();
  outputs.clear();
  for (const auto& input : pred_def->input()) {
    inputs.push_back(input);
  }
  for (const auto& output : pred_def->output()) {
    outputs.push_back(output);
  }
}

#define INSTANTIATE_API(T)                \
  template DRAGON_API void FeedTensor<T>( \
      const string&,                      \
      const vector<int64_t>&,             \
      const T*,                           \
      const Device&,                      \
      Workspace*);

INSTANTIATE_API(float);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
#undef INSTANTIATE_API

#define INSTANTIATE_API(T)               \
  template DRAGON_API T* FetchTensor<T>( \
      const string&, vector<int64_t>&, Workspace*, const bool);

INSTANTIATE_API(float16);
INSTANTIATE_API(float);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
#undef INSTANTIATE_API

} // namespace dragon
