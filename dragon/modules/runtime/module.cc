#include "dragon/core/workspace.h"
#include "dragon/modules/runtime/dragon_runtime.h"
#include "dragon/onnx/onnx_backend.h"
#include "dragon/utils/proto_utils.h"

namespace dragon {

std::mutex g_mutex;
Map<string, unique_ptr<Workspace>> g_workspaces;
Map<string, vector<string>> sub_workspaces;

Workspace_t CreateWorkspace(const string& name) {
  std::unique_lock<std::mutex> lock(g_mutex);
  LOG(INFO) << "Create the Workspace(" << name << ").";
  if (g_workspaces.count(name)) return g_workspaces[name].get();
  unique_ptr<Workspace> new_workspace(new Workspace(name));
  g_workspaces[name] = std::move(new_workspace);
  sub_workspaces[name] = vector<string>();
  return g_workspaces[name].get();
}

Workspace_t ResetWorkspace(const string& name) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CHECK(g_workspaces.count(name))
      << "\nWorkspace(" << name << ") does not exist."
      << "\nCan not be reset.";
  LOG(INFO) << "Reset the Workspace(" << name << ").";
  g_workspaces[name].reset(new Workspace(name));
  for (auto& sub_workspace : sub_workspaces[name]) {
    if (g_workspaces.count(sub_workspace) > 0) {
      g_workspaces[name]->MergeFrom(g_workspaces[sub_workspace].get());
    }
  }
  return g_workspaces[name].get();
}

Workspace_t ResetWorkspace(Workspace_t ws) {
  CHECK(ws) << "\nGiven workspace is invalid.";
  return ResetWorkspace(ws->name());
}

void MoveWorkspace(Workspace_t dest, Workspace_t src) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CHECK(src) << "\nGiven source workspace is invalid.";
  CHECK(dest) << "\nGiven destination workspace is invalid.";
  dest->MergeFrom(src);
  sub_workspaces[dest->name()].push_back(src->name());
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

void DestroyWorkspace(Workspace_t ws) {
  CHECK(ws) << "\nGiven workspace is invalid.";
  return DestroyWorkspace(ws->name());
}

string CreateGraph(const GraphDef_t def, const Device& device, Workspace_t ws) {
  auto def_v2(*def);
  auto* device_option = def_v2.mutable_device_option();
  device_option->set_device_type((DeviceTypeProto)device.device_type());
  device_option->set_device_id(device.device_id());
  auto* graph = ws->CreateGraph(def_v2);
  if (!graph) LOG(FATAL) << "Can not create the graph.";
  return graph->name();
}

std::string
CreateGraph(const string& file, const Device& device, Workspace_t ws) {
  GraphDef graph_def;
  ParseProtoFromText(file.c_str(), &graph_def);
  return CreateGraph(&graph_def, device, ws);
}

void RunGraph(const string& name, Workspace_t ws, int stream) {
  ws->RunGraph(name, "", "", stream);
}

void CreateTensor(const string& name, Workspace_t ws) {
  ws->CreateTensor(name);
}

template <typename T>
T* FetchTensor(
    const string& name,
    vector<int64_t>& shape,
    Workspace_t ws,
    bool copy) {
  if (!ws->HasTensor(name)) {
    LOG(FATAL) << "Tensor(" << name << ")"
               << " doesn't exist, try to create it before.";
  }
  Tensor* tensor = ws->GetTensor(name);
  if (tensor->meta().id() == 0) {
    LOG(FATAL) << "Tensor(" << name << ")"
               << " has not been computed yet";
  }
  shape = tensor->dims();
  if (copy) {
    auto nbytes = tensor->nbytes();
    void* data = malloc(nbytes);
    if (tensor->memory_state() == UnifiedMemory::STATE_AT_CUDA) {
      CUDAContext::Memcpy<CPUContext, CUDAContext>(
          nbytes, data, tensor->raw_data<CUDAContext>());
    } else {
      CPUContext::Memcpy<CPUContext, CPUContext>(
          nbytes, data, tensor->raw_data<CPUContext>());
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
    Workspace_t ws) {
  Tensor* tensor = ws->CreateTensor(name);
  tensor->Reshape(shape);
  if (device.device_type() == 1) {
    CUDAContext context(device.device_id());
    context.SwitchToDevice();
    tensor->mutable_data<T, CUDAContext>();
    context.Memcpy<CUDAContext, CPUContext>(
        tensor->nbytes(),
        tensor->raw_mutable_data<CUDAContext>(),
        static_cast<const void*>(data));
  } else if (device.device_type() == 0) {
    CPUContext context;
    tensor->mutable_data<T, CPUContext>();
    context.Memcpy<CPUContext, CPUContext>(
        tensor->nbytes(),
        tensor->raw_mutable_data<CPUContext>(),
        static_cast<const void*>(data));
  } else {
    LOG(FATAL) << "Unsupported device type.";
  }
}

void CreateGraphDef(GraphDef_t* def) {
  *def = new GraphDef();
}

void DestroyGraphDef(GraphDef_t def) {
  if (def) {
    delete def;
  }
}

void LoadONNXModel(
    const string& model_file,
    GraphDef_t init_def,
    GraphDef_t pred_def,
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
      Workspace_t);

INSTANTIATE_API(float);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
#undef INSTANTIATE_API

#define INSTANTIATE_API(T)               \
  template DRAGON_API T* FetchTensor<T>( \
      const string&, vector<int64_t>&, Workspace_t, const bool);

INSTANTIATE_API(float16);
INSTANTIATE_API(float);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
#undef INSTANTIATE_API

} // namespace dragon
