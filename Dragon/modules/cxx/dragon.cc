#include <mutex>

#include "core/common.h"
#include "utils/proto_utils.h"
#include "utils/caffemodel.h"
#include "onnx/onnx_backend.h"

#include "dragon.h"

namespace dragon {

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *               Workspace               *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

Map<string, unique_ptr < Workspace > > g_workspaces;
Map<string, vector<string> > sub_workspaces;
std::mutex g_mutex;

Workspace* CreateWorkspace(const std::string& name){
    std::unique_lock<std::mutex> lock(g_mutex);
    LOG(INFO) << "Create the Workspace(" << name << ").";
    if (g_workspaces.count(name)) return g_workspaces[name].get();
    unique_ptr<Workspace> new_workspace(new Workspace(name));
    g_workspaces[name] = std::move(new_workspace);
    sub_workspaces[name] = vector<string>();
    return g_workspaces[name].get();
}

Workspace* ResetWorkspace(const std::string& name) {
    std::unique_lock<std::mutex> lock(g_mutex);
    CHECK(g_workspaces.count(name))
        << "\nWorkspace(" << name << ") does not exist."
        << "\nCan not be reset.";
    LOG(INFO) << "Reset the Workspace(" << name << ").";
    g_workspaces[name].reset(new Workspace(name));
    for (auto& sub_workspace : sub_workspaces[name]) {
        if (g_workspaces.count(sub_workspace) > 0)
            g_workspaces[name]->Move(
                g_workspaces[sub_workspace].get());
    }
    return g_workspaces[name].get();
}

Workspace* ResetWorkspace(Workspace_t ws) {
    CHECK(ws) << "\nGiven workspace is invalid.";
    return ResetWorkspace(ws->name());
}

void MoveWorkspace(
    Workspace_t                     dst,
    Workspace_t                     src) {
    std::unique_lock<std::mutex> lock(g_mutex);
    CHECK(src) << "\nGiven source workspace is invalid.";
    CHECK(dst) << "\nGiven destination workspace is invalid.";
    dst->Move(src);
    sub_workspaces[dst->name()].push_back(src->name());
    LOG(INFO) << "Move the Workspace(" << src->name() << ") "
        << "into the Workspace(" << dst->name() << ").";
}

void DestroyWorkspace(const std::string& name) {
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

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Graph                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

std::string CreateGraph(
    const GraphDef_t                graph_def,
    const Device&                   device,
    Workspace_t                     ws) {
    auto graph_def_copy(*graph_def);
    // Overwritten device options
    DeviceOption* device_option = graph_def_copy.mutable_device_option();
    device_option->set_device_type((DeviceType)device.device_type());
    device_option->set_device_id(device.device_id());
    device_option->set_engine("CUDNN");
    auto* graph = ws->CreateGraph(graph_def_copy);
    if (!graph) LOG(FATAL) << "Can not create the graph.";
    return graph->name();
}

std::string CreateGraph(
    const std::string&              graph_file,
    const Device&                   device,
    Workspace_t                     ws) {
    GraphDef graph_def;
    ParseProtoFromText(graph_file.c_str(), &graph_def);
    return CreateGraph(&graph_def, device, ws);
}

void RunGraph(
    const std::string&              graph_name,
    Workspace_t                     ws,
    const int                       stream_id) {
    ws->RunGraph(graph_name, "", "", stream_id);
}

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Tensor                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

void CreateTensor(
    const std::string&              name,
    Workspace_t                     ws) {
    ws->CreateTensor(name);
}

template <typename T>
T* FetchTensor(
    const std::string&              name,
    vector<int64_t>&                shape,
    Workspace_t                     ws,
    const bool                      copy) {
    if (!ws->HasTensor(name)){
        LOG(FATAL) << "Tensor(" << name << ")"
            << " doesn't exist, try to create it before.";
    }
    Tensor* tensor = ws->GetTensor(name);
    if (tensor->meta().id() == 0){
        LOG(FATAL) << "Tensor(" << name << ")"
            << " has not been computed yet";
    }
    shape = tensor->dims();
    if (copy) {
        auto nbytes = tensor->nbytes();
        void* data = malloc(nbytes);
        if (tensor->memory_state() == MixedMemory::STATE_AT_CUDA) {
            CUDAContext::Memcpy<CPUContext, CUDAContext>(
                nbytes, data, tensor->raw_data<CUDAContext>());
        } else {
            CPUContext::Memcpy<CPUContext, CPUContext>(
                nbytes, data, tensor->raw_data<CPUContext>());
        }
        return static_cast<T*>(data);
    } else {
        return const_cast<T*>(
            static_cast<const T*>(
                tensor->raw_data<CPUContext>()));
    }
}

template <typename T>
void FeedTensor(
    const std::string&              name,
    const vector<int64_t>&          shape,
    const T*                        data,
    const Device&                   device,
    Workspace_t                     ws) {
    Tensor* tensor = ws->CreateTensor(name);
    tensor->Reshape(shape);
    if (device.device_type() == 1) {
        CUDAContext context(device.device_id());
        context.SwitchToDevice();
        tensor->mutable_data<T, CUDAContext>();
        context.Memcpy<CUDAContext, CPUContext>(tensor->nbytes(),
                         tensor->raw_mutable_data<CUDAContext>(),
                                 static_cast<const void*>(data));
    } else if (device.device_type() == 0) {
        CPUContext context;
        tensor->mutable_data<T, CPUContext>();
        context.Memcpy<CPUContext, CPUContext>(tensor->nbytes(),
                         tensor->raw_mutable_data<CPUContext>(),
                                static_cast<const void*>(data));
    } else {
        LOG(FATAL) << "Unknown device type.";
    }
}

/* * * * * * * * * * * * * * * * * * * * *
*                                        *
*                 Proto                  *
*                                        *
* * * * * * * * * * * * * * * * * * * * */

DRAGON_API void CreateGraphDef(GraphDef_t* graph_def) {
    *graph_def = new GraphDef();
}

DRAGON_API void DestroyGraphDef(GraphDef_t graph_def) {
    if (graph_def) delete graph_def;
}

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                I / O                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

void LoadCaffeModel(
    const std::string&          model_file,
    Workspace_t                 ws) {
    NetParameter net_param;
    ReadProtoFromBinaryFile(model_file.c_str(), &net_param);
    std::string scope = "";
    LOG(INFO) << "Load Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: Caffe";
    for (int i = 0; i < net_param.layer_size(); i++){
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = scope + layer_name + "/param:";
        for (int j = 0; j < layer.blobs_size(); j++){
            string tensor_name = prefix + std::to_string(j);
            if (!ws->HasTensor(tensor_name))
                ws->CreateTensor(tensor_name);
            BlobProto blob = layer.blobs(j);
            vector<int64_t> dims;
            for (auto dim : blob.shape().dim()) dims.push_back(dim);
            Tensor* tensor = ws->GetTensor(tensor_name);
            std::stringstream DimString;
            if (dims.size() > 0) {
                tensor->Reshape(dims);
                CHECK_EQ(tensor->count(), blob.data_size())
                    << "Tensor(" << tensor_name << ") "
                    << "failed to load, except size:  "
                    << tensor->count() << ", loaded " << blob.data_size();
                DimString << tensor->DimString();
            }
            else{
                tensor->Reshape(vector<int64_t>(1, blob.data_size()));
                DimString << "(missing)";
            }
            float* Xdata = tensor->mutable_data<float, CPUContext>();
            for (int idx = 0; idx < blob.data_size(); idx++)
                Xdata[idx] = blob.data(idx);
            LOG(INFO) << "Tensor(" << tensor_name << ") "
                << "loaded, shape: " << DimString.str()
                << ", size: " << blob.data_size();
        }
    }
}

void LoadONNXModel(
    const std::string&              model_file,
    GraphDef_t                      init_graph,
    GraphDef_t                      pred_graph,
    std::vector<std::string>&       inputs,
    std::vector<std::string>&       outputs) {
    LOG(INFO) << "Load Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: ONNX";
    onnx::ONNXBackend onnx_backend;
    onnx_backend.Prepare(model_file, init_graph, pred_graph);
    inputs.clear(); outputs.clear();
    for (const auto& e : pred_graph->input()) inputs.emplace_back(e);
    for (const auto& e : pred_graph->output()) outputs.emplace_back(e);
}

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Config                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

void SetLogLevel(const std::string& level) {
    SetLogDestination(StrToLogSeverity(level));
}

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *               Template                *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

template DRAGON_API float16* FetchTensor<float16>(
    const std::string&, std::vector<int64_t>&,
    Workspace_t, const bool);

template DRAGON_API float* FetchTensor<float>(
    const std::string&, std::vector<int64_t>&,
    Workspace_t, const bool);

template DRAGON_API uint8_t* FetchTensor<uint8_t>(
    const std::string&, std::vector<int64_t>&,
    Workspace_t, const bool);

template DRAGON_API int* FetchTensor<int>(
    const std::string&, std::vector<int64_t>&,
    Workspace_t, const bool);

template DRAGON_API int64_t* FetchTensor<int64_t>(
    const std::string&, std::vector<int64_t>&,
    Workspace_t, const bool);

template DRAGON_API void FeedTensor<float16>(
    const std::string&, const std::vector<int64_t>&,
    const float16*, const Device&, Workspace_t);

template DRAGON_API void FeedTensor<float>(
    const std::string&, const std::vector<int64_t>&,
    const float*, const Device&, Workspace_t);

template DRAGON_API void FeedTensor<uint8_t>(
    const std::string&, const std::vector<int64_t>&,
    const uint8_t*, const Device&, Workspace_t);

template DRAGON_API void FeedTensor<int>(
    const std::string&, const std::vector<int64_t>&,
    const int*, const Device&, Workspace_t);

template DRAGON_API void FeedTensor<int64_t>(
    const std::string&, const std::vector<int64_t>&,
    const int64_t*, const Device&, Workspace_t);

}  // namespace dragon