#include <fcntl.h>
#include <unistd.h>
#include <mutex>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "dragon.h"
#include "core/common.h"
#include "core/workspace.h"
#include "utils/caffemodel.h"

namespace dragon {

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
        << "\nWorkspace(" << name << ") does not exist, can not be reset.";
    LOG(INFO) << "Reset the Workspace(" << name << ").";
    g_workspaces[name].reset(new Workspace(name));
    for (auto& sub_workspace : sub_workspaces[name]) {
        if (g_workspaces.count(sub_workspace) > 0)
            g_workspaces[name]->MoveWorkspace(
                g_workspaces[sub_workspace].get());
    }
    return g_workspaces[name].get();
}

void ReleaseWorkspace(const std::string& name) {
    std::unique_lock<std::mutex> lock(g_mutex);
    CHECK(g_workspaces.count(name))
        << "\nWorkspace(" << name << ") does not exist, can not be released.";
    LOG(INFO) << "Release the Workspace(" << name << ").";
    g_workspaces[name].reset();
    g_workspaces.erase(name);
}

void MoveWorkspace(
    Workspace*                  target_ws,
    Workspace*                  source_ws) {
    std::unique_lock<std::mutex> lock(g_mutex);
    CHECK(source_ws) << "\nThe given source workspace is invalid."; 
    CHECK(target_ws) << "\nThe given target workspace is invalid.";
    target_ws->MoveWorkspace(source_ws);
    sub_workspaces[target_ws->name()].push_back(string(source_ws->name()));
    LOG(INFO) << "Move the Workspace(" << source_ws->name() << ") "
              << "into the Workspace(" << target_ws->name() << ").";
}

std::string CreateGraph(
    const std::string&          graph_file,
    Workspace*                  ws) {
    GraphDef meta_graph;
    int fd = open(graph_file.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "\nFile not found: " << graph_file;
    google::protobuf::io::FileInputStream* input = 
        new google::protobuf::io::FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, &meta_graph);
    delete input;
    close(fd);
    if (!success) LOG(FATAL) << "Invalid graph file for Dragon.";
    //  overwritten device options
    dragon::GraphBase* graph = ws->CreateGraph(meta_graph);
    if (!graph) LOG(FATAL) << "Can not create the graph.";
    return meta_graph.name();
}

std::string CreateGraph(
    const std::string&          graph_file,
    const Device&               device,
    Workspace*                  ws) {
    GraphDef meta_graph;
    int fd = open(graph_file.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "\nFile not found: " << graph_file;
    google::protobuf::io::FileInputStream* input = 
        new google::protobuf::io::FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, &meta_graph);
    delete input;
    close(fd);
    if (!success) LOG(FATAL) << "Invalid graph file for Dragon.";
    //  overwritten device options
    DeviceOption* device_option = meta_graph.mutable_device_option();
    device_option->set_device_type((DeviceType)device.device_type());
    device_option->set_device_id(device.device_id());
    device_option->set_engine("CUDNN");
    dragon::GraphBase* graph = ws->CreateGraph(meta_graph);
    if (!graph) LOG(FATAL) << "Can not create the graph.";
    return meta_graph.name();
}

void CreateTensor(
    const std::string&          name,
    Workspace*                  ws) {
    ws->CreateTensor(name);
}

template <typename T>
void FeedTensor(
    const std::string&          name,
    const vector<TIndex>&       shape,
    const T*                    data,
    const Device&               device,
    Workspace*                  ws) {
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

void TransplantCaffeModel(
    const std::string&          input_model,
    const std::string&          output_model) {
    TensorProtos protos;
    NetParameter net_param;
    ReadProtoFromBinaryFile(input_model.c_str(), &net_param);
    for (int i = 0; i < net_param.layer_size(); i++) {
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = layer_name + "/param:";
        for (int j = 0; j < layer.blobs_size(); j++) {
            string tensor_name = prefix + dragon_cast<string, int>(j);
            BlobProto blob = layer.blobs(j);
            TensorProto* proto = protos.add_protos();
            proto->set_data_type(TensorProto_DataType_FLOAT);
            proto->set_name(tensor_name);
            vector<TIndex> dims;
            for (auto dim : blob.shape().dim()) {
                proto->add_dims(dim);
                dims.push_back(dim);
            }
            for (auto data : blob.data()) proto->add_float_data(data);
            Tensor fake_tensor; fake_tensor.Reshape(dims);
            LOG(INFO) << "Tensor(" << tensor_name << ") "
                << "transplanted, shape: " << fake_tensor.DimString()
                << ", size: " << blob.data_size();
        }
    }
    std::fstream output(output_model,
        std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(protos.SerializeToOstream(&output));
    LOG(INFO) << "save the model @: " << output_model << "......";
    LOG(INFO) << "model format: DragonMoel";
}

void LoadDragonmodel(
    const std::string&          model_file,
    Workspace*                  ws){
    TensorProtos tensors;
    ReadProtoFromBinaryFile(model_file.c_str(), &tensors);
    LOG(INFO) << "Restore From Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: DragonModel";
    for (int i = 0; i < tensors.protos_size(); i++) {
        const TensorProto& proto = tensors.protos(i);
        const string& tensor_name = proto.name();
        if (!ws->HasTensor(tensor_name)) ws->CreateTensor(tensor_name);
        vector<TIndex> dims;
        for (auto dim : proto.dims()) dims.push_back(dim);
        Tensor* tensor = ws->GetTensor(tensor_name);
        std::stringstream DimString;
        if (dims.size() > 0) {
            tensor->Reshape(dims);
            CHECK_EQ(tensor->count(), proto.float_data_size())
                    << "Tensor(" << tensor_name << ") "
                    << "failed to load, except size:  "
                    << tensor->count() << ", loaded " << proto.float_data_size();
                DimString << tensor->DimString();
        } else{
            tensor->Reshape(vector<TIndex>(1, proto.float_data_size()));
            DimString << "(missing)";
        }
        float* Xdata = tensor->mutable_data<float, CPUContext>();
        for (int idx = 0; idx < proto.float_data_size(); idx++) 
            Xdata[idx] = proto.float_data(idx);
        LOG(INFO) << "Tensor(" << tensor_name << ") "
                  << "loaded, shape: " << DimString.str()
                  << ", size: " << proto.float_data_size();
    }
}

void LoadCaffemodel(
    const std::string&          model_file,
    Workspace*                  ws){
    NetParameter net_param;
    ReadProtoFromBinaryFile(model_file.c_str(), &net_param);
    std::string scope = "";
    LOG(INFO) << "Restore From Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: CaffeModel";
    for (int i = 0; i < net_param.layer_size(); i++){
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = scope + layer_name + "/param:";
        for (int j = 0; j < layer.blobs_size(); j++){
            string tensor_name = prefix + dragon_cast<string, int>(j);
            if (!ws->HasTensor(tensor_name))
                ws->CreateTensor(tensor_name);
            BlobProto blob = layer.blobs(j);
            vector<TIndex> dims;
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
                tensor->Reshape(vector<TIndex>(1, blob.data_size()));
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

void RunGraph(
    const std::string&          graph_name,
    Workspace*                  ws) {
    ws->RunGraph(graph_name, "", "");
}

template <typename T>
T* FetchTensor(
    const std::string&          name,
    vector<TIndex>&             shape,
    Workspace*                  ws){
    if (!ws->HasTensor(name)){
        LOG(FATAL) << "Tensor(" << name << ")"
            << " doesn't exist, try create it before.";
    }
    Tensor* tensor = ws->GetTensor(name);
    if (tensor->meta().id() == 0){
        LOG(FATAL) << "Tensor(" << name << ")"
            << " has not been computed yet";
    }
    shape = tensor->dims();
    void* data = malloc(tensor->nbytes());
    if (tensor->memory_state() == MixedMemory::STATE_AT_CUDA) {
        CUDAContext::Memcpy<CPUContext, CUDAContext>(
            tensor->nbytes(), data, tensor->raw_data<CUDAContext>());
    } else {
        CPUContext::Memcpy<CPUContext, CPUContext>(
            tensor->nbytes(), data, tensor->raw_data<CPUContext>());
    }
    return static_cast<T*>(data);
}

void SetLogLevel(const std::string& level) {
    SetLogDestination(StrToLogSeverity(level));
}

template float* FetchTensor<float>(
    const std::string&,
    std::vector<TIndex>&,
    Workspace*);

template void FeedTensor<float>(
    const std::string&,
    const std::vector<TIndex>&,
    const float*,
    const Device&,
    Workspace*);

template void FeedTensor<int>(
    const std::string&,
    const std::vector<TIndex>&,
    const int*,
    const Device&,
    Workspace*);

template void FeedTensor<uint8_t>(
    const std::string&,
    const std::vector<TIndex>&,
    const uint8_t*,
    const Device&,
    Workspace*);

}    // namespace dragon