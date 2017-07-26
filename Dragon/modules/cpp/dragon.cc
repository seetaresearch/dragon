#include <fcntl.h>
#include <unistd.h>
#include <mutex>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "dragon.h"
#include "protos/dragon.pb.h"
#include "core/common.h"
#include "core/workspace.h"
#include "utils/caffemodel.h"

namespace dragon {

std::unordered_map<std::string, std::shared_ptr < Workspace > > g_workspaces;
std::mutex g_mutex;

Workspace* CreateWorkspace(const std::string& name){
    std::unique_lock<std::mutex> lock(g_mutex);
    if (g_workspaces.count(name)) return g_workspaces[name].get();
    std::shared_ptr<Workspace> new_workspace(new Workspace());
    g_workspaces[name] = new_workspace;
    return new_workspace.get();
}

void CreateGraph(const std::string& graph_file, Workspace* ws){
    GraphDef graph_def;
    int fd = open(graph_file.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << graph_file;
    google::protobuf::io::FileInputStream* input = 
        new google::protobuf::io::FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    close(fd);
    if (!success) LOG(FATAL) << "Invalid graph file for Dragon.";
    dragon::GraphBase* graph = ws->CreateGraph(graph_def);
    if (!graph) LOG(FATAL) << "Can not create the graph.";
}

void CreateTensor(const std::string& name, Workspace* ws){
    ws->CreateTensor(name);
}

template <typename T>
void FeedTensor(const std::string& name,
                const vector<TIndex>& shape,
                const T* data, Workspace* ws){
    Tensor* tensor = ws->CreateTensor(name);
    tensor->Reshape(shape);
    tensor->mutable_data<T, CUDAContext>();
    CUDAContext context;
    context.SwitchToDevice();
    context.Memcpy<CUDAContext, CPUContext>(tensor->nbytes(),
        tensor->raw_mutable_data<CUDAContext>(), static_cast<const void*>(data));
}

void TransplantCaffeModel(const std::string& input_model, const std::string& output_model) {
    TensorProtos protos;
    NetParameter net_param;
    ReadProtoFromBinaryFile(input_model.c_str(), &net_param);
    for (int i = 0; i < net_param.layer_size(); i++) {
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = layer_name + "@param";
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
                << "transplanted, shape: " << fake_tensor.dim_string()
                << ", size: " << blob.data_size();
        }
    }
    std::fstream output(output_model, std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(protos.SerializeToOstream(&output));
    LOG(INFO) << "save the model @: " << output_model << "......";
    LOG(INFO) << "model format: DragonMoel";
}

void LoadDragonmodel(const std::string& model_file, Workspace* ws){
    TensorProtos tensors;
    ReadProtoFromBinaryFile(model_file.c_str(), &tensors);
    LOG(INFO) << "Restore From Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: DragonModel";
    for (int i = 0; i < tensors.protos_size(); i++){
        const TensorProto& proto = tensors.protos(i);
        const string& tensor_name = proto.name();
        if (!ws->HasTensor(tensor_name)) ws->CreateTensor(tensor_name);
        vector<TIndex> dims;
        for (auto dim : proto.dims()) dims.push_back(dim);
        Tensor* tensor = ws->GetTensor(tensor_name);
        std::stringstream dim_string;
        if (dims.size() > 0) {
            tensor->Reshape(dims);
            CHECK_EQ(tensor->count(), proto.float_data_size())
                    << "Tensor(" << tensor_name << ") "
                    << "failed to load, except size:  "
                    << tensor->count() << ", loaded " << proto.float_data_size();
                dim_string << tensor->dim_string();
        } else{
            tensor->Reshape(vector<TIndex>(1, proto.float_data_size()));
            dim_string << "(missing)";
        }
        float* Xdata = tensor->mutable_data<float, CPUContext>();
        for (int idx = 0; idx < proto.float_data_size(); idx++) 
            Xdata[idx] = proto.float_data(idx);
        LOG(INFO) << "Tensor(" << tensor_name << ") "
                  << "loaded, shape: " << dim_string.str()
                  << ", size: " << proto.float_data_size();
    }
}

void LoadCaffemodel(const std::string& model_file, Workspace* ws){
    NetParameter net_param;
    ReadProtoFromBinaryFile(model_file.c_str(), &net_param);
    std::string scope = "";
    LOG(INFO) << "Restore From Model @: " << model_file << "......";
    LOG(INFO) << "Model Format: CaffeModel";
    for (int i = 0; i < net_param.layer_size(); i++){
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = scope + layer_name + "@param";
        for (int j = 0; j < layer.blobs_size(); j++){
            string tensor_name = prefix + dragon_cast<string, int>(j);
            if (!ws->HasTensor(tensor_name))
                ws->CreateTensor(tensor_name);
            BlobProto blob = layer.blobs(j);
            vector<TIndex> dims;
            for (auto dim : blob.shape().dim()) dims.push_back(dim);
            Tensor* tensor = ws->GetTensor(tensor_name);
            std::stringstream dim_string;
            if (dims.size() > 0) {
                tensor->Reshape(dims);
                CHECK_EQ(tensor->count(), blob.data_size())
                    << "Tensor(" << tensor_name << ") "
                    << "failed to load, except size:  "
                    << tensor->count() << ", loaded " << blob.data_size();
                dim_string << tensor->dim_string();
            }
            else{
                tensor->Reshape(vector<TIndex>(1, blob.data_size()));
                dim_string << "(missing)";
            }
            float* Xdata = tensor->mutable_data<float, CPUContext>();
            for (int idx = 0; idx < blob.data_size(); idx++)
                Xdata[idx] = blob.data(idx);
            LOG(INFO) << "Tensor(" << tensor_name << ") "
                << "loaded, shape: " << dim_string.str()
                << ", size: " << blob.data_size();
        }
    }
}

void RunGraph(const std::string& graph_name, Workspace* ws){
    ws->RunGraph(graph_name, "", "");
}

template <typename T>
T* FetchTensor(const std::string& name,
               vector<TIndex>& shape, 
               Workspace* ws){
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
        CUDAContext::Memcpy<CPUContext, CUDAContext>(tensor->nbytes(), 
            data, tensor->raw_data<CUDAContext>());
    }else{
        CPUContext::Memcpy<CPUContext, CPUContext>(tensor->nbytes(),
            data, tensor->raw_data<CPUContext>());
    }
    return static_cast<T*>(data);
}

void SetLogLevel(const std::string& level) {
    SetLogDestination(StrToLogSeverity(level));
}

}    // namespace dragon