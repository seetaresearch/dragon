// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_CAFFEMODEL_H_
#define DRAGON_UTILS_CAFFEMODEL_H_

#include <fcntl.h>
#include <unistd.h>
#include <fstream>

#include <google/protobuf/message.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "protos/caffemodel.pb.h"
#include "core/workspace.h"
#include "utils/string.h"
#include "utils/logging.h"

namespace dragon {

using google::protobuf::Message;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;

inline void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
    std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    proto.SerializeToOstream(&output);
}

inline bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
#ifdef _MSC_VER
    int fd = _open(filename, O_RDONLY | O_BINARY);
#else
    int fd = open(filename, O_RDONLY);
#endif
    ZeroCopyInputStream *raw_input = new FileInputStream(fd);
    CodedInputStream *coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(INT_MAX, -1);
    bool success = proto->ParseFromCodedStream(coded_input);
    delete raw_input;
    delete coded_input;
    close(fd);
    return success;
}

inline void LoadCaffeModel(string file, Workspace* ws) {
    NetParameter net_param;
    ReadProtoFromBinaryFile(file.c_str(), &net_param);
    LOG(INFO) << "Restore From Model @: " << file << "......";
    LOG(INFO) << "Model Format: CaffeModel";
    for (int i = 0; i < net_param.layer_size(); i++) {
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = layer_name + "@param";
        for (int j = 0; j < layer.blobs_size(); j++) {
            string tensor_name = prefix + dragon_cast<string, int>(j);
            if (!ws->HasTensor(tensor_name))
                LOG(WARNING) << "Tensor(" << tensor_name << ") "
                << "does not exist in any Graphs, skip.";
            else{
                BlobProto blob = layer.blobs(j);
                vector<TIndex> dims;
                for (auto dim : blob.shape().dim()) dims.push_back(dim);
                Tensor* tensor = ws->GetTensor(tensor_name);
                std::stringstream dim_string;
                if (dims.size() > 0) {
                    tensor->Reshape(dims);
                    CHECK_EQ(tensor->count(), blob.data_size())
                        << "\nTensor(" << tensor_name << ") "
                        << "failed to load, except size:  "
                        << tensor->count() << ", loaded: " << blob.data_size();
                    dim_string << tensor->dim_string();
                } else {
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
}

inline void SavaCaffeModel(string file, const vector<Tensor*>& tensors) {
    NetParameter net_param;
    Map<string, int> layer_hash;
    int layer_idx = -1;
    for (int i = 0; i < tensors.size(); i++) {
        if (tensors[i]->count() <= 0) continue;
        vector<string> splits = SplitString(tensors[i]->name(), "@");
        if (layer_hash.count(splits[0]) == 0) {
            layer_hash[splits[0]] = ++layer_idx;
            LayerParameter* layer = net_param.add_layer();
            layer->set_name(splits[0]);
        }
        BlobProto* blob = net_param.mutable_layer(layer_idx)->add_blobs();
        for (auto dim : tensors[i]->dims()) blob->mutable_shape()->add_dim(dim);
        const float* Xdata = tensors[i]->data < float, CPUContext >();
        for (int id = 0; id < tensors[i]->count(); id++)
            blob->mutable_data()->Add(Xdata[id]);
    }
    std::fstream output(file, std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(net_param.SerializeToOstream(&output));
    LOG(INFO) << "Save the model @: " << file << "......";
    LOG(INFO) << "Model format: caffemodel";
}

}    // namespace dragon

#endif    // DRAGON_UTILS_CAFFEMODEL_H_