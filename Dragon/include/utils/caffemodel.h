/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_CAFFEMODEL_H_
#define DRAGON_UTILS_CAFFEMODEL_H_

#include "core/workspace.h"
#include "proto/caffemodel.pb.h"

namespace dragon {

inline void LoadCaffeModel(
    string                          file,
    Workspace*                      ws) {
    NetParameter net_param;
    ReadProtoFromBinaryFile(file.c_str(), &net_param);
    LOG(INFO) << "Restore From Model @: " << file << "......";
    LOG(INFO) << "Model Format: CaffeModel";
    for (int i = 0; i < net_param.layer_size(); i++) {
        const LayerParameter& layer = net_param.layer(i);
        const string& layer_name = layer.name();
        string prefix = layer_name + "/param:";
        for (int j = 0; j < layer.blobs_size(); j++) {
            string tensor_name = prefix + std::to_string(j);
            if (!ws->HasTensor(tensor_name))
                LOG(WARNING) << "Tensor(" << tensor_name << ") "
                << "does not exist in any Graphs, skip.";
            else{
                BlobProto blob = layer.blobs(j);
                vector<int64_t> dims;
                for (auto dim : blob.shape().dim()) dims.push_back(dim);
                Tensor* tensor = ws->GetTensor(tensor_name);
                std::stringstream DimString;
                if (dims.size() > 0) {
                    tensor->Reshape(dims);
                    CHECK_EQ(tensor->count(), blob.data_size())
                        << "\nTensor(" << tensor_name << ") "
                        << "failed to load, except size:  "
                        << tensor->count()
                        << ", loaded: " << blob.data_size();
                    DimString << tensor->DimString();
                } else {
                    tensor->Reshape({ blob.data_size() });
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
}

inline void SavaCaffeModel(
    string                          file,
    const vector<Tensor*>&          tensors) {
    NetParameter net_param;
    Map<string, int> layer_hash;
    int layer_idx = -1;
    for (int i = 0; i < tensors.size(); i++) {
        if (tensors[i]->count() <= 0) continue;
        vector<string> splits = str::split(
            tensors[i]->name(), "/param:");
        if (layer_hash.count(splits[0]) == 0) {
            layer_hash[splits[0]] = ++layer_idx;
            LayerParameter* layer = net_param.add_layer();
            layer->set_name(splits[0]);
        }
        BlobProto* blob = net_param.mutable_layer(layer_idx)->add_blobs();
        for (auto dim : tensors[i]->dims()) blob->mutable_shape()->add_dim(dim);
        if (XIsType((*tensors[i]), float)) {
            auto* Xdata = tensors[i]->data<float, CPUContext>();
            for (int id = 0; id < tensors[i]->count(); id++)
                blob->mutable_data()->Add(Xdata[id]);
        } else if (XIsType((*tensors[i]), float16)) {
            auto* Xdata = tensors[i]->data<float16, CPUContext>();
            for (int id = 0; id < tensors[i]->count(); id++)
                blob->mutable_data()->Add(
                    cast::to<float>(Xdata[id]));
        }
    }
    WriteProtoToBinaryFile(net_param, file.c_str());
    LOG(INFO) << "Save the model @: " << file << "......";
    LOG(INFO) << "Model format: Caffe";
}

}  // namespace dragon

#endif  // DRAGON_UTILS_CAFFEMODEL_H_